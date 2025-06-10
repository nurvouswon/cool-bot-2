import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybaseball import statcast
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# --- EVENTS TO INCLUDE ---
ALLOWED_EVENTS = {'single', 'double', 'triple', 'home_run', 'field_out'}

st.title("⚾ Statcast MLB HR Analyzer — Event/Player Rolling Features + Park/Handed/PitchType Context")

#### --- 1. Data Source Section --- ####
data_source = st.radio("Select data source:", ["Upload CSV", "Fetch new data from MLB Statcast (pybaseball)"])
uploaded_file = None
start_date = None
end_date = None

if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a Statcast batted ball events CSV", type="csv")
elif data_source == "Fetch new data from MLB Statcast (pybaseball)":
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=datetime.today() - timedelta(days=7), key="start")
    with col2:
        end_date = st.date_input("End date", value=datetime.today(), key="end")

# ---- Park HR Rate, Handed HR Rate, Pitch Type HR Rate supplemental CSVs ----
st.markdown("#### (Optional) Upload supplemental rate CSVs for context features (park/handedness/pitch type):")
park_hr_file = st.file_uploader("Park HR Rates CSV", type="csv", key="park")
handed_hr_file = st.file_uploader("Handedness HR Rates CSV", type="csv", key="handed")
pitchtype_hr_file = st.file_uploader("Pitch Type HR Rates CSV", type="csv", key="pitchtype")

#### --- 2. Load/Filter Data --- ####
@st.cache_data(show_spinner="Loading Statcast data...")
def load_statcast_data(start_date, end_date):
    df = statcast(start_dt=start_date.strftime("%Y-%m-%d"), end_dt=end_date.strftime("%Y-%m-%d"))
    return df

if data_source == "Upload CSV" and uploaded_file is not None:
    df = pd.read_csv(uploaded_file, low_memory=False)
elif data_source == "Fetch new data from MLB Statcast (pybaseball)" and start_date and end_date:
    with st.spinner(f"Fetching events {start_date} to {end_date}..."):
        df = load_statcast_data(start_date, end_date)
else:
    df = None

if df is not None and not df.empty:
    # Filter to desired events
    if "events" in df.columns:
        original_rows = len(df)
        df = df[df["events"].isin(ALLOWED_EVENTS)].copy()
        st.info(f"Loaded {len(df):,} batted ball events from {original_rows:,} rows ({start_date} to {end_date})")
    else:
        st.error("CSV missing 'events' column. Please upload full statcast export with events.")
        st.stop()
else:
    st.stop()

#### --- 3. Rolling Feature Engineering --- ####
st.markdown("### Engineering Rolling Features (with progress bar)")
progress = st.progress(0)

# The features you want rolling stats for:
BATTED_BALL_METRICS = [
    "launch_speed", "launch_angle", "hit_distance_sc", "woba_value", "iso_value", "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle", "estimated_slg_using_speedangle", "xwoba"
]
ROLLING_WINDOWS = [3, 5, 7, 14]

def add_rolling_features(df):
    df = df.copy()
    # Ensure game_date is datetime
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])
    else:
        st.error("No game_date column in data. Please check your input.")
        st.stop()

    df = df.sort_values(["batter", "game_date", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    for metric in BATTED_BALL_METRICS:
        if metric not in df.columns:
            df[metric] = np.nan

    # Add rolling stats for each batter
    for i, metric in enumerate(BATTED_BALL_METRICS):
        for w in ROLLING_WINDOWS:
            roll_col = f"B_{metric}_{w}"
            df[roll_col] = (
                df.groupby("batter")[metric]
                .rolling(window=w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
        progress.progress((i+1)/len(BATTED_BALL_METRICS)/2, text=f"Batter rolling: {metric}")

    # Pitcher rolling features
    if "pitcher" in df.columns:
        for i, metric in enumerate(BATTED_BALL_METRICS):
            for w in ROLLING_WINDOWS:
                roll_col = f"P_{metric}_{w}"
                df[roll_col] = (
                    df.groupby("pitcher")[metric]
                    .rolling(window=w, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
            progress.progress(0.5 + (i+1)/len(BATTED_BALL_METRICS)/2, text=f"Pitcher rolling: {metric}")

    return df

df = add_rolling_features(df)
progress.progress(1.0, text="Rolling features complete!")

st.success("Feature engineering complete.")

#### --- 4. Park, Handedness, Pitch Type Rate Merge --- ####

# Park HR Rate (by park name, try to merge to 'home_team' or 'park' or similar)
if park_hr_file is not None:
    park_df = pd.read_csv(park_hr_file)
    # Guess the right join key
    park_key = [k for k in ["park", "stadium", "home_team"] if k in park_df.columns]
    join_key = park_key[0] if park_key else park_df.columns[0]
    if 'park' in df.columns:
        df = df.merge(park_df.rename(columns={join_key: "park"}), on="park", how="left")
    elif 'home_team' in df.columns:
        df = df.merge(park_df.rename(columns={join_key: "home_team"}), on="home_team", how="left")
    # Rename rate column to ParkHRRate
    if "ParkHRRate" not in df.columns:
        rate_col = [c for c in park_df.columns if "rate" in c][0]
        df = df.rename(columns={rate_col: "ParkHRRate"})
else:
    df["ParkHRRate"] = 1.0

# Handedness HR Rate (batter/pitcher handedness)
if handed_hr_file is not None:
    handed_df = pd.read_csv(handed_hr_file)
    # Ensure column names
    rename_map = {"stand": "batter_stand", "p_throws": "pitcher_hand"}
    for k,v in rename_map.items():
        if k in handed_df.columns: handed_df.rename(columns={k:v}, inplace=True)
    join_cols = [c for c in handed_df.columns if "batter" in c and "hand" in c] + [c for c in handed_df.columns if "pitcher" in c and "hand" in c]
    join_cols = join_cols[:2]
    # Fill missing
    for jc in join_cols:
        if jc not in df.columns:
            df[jc] = df.get(jc, "R")
    df = df.merge(handed_df, on=join_cols, how="left")
    # Rename to HandedHRRate
    for col in handed_df.columns:
        if "rate" in col:
            df.rename(columns={col: "HandedHRRate"}, inplace=True)
else:
    df["HandedHRRate"] = 1.0

# PitchType HR Rate (by event pitch_type)
if pitchtype_hr_file is not None:
    pitch_df = pd.read_csv(pitchtype_hr_file)
    join_col = "pitch_type" if "pitch_type" in df.columns else pitch_df.columns[0]
    df = df.merge(pitch_df.rename(columns={join_col: "pitch_type"}), on="pitch_type", how="left")
    for col in pitch_df.columns:
        if "rate" in col:
            df.rename(columns={col: "PitchTypeHRRate"}, inplace=True)
else:
    df["PitchTypeHRRate"] = 1.0

#### --- 5. Downloadable CSVs --- ####

player_level_features = [f"B_{metric}_{w}" for metric in BATTED_BALL_METRICS for w in ROLLING_WINDOWS] + \
                        [f"P_{metric}_{w}" for metric in BATTED_BALL_METRICS for w in ROLLING_WINDOWS] + \
                        ["ParkHRRate", "HandedHRRate", "PitchTypeHRRate"]

# Player-level (as of latest event per batter)
player_df = (
    df.sort_values("game_date")
    .groupby("batter")
    .tail(1)
    .reset_index(drop=True)
)

# Event-level export
event_cols = ["batter", "pitcher", "game_date", "events", "hit_distance_sc", "launch_speed", "launch_angle",
              "ParkHRRate", "HandedHRRate", "PitchTypeHRRate"] + player_level_features
if "home_run" in df.columns or "hr_outcome" in df.columns:
    if "home_run" in df.columns:
        df["hr_outcome"] = (df["events"] == "home_run").astype(int)
    event_cols.append("hr_outcome")

st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
event_csv = df[event_cols].to_csv(index=False).encode()
st.download_button("⬇️ Download Event-Level Feature CSV", data=event_csv, file_name="event_level_features.csv")

st.markdown("#### Download Player-Level CSV (1 row per batter):")
player_csv = player_df[["batter"] + player_level_features].to_csv(index=False).encode()
st.download_button("⬇️ Download Player-Level Rolling Feature CSV", data=player_csv, file_name="player_level_features.csv")

#### --- 6. Logistic Regression: Only Numeric & Relevant Features --- ####
st.markdown("### Logistic Regression Weights (Standardized Features)")
if "hr_outcome" not in df.columns:
    st.warning("No HR outcome labels found in your data. Logistic regression skipped.")
else:
    X_cols = [col for col in player_level_features if col in df.columns]
    X = df[X_cols]
    y = df["hr_outcome"]
    # Remove any non-numeric columns
    X = X.select_dtypes(include=[np.number]).fillna(0)
    # Only use columns with > 0 variance
    X = X.loc[:, X.std() > 0]
    X_cols = X.columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=500)
    model.fit(X_scaled, y)
    weights = pd.Series(model.coef_[0], index=X_cols).sort_values(ascending=False)
    weights_df = pd.DataFrame({'feature': weights.index, 'weight': weights.values})
    st.dataframe(weights_df.head(30))
    auc = roc_auc_score(y, model.predict_proba(X_scaled)[:, 1])
    st.caption(f"ROC AUC: **{auc:.3f}**")
    st.markdown("#### Download Weights CSV:")
    st.download_button("⬇️ Download Logistic Weights CSV", data=weights_df.to_csv(index=False).encode(), file_name="logit_hr_weights.csv")

st.info("**Done!** All outputs can be used for custom modeling or HR leaderboards. Session is NOT reset after downloads.")
