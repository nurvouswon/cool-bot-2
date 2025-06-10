import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from pybaseball import statcast

st.set_page_config(layout="wide")

st.title("⚾ Statcast MLB HR Analyzer — Full Feature/Context Event + Player Level")

st.markdown("""
**1. Choose data source:**
- Upload a Statcast batted ball events CSV  
- **OR** fetch new data from MLB Statcast for your date range

**2. This app will:**
- Compute *all* advanced rolling features (3/5/7/14)
- Output both event-level and player-level CSVs
- Fit logistic regression (with scaling) and output feature weights
""")

WINDOWS = [3, 5, 7, 14]

# --- Data source selection ---
data_source = st.radio(
    "Select data source:",
    ["Upload CSV", "Fetch new data from MLB Statcast (pybaseball)"]
)

@st.cache_data(show_spinner="Fetching Statcast data from MLB...")
def get_statcast_df(start_date, end_date):
    df = statcast(start_dt=start_date.strftime("%Y-%m-%d"), end_dt=end_date.strftime("%Y-%m-%d"))
    return df

df = None
if data_source == "Upload CSV":
    csv_file = st.file_uploader("Upload your Statcast Batted Ball Events CSV", type=["csv"])
    if csv_file:
        df = pd.read_csv(csv_file)
elif data_source == "Fetch new data from MLB Statcast (pybaseball)":
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=datetime.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End date", value=datetime.today())
    if st.button("Fetch Statcast Data"):
        df = get_statcast_df(start_date, end_date)
        st.success(f"Loaded {len(df)} events from {start_date} to {end_date}")

# ================== Feature Engineering ===================
@st.cache_data(show_spinner="Engineering rolling features...")
def engineer_features(df_raw):
    df = df_raw.copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])
    else:
        st.error("Missing 'game_date' column in dataset.")
        st.stop()

    # Filter to only batted ball events in play
    if 'type' in df.columns:
        df = df[df['type'] == 'X']
    elif 'events' in df.columns:
        batted_ball_events = [
            'single', 'double', 'triple', 'home_run',
            'field_out', 'force_out', 'grounded_into_double_play', 'fielders_choice_out',
            'other_out', 'fielders_choice', 'double_play', 'triple_play'
        ]
        df = df[df['events'].str.lower().isin(batted_ball_events)]
    else:
        st.warning("Could not auto-detect batted ball event filter; review your data.")

    # HR tagging
    hr_events = ['home_run', 'home run', 'hr']
    if 'events' in df.columns:
        df['hr_outcome'] = df['events'].str.lower().isin(hr_events).astype(int)
    elif 'result' in df.columns:
        df['hr_outcome'] = df['result'].str.lower().isin(hr_events).astype(int)
    else:
        df['hr_outcome'] = 0

    # Standardize columns
    df['batter_id'] = df.get('batter', df.get('batter_id', np.nan))
    df['pitcher_id'] = df.get('pitcher', df.get('pitcher_id', np.nan))
    df['exit_velocity'] = df.get('launch_speed', df.get('exit_velocity', np.nan))
    df['launch_angle'] = df.get('launch_angle', np.nan)
    df['release_spin_rate'] = df.get('release_spin_rate', np.nan)
    df['release_speed'] = df.get('release_speed', np.nan)
    df['p_throws'] = df.get('p_throws', df.get('pitcher_hand', 'NA')).fillna('NA')
    df['stand'] = df.get('stand', df.get('batter_hand', 'NA')).fillna('NA')
    df['xwoba'] = (
        df['estimated_woba_using_speedangle']
        if 'estimated_woba_using_speedangle' in df.columns else
        (df['woba_value'] if 'woba_value' in df.columns else np.nan)
    )
    df['xba'] = df['estimated_ba_using_speedangle'] if 'estimated_ba_using_speedangle' in df.columns else np.nan

    # Rolling features for batter and pitcher
    def rolling_features(group, prefix, id_col, windows=WINDOWS):
        group = group.sort_values('game_date')
        feats = {}
        for w in windows:
            lastN = group.tail(w)
            pa = len(lastN)
            barrels = lastN[(lastN['exit_velocity'] > 95) & (lastN['launch_angle'].between(20, 35))].shape[0]
            sweet = lastN[lastN['launch_angle'].between(8, 32)].shape[0]
            hard = lastN[lastN['exit_velocity'] >= 95].shape[0]
            slg = np.nansum([
                (lastN['events'] == 'single').sum(),
                2 * (lastN['events'] == 'double').sum(),
                3 * (lastN['events'] == 'triple').sum(),
                4 * (lastN['events'] == 'home_run').sum()
            ]) if 'events' in lastN.columns else np.nan
            ab = sum((lastN['events'] == x).sum() for x in ['single', 'double', 'triple', 'home_run', 'field_out', 'force_out', 'other_out']) if 'events' in lastN.columns else np.nan
            single = (lastN['xba'] >= 0.5) & (lastN['launch_angle'] < 15)
            double = (lastN['xba'] >= 0.5) & (lastN['launch_angle'].between(15, 30))
            triple = (lastN['xba'] >= 0.5) & (lastN['launch_angle'].between(30, 40))
            hr = (lastN['launch_angle'] > 35) & (lastN['exit_velocity'] > 100)
            xsingle = single.sum()
            xdouble = double.sum()
            xtriple = triple.sum()
            xhr = hr.sum()
            xab = xsingle + xdouble + xtriple + xhr
            xslg = (1 * xsingle + 2 * xdouble + 3 * xtriple + 4 * xhr) / xab if xab else np.nan
            xba = lastN['xba'].mean() if 'xba' in lastN.columns and not lastN['xba'].isnull().all() else np.nan
            xiso = (xslg - xba) if (xslg is not np.nan and xba is not np.nan) else np.nan

            feats[f'{prefix}_BarrelRate_{w}'] = barrels / pa if pa else np.nan
            feats[f'{prefix}_EV_{w}'] = lastN['exit_velocity'].mean() if pa else np.nan
            feats[f'{prefix}_SLG_{w}'] = slg / ab if ab else np.nan
            feats[f'{prefix}_xSLG_{w}'] = xslg
            feats[f'{prefix}_xISO_{w}'] = xiso
            feats[f'{prefix}_xwoba_{w}'] = lastN['xwoba'].mean() if pa else np.nan
            feats[f'{prefix}_sweet_spot_pct_{w}'] = sweet / pa if pa else np.nan
            feats[f'{prefix}_hardhit_pct_{w}'] = hard / pa if pa else np.nan
        return pd.Series(feats)

    # Player-level features: one row per player as of last game in period
    player_progress = st.progress(0, text="Rolling features for batters/pitchers")
    batter_ids = df['batter_id'].unique()
    pitcher_ids = df['pitcher_id'].unique()
    batter_list, pitcher_list = [], []
    for idx, batter_id in enumerate(batter_ids):
        group = df[df['batter_id'] == batter_id]
        feats = rolling_features(group, 'B', 'batter_id')
        feats['batter_id'] = batter_id
        batter_list.append(feats)
        player_progress.progress((idx + 1) / (len(batter_ids) + len(pitcher_ids)), text=f"Batter {idx + 1}/{len(batter_ids)}")
    for idx, pitcher_id in enumerate(pitcher_ids):
        group = df[df['pitcher_id'] == pitcher_id]
        feats = rolling_features(group, 'P', 'pitcher_id')
        feats['pitcher_id'] = pitcher_id
        pitcher_list.append(feats)
        player_progress.progress((len(batter_ids) + idx + 1) / (len(batter_ids) + len(pitcher_ids)), text=f"Pitcher {idx + 1}/{len(pitcher_ids)}")
    batter_feats = pd.DataFrame(batter_list)
    pitcher_feats = pd.DataFrame(pitcher_list)
    player_progress.progress(1.0, text="Rolling features done.")

    # Event-level merge: assign each event the rolling window features at that date
    df = df.merge(batter_feats, on='batter_id', how='left', suffixes=('', '_dup1'))
    df = df.merge(pitcher_feats, on='pitcher_id', how='left', suffixes=('', '_dup2'))

    # Remove duplicate columns if merging adds them
    df = df.loc[:,~df.columns.duplicated()]

    # Context features
    # HR rates by hand
    if 'stand' in df.columns and 'p_throws' in df.columns:
        hand_hr = df.groupby(['stand', 'p_throws'])['hr_outcome'].mean().reset_index().rename(
            columns={'stand':'BatterHandedness','p_throws':'PitcherHandedness','hr_outcome':'HandedHRRate'}
        )
        df = df.merge(hand_hr, left_on=['stand','p_throws'], right_on=['BatterHandedness','PitcherHandedness'], how='left')
    # HR rates by pitch type
    if 'pitch_type' in df.columns:
        pitch_hr = df.groupby('pitch_type')['hr_outcome'].mean().reset_index().rename(
            columns={'hr_outcome':'PitchTypeHRRate'}
        )
        df = df.merge(pitch_hr, on='pitch_type', how='left')
    # HR rates by park
    if 'home_team' in df.columns:
        park_hr = df.groupby('home_team')['hr_outcome'].mean().reset_index().rename(
            columns={'home_team':'park','hr_outcome':'ParkHRRate'}
        )
        df = df.merge(park_hr, on='park', how='left')
    return df, batter_feats, pitcher_feats

# ===== MAIN LOGIC ======
if df is not None and not df.empty:
    with st.spinner("Engineering Rolling Features (with progress bar)..."):
        df, batter_feats, pitcher_feats = engineer_features(df)
    st.success(f"Rolling features done, preparing event-level and player-level CSVs...")

    # ---------------- EVENT LEVEL EXPORT ----------------
    st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
    event_cols = [c for c in df.columns if c not in ['game_date', 'batter', 'pitcher', 'events', 'result']]
    st.dataframe(df[event_cols + ['hr_outcome']].head(20))
    st.download_button(
        "⬇️ Download Event-Level CSV",
        data=df[event_cols + ['hr_outcome']].to_csv(index=False).encode(),
        file_name="event_level_features.csv"
    )

    # ---------------- PLAYER LEVEL EXPORT ----------------
    st.markdown("#### Download Player-Level CSV (rolling windows, 1 row per player):")
    st.dataframe(batter_feats.head(20))
    st.download_button(
        "⬇️ Download Player-Level (Batter) CSV",
        data=batter_feats.to_csv(index=False).encode(),
        file_name="player_level_batter_features.csv"
    )
    st.dataframe(pitcher_feats.head(20))
    st.download_button(
        "⬇️ Download Player-Level (Pitcher) CSV",
        data=pitcher_feats.to_csv(index=False).encode(),
        file_name="player_level_pitcher_features.csv"
    )

    # ---------------- LOGISTIC REGRESSION WEIGHTS ----------------
    st.markdown("#### Logistic Regression: Feature Weights (event-level, with scaling)")
    features_for_logit = [c for c in event_cols if df[c].dtype != 'O' and c != 'hr_outcome']
    X = df[features_for_logit].fillna(0)
    y = df['hr_outcome']
    if len(X) > 100 and y.nunique() == 2:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=1000, solver='lbfgs')
        model.fit(X_scaled, y)
        weights = pd.Series(model.coef_[0], index=features_for_logit).sort_values(ascending=False)
        weights_df = pd.DataFrame({'feature': weights.index, 'weight': weights.values})
        st.write(weights_df)
        auc = roc_auc_score(y, model.predict_proba(X_scaled)[:, 1])
        st.markdown(f"**In-sample AUC:** `{auc:.3f}`")
        st.download_button(
            "⬇️ Download Logistic Weights CSV",
            data=weights_df.to_csv(index=False).encode(),
            file_name="logit_feature_weights.csv"
        )
    else:
        st.warning("Not enough data for regression.")

else:
    st.info("Upload or generate data to start analysis.")
