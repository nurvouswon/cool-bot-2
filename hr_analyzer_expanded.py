import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from pybaseball import statcast

st.set_page_config(layout="wide")

st.title("⚾ Statcast MLB HR Analyzer — Full Feature/Context Event + Player Level")

st.markdown("""
**1. Choose data source:**
- Upload a Statcast batted ball events CSV  
- OR fetch new data from MLB Statcast for your date range

**2. This app will:**
- Compute all advanced rolling features (3/5/7/14)
- Output both event-level and player-level CSVs
- Fit logistic regression (with scaling) and output feature weights
""")

# ========== Utility: Deduplicate columns ==============
def dedup_columns(df):
    df = df.copy()
    seen = {}
    new_cols = []
    for col in df.columns:
        base = col
        count = seen.get(base, 0)
        if count == 0:
            new_cols.append(base)
        else:
            new_cols.append(f"{base}_{count}")
        seen[base] = count + 1
    df.columns = new_cols
    return df

# ========== Data Source Selection ===========
data_source = st.radio(
    "Select data source:",
    ["Upload CSV", "Fetch new data from MLB Statcast (pybaseball)"]
)
df = None

if data_source == "Upload CSV":
    csv_file = st.file_uploader("Upload your Statcast Batted Ball Events CSV", type=["csv"])
    if csv_file:
        df = pd.read_csv(csv_file)
elif data_source == "Fetch new data from MLB Statcast (pybaseball)":
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=datetime.today() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End date", value=datetime.today())
    if st.button("Fetch Statcast Data"):
        with st.spinner("Fetching Statcast data from MLB... This may take several minutes."):
            df = statcast(start_dt=start_date.strftime("%Y-%m-%d"), end_dt=end_date.strftime("%Y-%m-%d"))
        st.success(f"Loaded {len(df)} events from {start_date} to {end_date}")

# ========== Main Feature Engineering & Analysis ==========
if df is not None and not df.empty:
    # Standardize columns
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    # Filter batted ball events in play
    allowed_events = ['single', 'double', 'triple', 'home_run', 'field_out']
    if 'type' in df.columns and 'events' in df.columns:
        df = df[(df['type'] == 'X') & (df['events'].str.lower().isin(allowed_events))]
    elif 'events' in df.columns:
        df = df[df['events'].str.lower().isin(allowed_events)]
    else:
        st.error("No 'events' column detected. Cannot filter to batted ball events.")
        st.stop()

    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])
    else:
        st.error("Missing 'game_date' column in dataset.")
        st.stop()

    # Tag HR outcomes
    hr_events = ['home_run', 'home run', 'hr']
    df['hr_outcome'] = df['events'].str.lower().isin(hr_events).astype(int)

    df['batter_id'] = df.get('batter', df.get('batter_id', np.nan))
    df['pitcher_id'] = df.get('pitcher', df.get('pitcher_id', np.nan))
    df['exit_velocity'] = df.get('launch_speed', df.get('exit_velocity', np.nan))
    df['launch_angle'] = df.get('launch_angle', np.nan)
    df['xwoba'] = (
        df['estimated_woba_using_speedangle']
        if 'estimated_woba_using_speedangle' in df.columns else
        (df['woba_value'] if 'woba_value' in df.columns else np.nan)
    )
    df['xba'] = df['estimated_ba_using_speedangle'] if 'estimated_ba_using_speedangle' in df.columns else np.nan

    windows = [3, 5, 7, 14]

    # ========== Rolling Feature Engineering ==========
    st.markdown("### Engineering Rolling Features (with progress bar)")
    feature_progress = st.progress(0)
    batter_ids = df['batter_id'].unique()
    pitcher_ids = df['pitcher_id'].unique()
    total_ids = len(batter_ids) + len(pitcher_ids)

    def compute_rolling(group, prefix):
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
            ])
            ab = sum((lastN['events'] == x).sum() for x in ['single', 'double', 'triple', 'home_run', 'field_out'])
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

    # Progress bar for batters
    batter_results = []
    for idx, b_id in enumerate(batter_ids):
        group = df[df['batter_id'] == b_id]
        feats = compute_rolling(group, 'B')
        row = {'batter_id': b_id}
        row.update(feats)
        batter_results.append(row)
        feature_progress.progress((idx + 1) / total_ids, text=f"Batter {idx + 1}/{len(batter_ids)}")
    batter_feats = pd.DataFrame(batter_results)
    df = df.merge(batter_feats, on='batter_id', how='left')

    # Progress bar for pitchers
    for idx, p_id in enumerate(pitcher_ids):
        group = df[df['pitcher_id'] == p_id]
        feats = compute_rolling(group, 'P')
        row = {'pitcher_id': p_id}
        row.update(feats)
        batter_results.append(row)  # just to update the bar
        feature_progress.progress((len(batter_ids) + idx + 1) / total_ids, text=f"Pitcher {idx + 1}/{len(pitcher_ids)}")
    pitcher_feats = pd.DataFrame([r for r in batter_results if 'pitcher_id' in r])
    df = df.merge(pitcher_feats, on='pitcher_id', how='left')

    feature_progress.progress(1.0, text="Feature engineering complete.")
    st.success("Rolling features done.")

    # ========== EVENT-LEVEL FEATURE EXPORT ==========
    st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
    event_cols = [c for c in df.columns if c not in ['game_date', 'batter', 'pitcher', 'events', 'result']]
    event_df = dedup_columns(df[event_cols + ['hr_outcome']])
    st.dataframe(event_df.head(20))
    st.download_button(
        "⬇️ Download Event-Level Feature CSV",
        data=event_df.to_csv(index=False).encode(),
        file_name="event_level_features.csv"
    )

    # ========== PLAYER-LEVEL (LATEST-ROLLING) FEATURE EXPORT ==========
    st.markdown("#### Download Player-Level CSV (1 row per batter):")
    # Use last game_date for each batter_id
    player_level = (
        df.sort_values('game_date')
        .groupby('batter_id')
        .last()
        .reset_index()
    )
    player_cols = [c for c in player_level.columns if c not in ['game_date', 'batter', 'pitcher', 'events', 'result']]
    player_df = dedup_columns(player_level[player_cols + ['hr_outcome']])
    st.dataframe(player_df.head(20))
    st.download_button(
        "⬇️ Download Player-Level Feature CSV",
        data=player_df.to_csv(index=False).encode(),
        file_name="player_level_features.csv"
    )

    # ========== LOGISTIC REGRESSION (WITH SCALING) ==========
    st.markdown("#### Logistic Regression Weights (Standardized Features)")
    logit_features = [c for c in event_df.columns if c not in ['batter_id', 'pitcher_id', 'game_date', 'hr_outcome']]
    X = event_df[logit_features].fillna(0)
    y = event_df['hr_outcome']

    if X.shape[0] > 100 and y.nunique() == 2:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=500)
        model.fit(X_scaled, y)
        weights = pd.Series(model.coef_[0], index=logit_features).sort_values(ascending=False)
        weights_df = pd.DataFrame({'feature': weights.index, 'weight': weights.values})
        st.dataframe(weights_df.head(30))
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
