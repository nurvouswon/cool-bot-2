import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from pybaseball import statcast

st.set_page_config(layout="wide")

st.title("⚾ MLB Statcast Analyzer — Event-Level & Player-Level Full Feature Engineering")

st.markdown("""
**1. Choose data source:**  
- Upload your own Statcast CSV  
- OR fetch new data from MLB Statcast API for your date range

**2. App will:**  
- Engineer all advanced rolling features (batter & pitcher, for all 3/5/7/14-game windows)
- Fit logistic regression to *all* features for HR prediction (event-level)
- Export:
    - Event-level feature CSV (for model training/discovery)
    - Player-level feature CSV (rolling stats as of target date, for leaderboard)
""")

#### --- SETTINGS ---
WINDOWS = [3, 5, 7, 14]
MIN_EVENT_ROWS = 100  # Minimum events for regression

#### --- DATA SOURCE ---
data_source = st.radio("Select data source:", ["Upload CSV", "Fetch new data from MLB Statcast (pybaseball)"])

df = None
event_download_ready = False
player_download_ready = False

if data_source == "Upload CSV":
    csv_file = st.file_uploader("Upload your Statcast Batted Ball Events CSV", type=["csv"])
    if csv_file:
        df = pd.read_csv(csv_file)
elif data_source == "Fetch new data from MLB Statcast (pybaseball)":
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=datetime.today() - timedelta(days=14))
    with col2:
        end_date = st.date_input("End date", value=datetime.today())
    if st.button("Fetch Statcast Data"):
        with st.spinner("Fetching MLB Statcast data..."):
            df = statcast(start_dt=start_date.strftime("%Y-%m-%d"), end_dt=end_date.strftime("%Y-%m-%d"))
        st.success(f"Loaded {len(df)} events from {start_date} to {end_date}")

if df is not None and not df.empty:
    progress = st.progress(0, text="Preparing data...")
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    progress.progress(0.03, text="Standardizing columns...")

    # --- Filter to only batted ball events in play ---
    if 'type' in df.columns:
        df = df[df['type'] == 'X']
    elif 'events' in df.columns:
        batted_ball_events = [
            'single', 'double', 'triple', 'home_run',
            'field_out', 'force_out', 'grounded_into_double_play', 'fielders_choice_out',
            'other_out', 'fielders_choice', 'double_play', 'triple_play'
        ]
        df = df[df['events'].str.lower().isin(batted_ball_events)]
    progress.progress(0.07, text="Filtering to batted ball events in play...")

    # --- Standardize/Fill missing critical columns ---
    df['game_date'] = pd.to_datetime(df.get('game_date', df.get('date', np.nan)))
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

    # --- HR tagging ---
    hr_events = ['home_run', 'home run', 'hr']
    if 'events' in df.columns:
        df['hr_outcome'] = df['events'].str.lower().isin(hr_events).astype(int)
    elif 'result' in df.columns:
        df['hr_outcome'] = df['result'].str.lower().isin(hr_events).astype(int)
    else:
        df['hr_outcome'] = 0

    progress.progress(0.15, text="Core stat columns and HR tags prepared...")

    # --- Rolling feature calculation ---
    def rolling_features(group, prefix, id_col):
        group = group.sort_values('game_date')
        feats = {}
        for w in WINDOWS:
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
            ab = sum((lastN['events'] == x).sum() for x in ['single', 'double', 'triple', 'home_run', 'field_out', 'force_out', 'other_out'])
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

    st.subheader("Engineering Rolling Features (with progress bar)")
    batter_ids = df['batter_id'].unique()
    pitcher_ids = df['pitcher_id'].unique()
    n_steps = len(batter_ids) + len(pitcher_ids)
    n_done = 0

    # --- Batters ---
    batter_results = []
    for idx, batter_id in enumerate(batter_ids):
        group = df[df['batter_id'] == batter_id]
        feats = rolling_features(group, 'B', 'batter_id')
        row = {'batter_id': batter_id}
        row.update(feats)
        batter_results.append(row)
        n_done += 1
        progress.progress(n_done / n_steps, text=f"Batters {n_done}/{n_steps}")

    batter_feats = pd.DataFrame(batter_results)
    df = df.merge(batter_feats, on='batter_id', how='left')

    # --- Pitchers ---
    pitcher_results = []
    for idx, pitcher_id in enumerate(pitcher_ids):
        group = df[df['pitcher_id'] == pitcher_id]
        feats = rolling_features(group, 'P', 'pitcher_id')
        row = {'pitcher_id': pitcher_id}
        row.update(feats)
        pitcher_results.append(row)
        n_done += 1
        progress.progress(n_done / n_steps, text=f"Pitchers {n_done}/{n_steps}")

    pitcher_feats = pd.DataFrame(pitcher_results)
    df = df.merge(pitcher_feats, on='pitcher_id', how='left')

    progress.progress(0.99, text="Rolling features done, preparing event-level CSV...")

    #### ==== EVENT-LEVEL CSV EXPORT ==== ####
    event_cols = [c for c in df.columns if c not in ['game_date', 'batter', 'pitcher', 'events', 'result']]
    st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
    st.dataframe(df[event_cols + ['hr_outcome']].head(20))
    st.download_button(
        "⬇️ Download Event-Level CSV",
        data=df[event_cols + ['hr_outcome']].to_csv(index=False).encode(),
        file_name="event_level_features.csv"
    )
    event_download_ready = True

    #### ==== PLAYER-LEVEL AGGREGATE CSV ==== ####
    st.markdown("#### Download Player-Level CSV (all rolling features as of last event for each player):")
    # Take the *last event per batter* as the "current" rolling window for that batter
    last_batter_rows = df.sort_values('game_date').groupby('batter_id').tail(1)
    last_batter_cols = [c for c in last_batter_rows.columns if c not in ['game_date', 'batter', 'pitcher', 'events', 'result', 'hr_outcome']]
    st.dataframe(last_batter_rows[last_batter_cols].head(20))
    st.download_button(
        "⬇️ Download Player-Level CSV",
        data=last_batter_rows[last_batter_cols].to_csv(index=False).encode(),
        file_name="player_level_features.csv"
    )
    player_download_ready = True

    progress.progress(1.0, text="Done!")

    #### ==== LOGISTIC REGRESSION FIT/EXPORT ==== ####
    st.markdown("### Logistic Regression Analysis (event-level, for HR prediction)")
    logit_features = [c for c in df.columns if (c.startswith('B_') or c.startswith('P_')) and any(str(w) in c for w in WINDOWS)]
    X = df[logit_features].fillna(0)
    y = df['hr_outcome']
    st.write(f"Training on {X.shape[0]} event rows and {X.shape[1]} features.")

    if X.shape[0] > MIN_EVENT_ROWS and y.nunique() == 2:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=800, solver='lbfgs')
        model.fit(X_scaled, y)
        weights = pd.Series(model.coef_[0], index=logit_features).sort_values(ascending=False)
        auc = roc_auc_score(y, model.predict_proba(X_scaled)[:, 1])
        st.markdown(f"**In-sample AUC:** `{auc:.3f}`")
        weights_df = pd.DataFrame({'feature': weights.index, 'weight': weights.values})
        st.write(weights_df)
        st.download_button(
            "⬇️ Download Logistic Regression Weights CSV",
            data=weights_df.to_csv(index=False).encode(),
            file_name="logit_feature_weights.csv"
        )
    else:
        st.warning("Not enough data for regression.")

else:
    st.info("Upload or fetch data to start analysis.")
