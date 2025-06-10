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
- OR fetch new data from MLB Statcast for your date range

**2. This app will:**
- Compute all advanced rolling features (3/5/7/14)
- Output both event-level and player-level CSVs
- Fit logistic regression (with scaling) and output feature weights
""")

#### ==== DATA SOURCE UI ==== ####
data_source = st.radio("Select data source:", ["Upload CSV", "Fetch new data from MLB Statcast (pybaseball)"])
df = None

if data_source == "Upload CSV":
    csv_file = st.file_uploader("Upload Statcast Batted Ball Events CSV", type=["csv"])
    if csv_file:
        df = pd.read_csv(csv_file)
elif data_source == "Fetch new data from MLB Statcast (pybaseball)":
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=datetime.today() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End date", value=datetime.today())
    if st.button("Fetch Statcast Data"):
        with st.spinner("Fetching Statcast data from MLB... (may take a minute)"):
            df = statcast(start_dt=start_date.strftime("%Y-%m-%d"), end_dt=end_date.strftime("%Y-%m-%d"))
            st.success(f"Loaded {len(df)} events from {start_date} to {end_date}")

#### ==== DATA PREP ==== ####
if df is not None and not df.empty:
    # --- Standardize columns ---
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])
    else:
        st.error("Missing 'game_date' column in dataset.")
        st.stop()
    # Filter for batted ball events
    if 'type' in df.columns:
        df = df[df['type'] == 'X']
    elif 'events' in df.columns:
        batted_ball_events = [
            'single', 'double', 'triple', 'home_run',
            'field_out'
        ]
        df = df[df['events'].str.lower().isin(batted_ball_events)]
    # Always create a 'park' column from 'home_team' if needed
    if 'park' not in df.columns and 'home_team' in df.columns:
        df['park'] = df['home_team']
    # --- Tag HR events ---
    hr_events = ['home_run', 'home run', 'hr']
    if 'events' in df.columns:
        df['hr_outcome'] = df['events'].str.lower().isin(hr_events).astype(int)
    elif 'result' in df.columns:
        df['hr_outcome'] = df['result'].str.lower().isin(hr_events).astype(int)
    else:
        df['hr_outcome'] = 0
    # --- ID columns ---
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

    # === Feature windows ===
    windows = [3, 5, 7, 14]

    # --- Rolling features (with progress bar) ---
    st.subheader("Engineering Rolling Features (with progress bar)")
    progress = st.progress(0)
    batters = df['batter_id'].dropna().unique()
    pitchers = df['pitcher_id'].dropna().unique()
    batter_results, pitcher_results = [], []

    def batter_rolling_feats(group):
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
            feats[f'B_BarrelRate_{w}'] = barrels / pa if pa else np.nan
            feats[f'B_EV_{w}'] = lastN['exit_velocity'].mean() if pa else np.nan
            feats[f'B_SLG_{w}'] = slg / ab if ab else np.nan
            feats[f'B_xSLG_{w}'] = xslg
            feats[f'B_xISO_{w}'] = xiso
            feats[f'B_xwoba_{w}'] = lastN['xwoba'].mean() if pa else np.nan
            feats[f'B_sweet_spot_pct_{w}'] = sweet / pa if pa else np.nan
            feats[f'B_hardhit_pct_{w}'] = hard / pa if pa else np.nan
        return pd.Series(feats)

    def pitcher_rolling_feats(group):
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
            feats[f'P_BarrelRate_{w}'] = barrels / pa if pa else np.nan
            feats[f'P_EV_{w}'] = lastN['exit_velocity'].mean() if pa else np.nan
            feats[f'P_SLG_{w}'] = slg / ab if ab else np.nan
            feats[f'P_xSLG_{w}'] = xslg
            feats[f'P_xISO_{w}'] = xiso
            feats[f'P_xwoba_{w}'] = lastN['xwoba'].mean() if pa else np.nan
            feats[f'P_sweet_spot_pct_{w}'] = sweet / pa if pa else np.nan
            feats[f'P_hardhit_pct_{w}'] = hard / pa if pa else np.nan
        return pd.Series(feats)

    for idx, b in enumerate(batters):
        batter_results.append({'batter_id': b, **batter_rolling_feats(df[df['batter_id'] == b])})
        progress.progress(min((idx+1)/len(batters)/2, 1.0), text=f"Batter: {idx+1}/{len(batters)}")

    for idx, p in enumerate(pitchers):
        pitcher_results.append({'pitcher_id': p, **pitcher_rolling_feats(df[df['pitcher_id'] == p])})
        progress.progress(min((0.5 + (idx+1)/len(pitchers)/2), 1.0), text=f"Pitcher: {idx+1}/{len(pitchers)}")

    batter_feats = pd.DataFrame(batter_results)
    pitcher_feats = pd.DataFrame(pitcher_results)
    df = df.merge(batter_feats, on='batter_id', how='left', suffixes=('', '_batterfeat'))
    df = df.merge(pitcher_feats, on='pitcher_id', how='left', suffixes=('', '_pitcherfeat'))
    progress.progress(1.0, text="Rolling features done.")

    #### ==== EVENT-LEVEL CSV EXPORT ==== ####
    st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
    event_cols = [c for c in df.columns if c not in ['game_date', 'batter', 'pitcher', 'events', 'result']]
    df_event = df[event_cols + ['hr_outcome']].copy()
    # Drop duplicate columns if any
    df_event = df_event.loc[:, ~df_event.columns.duplicated()]
    st.dataframe(df_event.head(20))
    st.download_button("⬇️ Download Event-Level CSV", data=df_event.to_csv(index=False).encode(), file_name="statcast_event_level_features.csv")

    #### ==== PLAYER-LEVEL CSV EXPORT ==== ####
    st.markdown("#### Download Player-Level CSV (rolling features as of end date):")
    player_level = batter_feats.set_index('batter_id').join(pitcher_feats.set_index('pitcher_id'), how='outer')
    st.dataframe(player_level.head(20))
    st.download_button("⬇️ Download Player-Level CSV", data=player_level.to_csv().encode(), file_name="statcast_player_level_features.csv")

    #### ==== LOGISTIC REGRESSION ANALYSIS ==== ####
    st.markdown("#### Logistic Regression Weights (all features):")
    # Choose only valid, non-id features
    all_features = [c for c in df_event.columns if c not in ['hr_outcome', 'batter_id', 'pitcher_id'] and pd.api.types.is_numeric_dtype(df_event[c])]
    # Remove any columns that are all-NaN
    X = df_event[all_features].fillna(0)
    y = df_event['hr_outcome']
    if X.shape[0] > 100 and y.nunique() == 2:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=1000, solver='lbfgs')
        model.fit(X_scaled, y)
        weights = pd.Series(model.coef_[0], index=all_features).sort_values(ascending=False)
        weights_df = pd.DataFrame({'feature': weights.index, 'weight': weights.values})
        auc = roc_auc_score(y, model.predict_proba(X_scaled)[:,1])
        st.write(f"**In-sample AUC:** `{auc:.3f}`")
        st.dataframe(weights_df)
        st.download_button("⬇️ Download Logistic Regression Weights CSV", data=weights_df.to_csv(index=False).encode(), file_name="logit_feature_weights.csv")
    else:
        st.warning("Not enough data for regression.")

else:
    st.info("Upload or fetch data to begin analysis.")
