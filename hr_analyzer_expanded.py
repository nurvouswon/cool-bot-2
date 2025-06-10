import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybaseball import statcast
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")

st.markdown("""
# ⚾ Statcast MLB HR Analyzer — Full Feature/Context Event + Player Level
**1. Choose data source:**
- Upload a Statcast batted ball events CSV  
- OR fetch new data from MLB Statcast for your date range

**2. This app will:**
- Compute all advanced rolling features (3/5/7/14)
- Output both event-level and player-level CSVs
- Fit logistic regression (with scaling) and output feature weights
""")

# === Data source selection ===
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
        start_date = st.date_input("Start date", value=datetime.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End date", value=datetime.today())
    if st.button("Fetch Statcast Data"):
        with st.spinner("Fetching Statcast data from MLB..."):
            df = statcast(start_dt=start_date.strftime("%Y-%m-%d"), end_dt=end_date.strftime("%Y-%m-%d"))
            st.success(f"Loaded {len(df)} events from {start_date} to {end_date}")

if df is not None and not df.empty:
    # --- Filter to only batted ball events you want ---
    if 'type' in df.columns:
        df = df[df['type'] == 'X']
    if 'events' in df.columns:
        keep_events = ['single', 'double', 'triple', 'home_run', 'field_out']
        df = df[df['events'].str.lower().isin(keep_events)]
    st.success(f"Filtered: {len(df)} matching events")

    # --- Data cleaning/standardization ---
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])
    else:
        st.error("Missing 'game_date' column in dataset.")
        st.stop()

    # --- HR tagging ---
    hr_events = ['home_run', 'home run', 'hr']
    if 'events' in df.columns:
        df['hr_outcome'] = df['events'].str.lower().isin(hr_events).astype(int)
    elif 'result' in df.columns:
        df['hr_outcome'] = df['result'].str.lower().isin(hr_events).astype(int)
    else:
        df['hr_outcome'] = 0

    # --- Minimal column harmonization for features ---
    df['batter_id'] = df.get('batter', df.get('batter_id', np.nan))
    df['pitcher_id'] = df.get('pitcher', df.get('pitcher_id', np.nan))
    df['exit_velocity'] = df.get('launch_speed', df.get('exit_velocity', np.nan))
    df['launch_angle'] = df.get('launch_angle', np.nan)
    df['xwoba'] = df['estimated_woba_using_speedangle'] if 'estimated_woba_using_speedangle' in df.columns else (
        df['woba_value'] if 'woba_value' in df.columns else np.nan)
    df['xba'] = df['estimated_ba_using_speedangle'] if 'estimated_ba_using_speedangle' in df.columns else np.nan

    #### ========== FEATURE ENGINEERING ========== ####

    @st.cache_data(show_spinner=False)
    def engineer_features(df):
        windows = [3, 5, 7, 14]
        progress = st.progress(0, text="Engineering batter rolling features...")
        batter_feats = []
        batters = df['batter_id'].dropna().unique()
        for i, b in enumerate(batters):
            group = df[df['batter_id'] == b].sort_values('game_date')
            row = {'batter_id': b}
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
                xba = lastN['xba'].mean() if 'xba' in lastN.columns and not lastN['xba'].isnull().all() else np.nan
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
                xiso = (xslg - xba) if (xslg is not np.nan and xba is not np.nan) else np.nan

                row[f'B_BarrelRate_{w}'] = barrels / pa if pa else np.nan
                row[f'B_EV_{w}'] = lastN['exit_velocity'].mean() if pa else np.nan
                row[f'B_SLG_{w}'] = slg / ab if ab else np.nan
                row[f'B_xSLG_{w}'] = xslg
                row[f'B_xISO_{w}'] = xiso
                row[f'B_xwoba_{w}'] = lastN['xwoba'].mean() if pa else np.nan
                row[f'B_sweet_spot_pct_{w}'] = sweet / pa if pa else np.nan
                row[f'B_hardhit_pct_{w}'] = hard / pa if pa else np.nan
            batter_feats.append(row)
            progress.progress(i / len(batters), text=f"Batter {i+1}/{len(batters)}")
        progress.progress(1.0, text="Batter features complete")

        progress = st.progress(0, text="Engineering pitcher rolling features...")
        pitcher_feats = []
        pitchers = df['pitcher_id'].dropna().unique()
        for i, p in enumerate(pitchers):
            group = df[df['pitcher_id'] == p].sort_values('game_date')
            row = {'pitcher_id': p}
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
                xba = lastN['xba'].mean() if 'xba' in lastN.columns and not lastN['xba'].isnull().all() else np.nan
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
                xiso = (xslg - xba) if (xslg is not np.nan and xba is not np.nan) else np.nan

                row[f'P_BarrelRate_{w}'] = barrels / pa if pa else np.nan
                row[f'P_EV_{w}'] = lastN['exit_velocity'].mean() if pa else np.nan
                row[f'P_SLG_{w}'] = slg / ab if ab else np.nan
                row[f'P_xSLG_{w}'] = xslg
                row[f'P_xISO_{w}'] = xiso
                row[f'P_xwoba_{w}'] = lastN['xwoba'].mean() if pa else np.nan
                row[f'P_sweet_spot_pct_{w}'] = sweet / pa if pa else np.nan
                row[f'P_hardhit_pct_{w}'] = hard / pa if pa else np.nan
            pitcher_feats.append(row)
            progress.progress(i / len(pitchers), text=f"Pitcher {i+1}/{len(pitchers)}")
        progress.progress(1.0, text="Pitcher features complete")

        batter_df = pd.DataFrame(batter_feats)
        pitcher_df = pd.DataFrame(pitcher_feats)

        # Merge batter features to event
        df = df.merge(batter_df, on='batter_id', how='left')
        df = df.merge(pitcher_df, on='pitcher_id', how='left')
        return df, batter_df, pitcher_df

    st.markdown("### Engineering Rolling Features (with progress bar)")
    df, batter_feats, pitcher_feats = engineer_features(df)
    st.success("Feature engineering complete.")

    #### ========== EVENT-LEVEL FEATURE EXPORT ========== ####
    st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
    event_cols = [c for c in df.columns if c not in ['game_date', 'batter_id', 'pitcher_id', 'hr_outcome']]

    # Deduplicate columns if necessary
    def dedup_columns(df):
        df = df.copy()
        new_cols = []
        seen = {}
        for col in df.columns:
            base = col
            i = seen.get(base, 0)
            while col in new_cols:
                i += 1
                col = f"{base}.{i}"
            seen[base] = i
            new_cols.append(col)
        df.columns = new_cols
        return df

    event_df = dedup_columns(df[event_cols + ['hr_outcome']])
    st.dataframe(event_df.head(20))
    st.download_button(
        "⬇️ Download Event-Level Feature CSV",
        data=event_df.to_csv(index=False).encode(),
        file_name="event_level_features.csv"
    )

    #### ========== PLAYER-LEVEL FEATURE EXPORT ========== ####
    st.markdown("#### Download Player-Level CSV (1 row per batter):")
    player_df = batter_feats.copy()
    st.dataframe(player_df.head(20))
    st.download_button(
        "⬇️ Download Player-Level Feature CSV",
        data=player_df.to_csv(index=False).encode(),
        file_name="player_level_features.csv"
    )

    #### ========== LOGISTIC REGRESSION (SCALED FEATURES) ========== ####
    st.markdown("#### Logistic Regression Weights (Standardized Features)")
    logit_features = [c for c in event_df.columns if c not in ['hr_outcome']]
    # Only keep numeric columns
    numeric_features = event_df[logit_features].select_dtypes(include=[np.number]).columns.tolist()
    X = event_df[numeric_features].fillna(0)
    y = event_df['hr_outcome']

    if X.shape[0] > 100 and y.nunique() == 2:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=500)
        model.fit(X_scaled, y)
        weights = pd.Series(model.coef_[0], index=numeric_features).sort_values(ascending=False)
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
