import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pybaseball import statcast

st.set_page_config(layout="wide")
st.title("⚾ MLB Statcast HR Analyzer – Event-Level & Player-Level Feature Generator")

st.markdown("""
**1. Choose data source:**
- Upload your own Statcast CSV  
- **OR** fetch new data from MLB Statcast for your date range

**2. App will:**
- Engineer all advanced rolling features (batter & pitcher, for all 3/5/7/14-game windows)
- Fit logistic regression to *all* features for HR prediction (event-level)
- Export:
  - Event-level feature CSV (for model training/discovery)
  - Player-level feature CSV (rolling stats as of target date, for leaderboard)
""")

# ==== Data source selection ====
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

# ========== Main Data Pipeline ==========
if df is not None and not df.empty:
    # --- Data cleaning/standardization ---
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])
    else:
        st.error("Missing 'game_date' column in dataset.")
        st.stop()

    # --- Only batted ball events in play ---
    if 'type' in df.columns:
        df = df[df['type'] == 'X']
    elif 'events' in df.columns:
        batted_ball_events = [
            'single', 'double', 'triple', 'home_run',
            'field_out', 'force_out', 'grounded_into_double_play', 'fielders_choice_out',
            'other_out', 'fielders_choice', 'double_play', 'triple_play'
        ]
        df = df[df['events'].str.lower().isin(batted_ball_events)]
    if df.empty:
        st.warning("No batted ball events found for selected data/range.")
        st.stop()

    # --- HR tagging ---
    hr_events = ['home_run', 'home run', 'hr']
    if 'events' in df.columns:
        df['hr_outcome'] = df['events'].str.lower().isin(hr_events).astype(int)
    elif 'result' in df.columns:
        df['hr_outcome'] = df['result'].str.lower().isin(hr_events).astype(int)
    else:
        df['hr_outcome'] = 0

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

    # --- Rolling feature functions ---
    ROLLING_WINDOWS = [3, 5, 7, 14]

    def rolling_features(group, prefix, id_col, windows=ROLLING_WINDOWS):
        group = group.sort_values('game_date')
        feats = {}
        for w in windows:
            lastN = group.tail(w)
            pa = len(lastN)
            barrels = lastN[(lastN['exit_velocity'] > 95) & (lastN['launch_angle'].between(20, 35))].shape[0]
            sweet = lastN[lastN['launch_angle'].between(8, 32)].shape[0]
            hard = lastN[lastN['exit_velocity'] >= 95].shape[0]
            slg = np.nansum([
                (lastN['events'] == 'single').sum() if 'events' in lastN else 0,
                2 * (lastN['events'] == 'double').sum() if 'events' in lastN else 0,
                3 * (lastN['events'] == 'triple').sum() if 'events' in lastN else 0,
                4 * (lastN['events'] == 'home_run').sum() if 'events' in lastN else 0
            ]) if 'events' in lastN else np.nan
            ab = sum((lastN['events'] == x).sum() for x in ['single', 'double', 'triple', 'home_run', 'field_out', 'force_out', 'other_out']) if 'events' in lastN else np.nan
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

    # --- Progress bar for rolling features ---
    progress = st.progress(0, text="Processing batter rolling features...")
    batter_ids = df['batter_id'].unique()
    batter_feats = []
    for idx, bid in enumerate(batter_ids):
        batter_feats.append(rolling_features(df[df['batter_id'] == bid], 'B', 'batter_id'))
        progress.progress((idx + 1) / len(batter_ids), text=f"Batter {idx+1}/{len(batter_ids)}")
    batter_feats_df = pd.DataFrame(batter_feats, index=batter_ids).reset_index().rename(columns={'index':'batter_id'})

    progress.progress(0, text="Processing pitcher rolling features...")
    pitcher_ids = df['pitcher_id'].unique()
    pitcher_feats = []
    for idx, pid in enumerate(pitcher_ids):
        pitcher_feats.append(rolling_features(df[df['pitcher_id'] == pid], 'P', 'pitcher_id'))
        progress.progress((idx + 1) / len(pitcher_ids), text=f"Pitcher {idx+1}/{len(pitcher_ids)}")
    pitcher_feats_df = pd.DataFrame(pitcher_feats, index=pitcher_ids).reset_index().rename(columns={'index':'pitcher_id'})

    # --- Merge on only unique batter/pitcher IDs and ONLY feature columns, no overlap
    batter_feats_df = batter_feats_df.loc[:,~batter_feats_df.columns.duplicated()]
    pitcher_feats_df = pitcher_feats_df.loc[:,~pitcher_feats_df.columns.duplicated()]
    if 'batter_id' in df.columns:
        df = df.merge(batter_feats_df, on='batter_id', how='left', suffixes=('', '_dupbat'))
    if 'pitcher_id' in df.columns:
        df = df.merge(pitcher_feats_df, on='pitcher_id', how='left', suffixes=('', '_duppit'))
    # Drop any duplicate columns created in merge (e.g., from bad merges)
    df = df.loc[:, ~df.columns.duplicated()]

    progress.progress(1.0, text="Rolling features done, preparing event-level CSV...")

    # --- Final event-level columns for logistic regression ---
    # You may adjust this to your required feature list
    event_cols = [c for c in df.columns if (
        c.startswith('B_') or c.startswith('P_')
        or c in ['batter_id','pitcher_id','exit_velocity','launch_angle','xwoba','xba']
    )]
    # Remove again if any dupes
    event_cols = pd.unique(event_cols).tolist()

    st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
    st.dataframe(df[event_cols + ['hr_outcome']].head(20), use_container_width=True)
    st.download_button(
        "⬇️ Download Event-Level CSV",
        data=df[event_cols + ['hr_outcome']].to_csv(index=False).encode(),
        file_name="analyzer_event_level_features.csv"
    )

    # --- Player-level (as-of-date) rolling stats
    st.markdown("#### Download Player-Level CSV (latest rolling features per player):")
    player_level = batter_feats_df.copy()
    st.dataframe(player_level.head(20), use_container_width=True)
    st.download_button(
        "⬇️ Download Player-Level CSV",
        data=player_level.to_csv(index=False).encode(),
        file_name="analyzer_player_level_features.csv"
    )

    # --- Logistic Regression (fit on event-level, all features)
    st.markdown("#### Logistic Regression Feature Analysis (event-level)")
    feature_cols = [col for col in event_cols if df[col].notnull().sum() > 0]
    X = df[feature_cols].fillna(0)
    y = df['hr_outcome']
    if X.shape[0] > 100 and y.nunique() == 2:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=500, solver='lbfgs')
        model.fit(X_scaled, y)
        weights = pd.Series(model.coef_[0], index=feature_cols).sort_values(ascending=False)
        weights_df = pd.DataFrame({'feature': weights.index, 'weight': weights.values})
        st.write(weights_df)
        auc = roc_auc_score(y, model.predict_proba(X_scaled)[:, 1])
        st.markdown(f"**In-sample AUC:** `{auc:.3f}`")
        st.download_button(
            "⬇️ Download All Feature Logit Weights CSV",
            data=weights_df.to_csv(index=False).encode(),
            file_name="logit_feature_weights.csv"
        )
    else:
        st.warning("Not enough data for regression.")

else:
    st.info("Upload or generate data to start analysis.")
