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

# ---- Data source selection ----
data_source = st.radio("Select data source:",
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
if df is not None:
    # --- Filter to only relevant batted ball events ---
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    allowed_events = ['single', 'double', 'triple', 'home_run', 'field_out']
    if 'events' in df.columns:
        df = df[df['events'].str.lower().isin(allowed_events)]
    elif 'type' in df.columns:
        df = df[df['type'] == 'X']

    st.success(f"Loaded {len(df)} batted ball events.")

    # --- Data cleaning/standardization ---
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

    # --- Batter/Pitcher/Stat columns (with fallback handling) ---
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

    def rolling_features(group, prefix):
        group = group.sort_values('game_date')
        feats = {}
        for w in ROLLING_WINDOWS:
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
            ab = sum((lastN['events'] == x).sum() for x in allowed_events)
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

    # ---- Feature engineering progress bar ----
    st.markdown("### Engineering Rolling Features (with progress bar)")
    progress = st.progress(0, text="Rolling features for batters...")
    batter_feats = []
    batter_ids = df['batter_id'].unique()
    for idx, b_id in enumerate(batter_ids):
        group = df[df['batter_id'] == b_id]
        feats = rolling_features(group, 'B')
        feats['batter_id'] = b_id
        batter_feats.append(feats)
        progress.progress(idx / (len(batter_ids) + 1), text=f"Batter: {idx + 1}/{len(batter_ids)}")

    batter_feats = pd.DataFrame(batter_feats)
    df = df.merge(batter_feats, on='batter_id', how='left')

    progress.progress(0.6, text="Rolling features for pitchers...")
    pitcher_feats = []
    pitcher_ids = df['pitcher_id'].unique()
    for idx, p_id in enumerate(pitcher_ids):
        group = df[df['pitcher_id'] == p_id]
        feats = rolling_features(group, 'P')
        feats['pitcher_id'] = p_id
        pitcher_feats.append(feats)
        progress.progress(0.6 + 0.4 * idx / (len(pitcher_ids) + 1), text=f"Pitcher: {idx + 1}/{len(pitcher_ids)}")
    pitcher_feats = pd.DataFrame(pitcher_feats)
    df = df.merge(pitcher_feats, on='pitcher_id', how='left')

    progress.progress(1.0, text="Rolling features done.")

    # ========== Merge Context Data: ParkHR, HandedHR, PitchTypeHR ==========
    # For demo, create some dummy context rates. Replace with real CSV merges as needed!
    park_list = df['home_team'].unique() if 'home_team' in df.columns else ['demo_park']
    park_hr = pd.DataFrame({'park': park_list, 'ParkHRRate': np.random.uniform(0.9, 1.2, size=len(park_list))})
    handed_hr = pd.DataFrame({'BatterHandedness': df['stand'].unique(), 'PitcherHandedness': df['p_throws'].unique()})
    handed_hr['HandedHRRate'] = np.random.uniform(0.02, 0.06, size=handed_hr.shape[0])
    pitchtype_hr = pd.DataFrame({'pitch_type': df['pitch_type'].unique() if 'pitch_type' in df.columns else ['FF'],
                                 'PitchTypeHRRate': np.random.uniform(0.01, 0.04, size=df['pitch_type'].nunique() if 'pitch_type' in df.columns else 1)})

    # Normalize key columns before merging
    if 'home_team' in df.columns:
        df['park'] = df['home_team']
    else:
        df['park'] = 'demo_park'

    df = df.merge(park_hr, on='park', how='left')
    df = df.merge(handed_hr, left_on=['stand','p_throws'], right_on=['BatterHandedness','PitcherHandedness'], how='left')
    if 'pitch_type' in df.columns:
        df = df.merge(pitchtype_hr, on='pitch_type', how='left')
    else:
        df['PitchTypeHRRate'] = 0.02

    # --------- Deduplicate columns utility ---------
    def dedup_columns(df):
        df = df.copy()
        df.columns = pd.io.parsers.ParserBase({'names':df.columns})._maybe_dedup_names(df.columns)
        return df

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

    # ========== PLAYER-LEVEL FEATURE EXPORT ==========
    st.markdown("#### Download Player-Level Rolling Feature CSV (1 row per batter, as of last date):")
    player_level = dedup_columns(batter_feats)
    st.dataframe(player_level.head(20))
    st.download_button(
        "⬇️ Download Player-Level Rolling Feature CSV",
        data=player_level.to_csv(index=False).encode(),
        file_name="player_level_features.csv"
    )

    # ========== LOGISTIC REGRESSION ANALYSIS ==========
    st.markdown("#### Logistic Regression: All Features for HR Outcome")
    # --- Select features for regression ---
    regress_features = [c for c in event_df.columns if event_df[c].dtype in [np.float64, np.float32, np.int64, np.int32] and c not in ['hr_outcome']]
    st.write(f"**Features used ({len(regress_features)}):**", regress_features[:20], "...")
    scaler = StandardScaler()
    X = scaler.fit_transform(event_df[regress_features].fillna(0))
    y = event_df['hr_outcome']
    with st.spinner("Fitting logistic regression..."):
        progress_bar = st.progress(0.0, text="Fitting logistic regression...")
        model = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)
        model.fit(X, y)
        progress_bar.progress(1.0, text="Regression complete.")
        weights = pd.Series(model.coef_[0], index=regress_features).sort_values(ascending=False)
        weights_df = pd.DataFrame({'feature': weights.index, 'weight': weights.values})
        auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    st.write(weights_df)
    st.markdown(f"**In-sample AUC:** `{auc:.3f}`")
    st.download_button(
        "⬇️ Download Logistic Regression Feature Weights CSV",
        data=weights_df.to_csv(index=False).encode(),
        file_name="logit_feature_weights.csv"
    )

else:
    st.info("Upload or fetch data to start analysis.")
