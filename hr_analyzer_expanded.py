import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybaseball import statcast
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

st.set_page_config(layout="wide")
st.title("⚾ Statcast MLB HR Analyzer — Event & Player Level + Contextual Logistic Weights")

# --- Settings ---
batted_ball_events = ['single', 'double', 'triple', 'home_run', 'field_out']
rolling_windows = [3, 5, 7, 14]

# --- Data upload/fetch UI ---
st.markdown("""
**1. Choose data source:**
- Upload a Statcast batted ball events CSV  
- **OR** fetch new data from MLB Statcast for your date range

**2. This app will:**
- Compute all advanced rolling features (3/5/7/14 games)
- Merge context features (Park HR, Handed HR, Pitch Type HR)
- Output both event-level and player-level CSVs
- Fit logistic regression (with scaling) and output feature weights
""")

data_source = st.radio("Select data source:", ["Upload CSV", "Fetch new data from MLB Statcast (pybaseball)"])
df = None

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
        with st.spinner("Fetching Statcast data from MLB..."):
            df = statcast(start_dt=start_date.strftime("%Y-%m-%d"), end_dt=end_date.strftime("%Y-%m-%d"))
            st.session_state['df_raw'] = df
if 'df_raw' in st.session_state and (df is None or df.empty):
    df = st.session_state['df_raw']

if df is not None and not df.empty:
    # --- Clean/standardize, only allowed events ---
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])
    else:
        st.error("Missing 'game_date' column in dataset."); st.stop()
    if 'events' in df.columns:
        df = df[df['events'].str.lower().isin(batted_ball_events)]
    elif 'description' in df.columns:
        df = df[df['description'].str.lower().isin(batted_ball_events)]
    else:
        st.error("Could not detect event column for filtering."); st.stop()
    st.success(f"Loaded {len(df)} batted ball events.")

    # --- Tag HR outcome ---
    hr_events = ['home_run', 'home run', 'hr']
    df['hr_outcome'] = df['events'].str.lower().isin(hr_events).astype(int)
    df['batter_id'] = df.get('batter', df.get('batter_id', np.nan))
    df['pitcher_id'] = df.get('pitcher', df.get('pitcher_id', np.nan))
    df['exit_velocity'] = df.get('launch_speed', df.get('exit_velocity', np.nan))
    df['launch_angle'] = df.get('launch_angle', np.nan)
    df['xwoba'] = df['estimated_woba_using_speedangle'] if 'estimated_woba_using_speedangle' in df.columns else np.nan
    df['xba'] = df['estimated_ba_using_speedangle'] if 'estimated_ba_using_speedangle' in df.columns else np.nan

    ###### --- CONTEXT CSV UPLOADS (required for context merge) --- ######
    st.subheader("Upload contextual rate CSVs:")
    park_hr_file = st.file_uploader("Ballpark HR Rates (CSV)", type=['csv'])
    handed_hr_file = st.file_uploader("Handedness HR Rates (CSV)", type=['csv'])
    pitchtype_hr_file = st.file_uploader("Pitch Type HR Rates (CSV)", type=['csv'])

    if not all([park_hr_file, handed_hr_file, pitchtype_hr_file]):
        st.info("⬆️ Upload all 3 contextual HR rate CSVs to enable full feature set/logit weights!")
        st.stop()

    # --- Merge context: ParkHRRate ---
    park_hr = pd.read_csv(park_hr_file)
    park_col = next((c for c in park_hr.columns if 'park' in c.lower()), park_hr.columns[0])
    rate_col = next((c for c in park_hr.columns if 'hr' in c.lower()), park_hr.columns[1])
    park_hr = park_hr.rename(columns={park_col: 'park', rate_col: 'ParkHRRate'})
    # normalize park name
    df['park'] = df['home_team'] if 'home_team' in df.columns else 'unknown'
    park_hr['park'] = park_hr['park'].astype(str).str.lower().str.replace(" ", "_")
    df['park'] = df['park'].astype(str).str.lower().str.replace(" ", "_")
    df = df.merge(park_hr[['park','ParkHRRate']], on='park', how='left')

    # --- Merge context: HandedHRRate ---
    handed_hr = pd.read_csv(handed_hr_file)
    bhand = next((c for c in handed_hr.columns if 'batter' in c.lower()), handed_hr.columns[0])
    phand = next((c for c in handed_hr.columns if 'pitcher' in c.lower()), handed_hr.columns[1])
    hrrate = next((c for c in handed_hr.columns if 'hr' in c.lower()), handed_hr.columns[2])
    handed_hr = handed_hr.rename(columns={bhand:'BatterHandedness', phand:'PitcherHandedness', hrrate:'HandedHRRate'})
    # fallback: assign handedness from stand/p_throws if possible
    if 'stand' in df.columns and 'p_throws' in df.columns:
        df['BatterHandedness'] = df['stand']
        df['PitcherHandedness'] = df['p_throws']
    else:
        df['BatterHandedness'] = 'NA'
        df['PitcherHandedness'] = 'NA'
    df = df.merge(handed_hr, on=['BatterHandedness','PitcherHandedness'], how='left')

    # --- Merge context: PitchTypeHRRate ---
    pitchtype_hr = pd.read_csv(pitchtype_hr_file)
    ptcol = next((c for c in pitchtype_hr.columns if 'type' in c.lower()), pitchtype_hr.columns[0])
    ptrate = next((c for c in pitchtype_hr.columns if 'hr' in c.lower()), pitchtype_hr.columns[1])
    pitchtype_hr = pitchtype_hr.rename(columns={ptcol: 'pitch_type', ptrate: 'PitchTypeHRRate'})
    if 'pitch_type' in df.columns:
        df = df.merge(pitchtype_hr, on='pitch_type', how='left')
    else:
        df['PitchTypeHRRate'] = np.nan

    ### --- Rolling features, with progress bar ---
    def get_rolling_feats(g, prefix):
        out = {}
        g = g.sort_values('game_date')
        for w in rolling_windows:
            lastN = g.tail(w)
            pa = len(lastN)
            barrels = lastN[(lastN['exit_velocity'] > 95) & (lastN['launch_angle'].between(20,35))].shape[0]
            sweet = lastN[lastN['launch_angle'].between(8, 32)].shape[0]
            hard = lastN[lastN['exit_velocity'] >= 95].shape[0]
            slg = (
                (lastN['events'] == 'single').sum() +
                2 * (lastN['events'] == 'double').sum() +
                3 * (lastN['events'] == 'triple').sum() +
                4 * (lastN['events'] == 'home_run').sum()
            )
            ab = sum((lastN['events'] == x).sum() for x in batted_ball_events)
            xsingle = (lastN['xba'] >= 0.5) & (lastN['launch_angle'] < 15)
            double = (lastN['xba'] >= 0.5) & (lastN['launch_angle'].between(15, 30))
            triple = (lastN['xba'] >= 0.5) & (lastN['launch_angle'].between(30, 40))
            hr = (lastN['launch_angle'] > 35) & (lastN['exit_velocity'] > 100)
            xsingle = xsingle.sum()
            xdouble = double.sum()
            xtriple = triple.sum()
            xhr = hr.sum()
            xab = xsingle + xdouble + xtriple + xhr
            xslg = (1 * xsingle + 2 * xdouble + 3 * xtriple + 4 * xhr) / xab if xab else np.nan
            xba = lastN['xba'].mean() if 'xba' in lastN.columns and not lastN['xba'].isnull().all() else np.nan
            xiso = (xslg - xba) if (xslg is not np.nan and xba is not np.nan) else np.nan
            out[f'{prefix}_BarrelRate_{w}'] = barrels / pa if pa else np.nan
            out[f'{prefix}_EV_{w}'] = lastN['exit_velocity'].mean() if pa else np.nan
            out[f'{prefix}_SLG_{w}'] = slg / ab if ab else np.nan
            out[f'{prefix}_xSLG_{w}'] = xslg
            out[f'{prefix}_xISO_{w}'] = xiso
            out[f'{prefix}_xwoba_{w}'] = lastN['xwoba'].mean() if pa else np.nan
            out[f'{prefix}_sweet_spot_pct_{w}'] = sweet / pa if pa else np.nan
            out[f'{prefix}_hardhit_pct_{w}'] = hard / pa if pa else np.nan
        return pd.Series(out)

    st.markdown("### Engineering Rolling Features (with progress bar)")
    # --- Batters
    batter_feats = df.groupby('batter_id').apply(lambda x: get_rolling_feats(x, 'B')).reset_index()
    df = df.merge(batter_feats, on='batter_id', how='left')
    # --- Pitchers
    pitcher_feats = df.groupby('pitcher_id').apply(lambda x: get_rolling_feats(x, 'P')).reset_index()
    df = df.merge(pitcher_feats, on='pitcher_id', how='left')

    # --- Player-level snapshot (latest game for each player)
    player_level = df.sort_values('game_date').groupby('batter_id').tail(1).copy()
    player_level.reset_index(drop=True, inplace=True)

    # --- Select all numeric/engineered features for modeling ---
    ignore_cols = ['game_date','events','hr_outcome','batter','pitcher','home_team','park','BatterHandedness','PitcherHandedness']
    model_cols = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c]) and df[c].notnull().sum() > 10]
    model_cols += ['ParkHRRate','HandedHRRate','PitchTypeHRRate']
    model_cols = list(sorted(set(model_cols)))
    X = df[model_cols].fillna(0)
    y = df['hr_outcome']

    # --- Fit logistic regression with scaling ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logit = LogisticRegression(max_iter=600, solver='lbfgs')
    logit.fit(X_scaled, y)
    weights = pd.Series(logit.coef_[0], index=model_cols)
    auc = roc_auc_score(y, logit.predict_proba(X_scaled)[:,1])

    st.markdown("#### Logistic Regression Feature Weights")
    weights_df = pd.DataFrame({'feature':model_cols, 'weight':weights}).sort_values('weight', ascending=False)
    st.dataframe(weights_df)
    st.markdown(f"**In-sample AUC:** `{auc:.3f}`")
    st.download_button(
        "⬇️ Download Logit Weights CSV",
        data=weights_df.to_csv(index=False).encode(),
        file_name="logit_feature_weights.csv"
    )

    st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
    event_cols = [c for c in df.columns if c not in ['game_date','batter','pitcher','home_team','park']]
    st.dataframe(df[event_cols + ['hr_outcome']].head(20))
    st.download_button(
        "⬇️ Download Event-Level Feature CSV",
        data=df[event_cols + ['hr_outcome']].to_csv(index=False).encode(),
        file_name="event_level_features.csv"
    )
    st.markdown("#### Download Player-Level CSV (1 row per batter, as of last event):")
    player_cols = [c for c in player_level.columns if c not in ['game_date','batter','pitcher','home_team','park']]
    st.dataframe(player_level[player_cols].head(20))
    st.download_button(
        "⬇️ Download Player-Level Feature CSV",
        data=player_level[player_cols].to_csv(index=False).encode(),
        file_name="player_level_features.csv"
    )
else:
    st.info("Upload or fetch data to start. Then upload all 3 context CSVs (park, handedness, pitch type) for full feature/weight analysis.")
