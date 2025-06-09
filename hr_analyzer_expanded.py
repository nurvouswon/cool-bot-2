import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybaseball import statcast

st.set_page_config(layout="wide")

st.markdown("""
**1. Choose data source:**
- Upload a Statcast batted ball events CSV  
- **OR** fetch new data from MLB Statcast for your date range

**2. Choose rolling window (games):**  
- All advanced & context features will be engineered  
- Logistic regression will be fit to *all* features for HR prediction  
- Download all outputs for your model building
""")

# === Rolling window selection (at the top, only ONCE) ===
min_win, max_win = 1, 60
st.subheader("Choose Rolling Window(s) for Feature Engineering")
window_range = st.slider(
    "Rolling window size(s) (games)",
    min_value=min_win,
    max_value=max_win,
    value=(3, 14),
    step=1
)
windows = list(range(window_range[0], window_range[1] + 1))

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
            # --- Filter to only batted ball events in play ---
if df is not None:
    if 'type' in df.columns:
        # Statcast convention: type=='X' means ball in play
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
        st.success(f"Loaded {len(df)} events from {start_date} to {end_date}")

if df is not None and not df.empty:
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

    # --- User selects rolling window(s) ---
    min_win, max_win = 1, 60
    st.subheader("Choose Rolling Window(s) for Feature Engineering")
    window_range = st.slider("Rolling window size(s) (games)", min_value=min_win, max_value=max_win, value=(3, 14), step=1)
    windows = list(range(window_range[0], window_range[1] + 1))

    # --- Rolling feature functions
    def rolling_features(group, prefix, id_col):
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

            feats[f'{prefix}_BarrelRate_{w}'] = barrels / pa if pa else np.nan
            feats[f'{prefix}_EV_{w}'] = lastN['exit_velocity'].mean() if pa else np.nan
            feats[f'{prefix}_SLG_{w}'] = slg / ab if ab else np.nan
            feats[f'{prefix}_xSLG_{w}'] = xslg
            feats[f'{prefix}_xISO_{w}'] = xiso
            feats[f'{prefix}_xwoba_{w}'] = lastN['xwoba'].mean() if pa else np.nan
            feats[f'{prefix}_sweet_spot_pct_{w}'] = sweet / pa if pa else np.nan
            feats[f'{prefix}_hardhit_pct_{w}'] = hard / pa if pa else np.nan
        return pd.Series(feats)

    st.subheader("Engineering Rolling Features")
    batter_feats = df.groupby('batter_id').apply(lambda x: rolling_features(x, 'B', 'batter_id')).reset_index()
    df = df.merge(batter_feats, on='batter_id', how='left')
    pitcher_feats = df.groupby('pitcher_id').apply(lambda x: rolling_features(x, 'P', 'pitcher_id')).reset_index()
    df = df.merge(pitcher_feats, on='pitcher_id', how='left')

    # --- Contextual HR rates
    st.subheader("Context HR Rates")
    context_dfs = {}

    if 'stand' in df.columns and 'p_throws' in df.columns:
        hand_hr_df = (
            df.groupby(['stand', 'p_throws'])['hr_outcome'].mean().reset_index()
            .rename(columns={
                'stand': 'BatterHandedness',
                'p_throws': 'PitcherHandedness',
                'hr_outcome': 'HandedHRRate'
            })
        )
        st.write("HR Rate by Batter vs Pitcher Handedness")
        st.dataframe(hand_hr_df)
        context_dfs['handed_hr'] = hand_hr_df

    if 'pitch_type' in df.columns:
        pitch_hr_df = (
            df.groupby('pitch_type')['hr_outcome'].mean()
            .reset_index()
            .rename(columns={
                'pitch_type': 'pitch_type',
                'hr_outcome': 'PitchTypeHRRate'
            })
        )
        st.write("HR Rate by Pitch Type")
        st.dataframe(pitch_hr_df)
        context_dfs['pitchtype_hr'] = pitch_hr_df

    if 'home_team' in df.columns:
        park_hr_df = (
            df.groupby('home_team')['hr_outcome'].mean()
            .reset_index()
            .rename(columns={
                'home_team': 'park',
                'hr_outcome': 'ParkHRRate'
            })
        )
        st.write("HR Rate by Ballpark")
        st.dataframe(park_hr_df)
        context_dfs['park_hr'] = park_hr_df

    # --- Feature list (insert your logit_features/context_features here) ---
    logit_features = [ ... ]  # insert your features
    context_features = [ ... ]  # insert your context features

    all_features = []
    for base in logit_features:
        all_features.append(base)
    for base in context_features:
        all_features.append(base)
    feature_cols = [f for f in all_features if f in df.columns]

    # --- Logistic Regression
    st.subheader("Logistic Regression Weights (All Features)")
    X = df[feature_cols].fillna(0)
    y = df['hr_outcome']
    if X.shape[0] > 100 and y.nunique() == 2:
        model = LogisticRegression(max_iter=200)
        model.fit(X, y)
        weights = pd.Series(model.coef_[0], index=feature_cols).sort_values(ascending=False)
        weights_df = pd.DataFrame({'feature': weights.index, 'weight': weights.values})
        st.write(weights_df)
        auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
        st.markdown(f"**In-sample AUC:** `{auc:.3f}`")
        st.download_button(
            "Download All Feature Logit Weights CSV",
            data=weights_df.to_csv(index=False).encode(),
            file_name="logit_feature_weights.csv"
        )
    else:
        st.warning("Not enough data for regression.")

    st.subheader("Sample Data with Features")
    st.dataframe(df[feature_cols + ['hr_outcome']].head(20))

    # --- Export all engineered features as CSV
    st.download_button(
        "Download Feature Data CSV (for model validation)",
        data=df[feature_cols + ['hr_outcome']].to_csv(index=False).encode(),
        file_name="analyzer_features_export.csv"
    )
else:
    st.info("Upload or generate data to start analysis.")
