import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybaseball import statcast
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import pybaseball
pybaseball.cache.enable()

# -------- Static Mapping Dictionaries -------- #
park_hr_rate_map = {
    'angels_stadium': 1.05, 'angel_stadium': 1.05, 'minute_maid_park': 1.06, 'coors_field': 1.30,
    'yankee_stadium': 1.19, 'fenway_park': 0.97, 'rogers_centre': 1.10, 'tropicana_field': 0.85,
    'camden_yards': 1.13, 'guaranteed_rate_field': 1.18, 'progressive_field': 1.01,
    'comerica_park': 0.96, 'kauffman_stadium': 0.98, 'globe_life_field': 1.00, 'dodger_stadium': 1.10,
    'oakland_coliseum': 0.82, 't-mobile_park': 0.86, 'tmobile_park': 0.86, 'oracle_park': 0.82,
    'wrigley_field': 1.12, 'great_american_ball_park': 1.26, 'american_family_field': 1.17,
    'pnc_park': 0.87, 'busch_stadium': 0.87, 'truist_park': 1.06, 'loan_depot_park': 0.86,
    'loandepot_park': 0.86, 'citi_field': 1.05, 'nationals_park': 1.05, 'petco_park': 0.85,
    'chase_field': 1.06, 'citizens_bank_park': 1.19, 'sutter_health_park': 1.12, 'target_field': 1.05
}
park_altitude_map = {
    'coors_field': 5280, 'chase_field': 1100, 'dodger_stadium': 338, 'minute_maid_park': 50,
    'fenway_park': 19, 'wrigley_field': 594, 'great_american_ball_park': 489, 'oracle_park': 10,
    'petco_park': 62, 'yankee_stadium': 55, 'citizens_bank_park': 30, 'kauffman_stadium': 750,
    'guaranteed_rate_field': 600, 'progressive_field': 650, 'busch_stadium': 466, 'camden_yards': 40,
    'rogers_centre': 250, 'angel_stadium': 160, 'tropicana_field': 3, 'citi_field': 3,
    'oakland_coliseum': 50, 'globe_life_field': 560, 'pnc_park': 725, 'loan_depot_park': 7,
    'loandepot_park': 7, 'nationals_park': 25, 'american_family_field': 633, 'sutter_health_park': 20,
    'target_field': 830
}
roof_status_map = {
    'rogers_centre': 'closed', 'chase_field': 'open', 'minute_maid_park': 'open',
    'loan_depot_park': 'closed', 'loandepot_park': 'closed', 'globe_life_field': 'open',
    'tropicana_field': 'closed', 'american_family_field': 'open'
}
team_code_to_park = {
    'PHI': 'citizens_bank_park', 'ATL': 'truist_park', 'NYM': 'citi_field',
    'BOS': 'fenway_park', 'NYY': 'yankee_stadium', 'CHC': 'wrigley_field',
    'LAD': 'dodger_stadium', 'OAK': 'sutter_health_park', 'CIN': 'great_american_ball_park',
    'DET': 'comerica_park', 'HOU': 'minute_maid_park', 'MIA': 'loandepot_park',
    'TB': 'tropicana_field', 'MIL': 'american_family_field', 'SD': 'petco_park',
    'SF': 'oracle_park', 'TOR': 'rogers_centre', 'CLE': 'progressive_field',
    'MIN': 'target_field', 'KC': 'kauffman_stadium', 'CWS': 'guaranteed_rate_field',
    'LAA': 'angel_stadium', 'SEA': 't-mobile_park', 'TEX': 'globe_life_field',
    'ARI': 'chase_field', 'COL': 'coors_field', 'PIT': 'pnc_park', 'STL': 'busch_stadium',
    'BAL': 'camden_yards', 'WSH': 'nationals_park', 'ATH': 'sutter_health_park'
}
mlb_team_city_map = {
    'ANA': 'Anaheim', 'ARI': 'Phoenix', 'ATL': 'Atlanta', 'BAL': 'Baltimore', 'BOS': 'Boston',
    'CHC': 'Chicago', 'CIN': 'Cincinnati', 'CLE': 'Cleveland', 'COL': 'Denver', 'CWS': 'Chicago',
    'DET': 'Detroit', 'HOU': 'Houston', 'KC': 'Kansas City', 'LAA': 'Anaheim', 'LAD': 'Los Angeles',
    'MIA': 'Miami', 'MIL': 'Milwaukee', 'MIN': 'Minneapolis', 'NYM': 'New York', 'NYY': 'New York',
    'OAK': 'Oakland', 'PHI': 'Philadelphia', 'PIT': 'Pittsburgh', 'SD': 'San Diego', 'SEA': 'Seattle',
    'SF': 'San Francisco', 'STL': 'St. Louis', 'TB': 'St. Petersburg', 'TEX': 'Arlington', 'TOR': 'Toronto',
    'WSH': 'Washington'
}

# ----------- Streamlit UI ----------- #
st.title("⚾ Statcast MLB HR Analyzer — Full Context, Interactions, Robust")
st.markdown("""
Fetches MLB Statcast events and engineers **advanced rolling, pitch-mix, park, weather, context, and interaction features** for HR prediction.
No CSV upload needed. Everything calculated for you.
""")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=7))
with col2:
    end_date = st.date_input("End Date", value=datetime.today())

run_query = st.button("Fetch Statcast Data and Run Analyzer")

if run_query:
    progress = st.progress(0, text="Initializing...")  # Single progress bar
    steps = 7

    # 1. Download Statcast Data
    progress.progress(1/steps, text="Pulling Statcast data...")
    df = statcast(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    st.success(f"Loaded {len(df)} events from {start_date} to {end_date}")

    # 2. Map and Prep Columns
    progress.progress(2/steps, text="Mapping parks and prepping features...")
    df = df[df['events'].isin(['single', 'double', 'triple', 'home_run', 'field_out'])].reset_index(drop=True)
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['home_team_code'] = df['home_team']
    df['park'] = df['home_team_code'].map(team_code_to_park).fillna(df['home_team'].str.lower().str.replace(' ', '_'))
    df['batter_id'] = df['batter']
    df['pitcher_id'] = df['pitcher']
    df['park_hr_rate'] = df['park'].map(park_hr_rate_map).fillna(1.00)
    df['park_altitude'] = df['park'].map(park_altitude_map).fillna(0)
    df['roof_status'] = df['park'].map(roof_status_map).fillna("open")
    df['handed_matchup'] = df['stand'].astype(str) + df['p_throws'].astype(str)
    df['hr_outcome'] = (df['events'] == 'home_run').astype(int)

    # 3. Compute Park-Handed HR Rate
    progress.progress(3/steps, text="Computing park-handed HR rates...")
    def compute_park_handed_hr_rate(df):
        grp = df.groupby(['park', 'handed_matchup'])
        rates = grp['hr_outcome'].agg(['sum', 'count']).reset_index()
        rates['park_handed_hr_rate'] = rates['sum'] / rates['count']
        df = df.merge(rates[['park', 'handed_matchup', 'park_handed_hr_rate']],
                      on=['park', 'handed_matchup'], how='left')
        return df
    df = compute_park_handed_hr_rate(df)

    # 4. Weather API
    progress.progress(4/steps, text="Merging weather data...")
    weather_features = ['temp', 'wind_mph', 'wind_dir', 'humidity', 'condition']
    df['weather_key'] = df['home_team_code'] + "_" + df['game_date'].dt.strftime("%Y%m%d")
    @st.cache_data(show_spinner=False)
    def get_weather(city, date):
        api_key = st.secrets["weather"]["api_key"]
        url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={date}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and resp.text.strip():
                data = resp.json()
                hours = data['forecast']['forecastday'][0]['hour']
                hour_data = min(hours, key=lambda h: abs(int(h['time'].split(' ')[1].split(':')[0]) - 19))
                return {
                    'temp': hour_data['temp_f'],
                    'wind_mph': hour_data['wind_mph'],
                    'wind_dir': hour_data['wind_dir'],
                    'humidity': hour_data['humidity'],
                    'condition': hour_data['condition']['text']
                }
        except Exception as e:
            st.warning(f"Weather API error for {city} {date}: {e}")
        return {k: None for k in weather_features}

    unique_keys = df['weather_key'].unique()
    for i, key in enumerate(unique_keys):
        team = key.split('_')[0]
        city = mlb_team_city_map.get(team, "New York")
        date = df[df['weather_key'] == key].iloc[0]['game_date'].strftime("%Y-%m-%d")
        weather = get_weather(city, date)
        for feat in weather_features:
            df.loc[df['weather_key'] == key, feat] = weather[feat]
    # (We batch the progress bar update here for clarity)

    # 5. Add Robust Statcast Columns
    progress.progress(5/steps, text="Filling missing advanced stat columns...")
    for col in ['xwoba', 'xslg', 'xba']:
        if col not in df.columns:
            st.warning(f"Missing advanced stat: {col}")
            df[col] = np.nan

    # 6. Feature Engineering (rolling, pitch mix, batted ball flags, interactions)
    progress.progress(6/steps, text="Engineering rolling, pitch mix, batted ball, and interaction features...")
    ROLL = [3, 5, 7, 14]
    # Batter/pitcher advanced stats
    batter_stats = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value', 'iso_value', 'xwoba', 'xslg', 'xba']
    pitcher_stats = batter_stats
    batter_stats = [s for s in batter_stats if s in df.columns]
    pitcher_stats = [s for s in pitcher_stats if s in df.columns]
    # Rolling means
    def add_rolling(df, group, stats, windows, prefix):
        for stat in stats:
            for w in windows:
                col = f"{prefix}_{stat}_{w}"
                df[col] = (
                    df.groupby(group)[stat]
                      .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                )
        return df
    df = add_rolling(df, 'batter_id', batter_stats, ROLL, 'B')
    df = add_rolling(df, 'pitcher_id', pitcher_stats, ROLL, 'P')
    # Rolling max/median EV
    for w in ROLL:
        df[f'B_max_ev_{w}'] = df.groupby('batter_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
        df[f'B_median_ev_{w}'] = df.groupby('batter_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).median())
        df[f'P_max_ev_{w}'] = df.groupby('pitcher_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
        df[f'P_median_ev_{w}'] = df.groupby('pitcher_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).median())
    # Batted ball flags (robust)
    df['is_hard_hit'] = (df['launch_speed'].fillna(-999) >= 95).astype(int)
    df['is_barrel'] = ((df['launch_speed'].fillna(-999) >= 98) & (df['launch_angle'].between(26, 30))).astype(int)
    df['is_sweet_spot'] = (df['launch_angle'].between(8, 32)).astype(int)
    df['flyball'] = (df['bb_type'] == 'fly_ball').astype(int)
    df['line_drive'] = (df['bb_type'] == 'line_drive').astype(int)
    df['groundball'] = (df['bb_type'] == 'ground_ball').astype(int)
    df['pull_air'] = ((df['bb_type'] == 'fly_ball') & (df['hc_x'].fillna(-1) < 125)).astype(int)
    df['pull_side'] = (df['hc_x'].fillna(-1) < 125).astype(int)
    # Pitch mix %
    if 'pitch_type' in df.columns:
        for pt in df['pitch_type'].dropna().unique():
            pt_col = f'pitch_pct_{pt}'
            df[pt_col] = (df['pitch_type'] == pt).astype(float)
            for w in ROLL:
                df[f'B_pitch_pct_{pt}_{w}'] = df.groupby('batter_id')[pt_col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                df[f'P_pitch_pct_{pt}_{w}'] = df.groupby('pitcher_id')[pt_col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
    # Interaction features (weather x batted ball/context)
    for f in ['flyball', 'is_hard_hit', 'is_sweet_spot', 'pull_air']:
        df[f'{f}_temp'] = df[f] * df['temp']
        df[f'{f}_wind'] = df[f] * df['wind_mph']
    df['park_hr_rate_temp'] = df['park_hr_rate'] * df['temp']
    df['park_handed_hr_rate_temp'] = df['park_handed_hr_rate'] * df['temp']

    # 7. Export and Model
    progress.progress(7/steps, text="Preparing outputs and fitting logistic model...")

    # Event-level export
    export_cols = [
        'game_date', 'batter', 'batter_id', 'pitcher', 'pitcher_id', 'events', 'description',
        'stand', 'p_throws', 'park', 'park_hr_rate', 'park_altitude', 'roof_status', 'park_handed_hr_rate',
        'temp', 'wind_mph', 'wind_dir', 'humidity', 'condition', 'handed_matchup'
    ]
    # Gather all rolling, pitch mix, context, batted ball, and interaction columns
    rolling_cols = [c for c in df.columns if any(x in c for x in ['B_', 'P_', 'max_ev', 'median_ev', 'pitch_pct_', '_temp', '_wind'])]
    flag_cols = ['is_hard_hit', 'is_barrel', 'is_sweet_spot', 'flyball', 'line_drive', 'groundball', 'pull_air', 'pull_side', 'hr_outcome']
    event_cols = [c for c in export_cols + rolling_cols + flag_cols if c in df.columns]
    event_df = df[event_cols].copy()
    st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
    st.dataframe(event_df.head(20))
    st.download_button("⬇️ Download Event-Level CSV", data=event_df.to_csv(index=False), file_name="event_level_hr_features.csv")

    # Player-level export
    st.markdown("#### Download Player-Level Rolling Feature CSV (1 row per batter):")
    player_cols = ['batter_id', 'batter'] + [c for c in event_df.columns if c.startswith('B_')]
    player_df = (
        event_df.groupby(['batter_id', 'batter'])
        .tail(1)[player_cols]
        .reset_index(drop=True)
    )
    st.dataframe(player_df.head(20))
    st.download_button(
        "⬇️ Download Player-Level CSV",
        data=player_df.to_csv(index=False),
        file_name="player_level_hr_features.csv"
    )

    # ====== LOGISTIC REGRESSION (Robust) ======
    st.markdown("#### Logistic Regression Weights (Standardized Features, Downloadable)")
    # Use all rolling, context, pitch mix, batted ball, park, weather, interaction columns
    # Add only features that are numeric and not identifiers/descriptions
    exclude = ['game_date', 'batter', 'batter_id', 'pitcher', 'pitcher_id', 'events', 'description', 'stand', 'p_throws', 'home_team_code', 'park', 'roof_status', 'wind_dir', 'condition']
    logit_features = [c for c in event_df.columns if c not in exclude and event_df[c].dtype in [np.float64, np.int64]]

    # Only use features with enough non-missing data
    nonnull_thresh = 0.9
    coverage = event_df[logit_features].notnull().mean()
    use_feats = coverage[coverage > nonnull_thresh].index.tolist()

    model_df = event_df.dropna(subset=use_feats + ['hr_outcome'], how='any')
    if len(model_df) > 20 and len(use_feats) > 2:
        X = model_df[use_feats].astype(float)
        y = model_df['hr_outcome'].astype(int)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=500)
        with st.spinner("Training logistic regression..."):
            model.fit(X_scaled, y)
        weights = pd.Series(model.coef_[0], index=use_feats).sort_values(ascending=False)
        weights_df = pd.DataFrame({'feature': weights.index, 'weight': weights.values})
        st.dataframe(weights_df.head(50))
        auc = roc_auc_score(y, model.predict_proba(X_scaled)[:, 1])
        st.write(f"Model ROC-AUC: **{auc:.3f}**")

        # Download button for weights
        st.download_button(
            "⬇️ Download Logistic Weights CSV",
            data=weights_df.to_csv(index=False),
            file_name="logistic_feature_weights.csv"
        )
    else:
        st.warning("Not enough data with complete features to fit logistic regression for HR prediction. (Try a larger date range or check for missing data.)")

    progress.progress(1.0, text="✅ All done! All features, interactions, and outputs are ready.")

else:
    st.info("Set your date range and click 'Fetch Statcast Data and Run Analyzer' to begin.")
