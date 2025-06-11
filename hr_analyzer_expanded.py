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

# Enable pybaseball caching
pybaseball.cache.enable()

# ----------------- PARK/WEATHER CONTEXT ----------------- #
park_hr_rate_map = {
    'angels_stadium': 1.05, 'angel_stadium': 1.05, 'minute_maid_park': 1.06, 'coors_field': 1.30,
    'yankee_stadium': 1.19, 'fenway_park': 0.97, 'rogers_centre': 1.10, 'tropicana_field': 0.85,
    'camden_yards': 1.13, 'guaranteed_rate_field': 1.18, 'progressive_field': 1.01,
    'comerica_park': 0.96, 'kauffman_stadium': 0.98, 'globe_life_field': 1.00, 'dodger_stadium': 1.10,
    'oakland_coliseum': 0.82, 't-mobile_park': 0.86, 'tmobile_park': 0.86, 'oracle_park': 0.82, 'wrigley_field': 1.12,
    'great_american_ball_park': 1.26, 'american_family_field': 1.17, 'pnc_park': 0.87, 'busch_stadium': 0.87,
    'truist_park': 1.06, 'loan_depot_park': 0.86, 'citi_field': 1.05, 'nationals_park': 1.05,
    'petco_park': 0.85, 'chase_field': 1.06, 'citizens_bank_park': 1.19, 'sutter_health_park': 1.12
}

park_altitude_map = {
    'coors_field': 5280, 'chase_field': 1100, 'dodger_stadium': 338, 'minute_maid_park': 50,
    'fenway_park': 19, 'wrigley_field': 594, 'great_american_ball_park': 489, 'oracle_park': 10,
    'petco_park': 62, 'yankee_stadium': 55, 'citizens_bank_park': 30, 'kauffman_stadium': 750,
    'guaranteed_rate_field': 600, 'progressive_field': 650, 'busch_stadium': 466, 'camden_yards': 40,
    'rogers_centre': 250, 'angel_stadium': 160, 'tropicana_field': 3, 'citi_field': 3,
    'oakland_coliseum': 50, 'globe_life_field': 560, 'pnc_park': 725, 'loan_depot_park': 7,
    'nationals_park': 25, 'american_family_field': 633, 'sutter_health_park': 20, 't-mobile_park': 0,
    'tmobile_park': 0
}

roof_status_map = {
    'rogers_centre': 'closed', 'chase_field': 'open', 'minute_maid_park': 'open',
    'loan_depot_park': 'closed', 'globe_life_field': 'open', 'tropicana_field': 'closed',
    'american_family_field': 'open', 't-mobile_park': 'open',
    'tmobile_park': 'open'
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

# --------------------- STREAMLIT UI --------------------- #
st.title("⚾ Statcast MLB HR Analyzer — All Context Features (No CSV Upload)")
st.markdown("""
Fetches MLB Statcast batted ball events and automatically engineers **advanced rolling, park, weather, matchup, and context features for HR prediction**.
**No extra uploads needed**—everything comes from Statcast API and Weather API.
""")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=7))
with col2:
    end_date = st.date_input("End Date", value=datetime.today())

run_query = st.button("Fetch Statcast Data and Run Analyzer")

if run_query:
    st.info("Pulling Statcast data... (can take a few min for big ranges)")
    df = statcast(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    st.success(f"Loaded {len(df)} events from {start_date} to {end_date}")

    # Target batted ball events only
    target_events = ['single', 'double', 'triple', 'home_run', 'field_out']
    df = df[df['events'].isin(target_events)].reset_index(drop=True)

    df['game_date'] = pd.to_datetime(df['game_date'])
    team_code_to_park = {
        'PHI': 'citizens_bank_park',
        'ATL': 'truist_park',
        'NYM': 'citi_field',
        'BOS': 'fenway_park',
        'NYY': 'yankee_stadium',
        'CHC': 'wrigley_field',
        'LAD': 'dodger_stadium',
        'OAK': 'sutter_health_park',
        'CIN': 'great_american_ball_park',
        'DET': 'comerica_park',
        'HOU': 'minute_maid_park',
        'MIA': 'loan_depot_park',  # Fixed spelling
        'TB': 'tropicana_field',
        'MIL': 'american_family_field',
        'SD': 'petco_park',
        'SF': 'oracle_park',
        'TOR': 'rogers_centre',
        'CLE': 'progressive_field',
        'MIN': 'target_field',
        'KC': 'kauffman_stadium',
        'CWS': 'guaranteed_rate_field',
        'LAA': 'angel_stadium',
        'SEA': 't-mobile_park',  # Only use once; Statcast uses 'SEA'
        'TEX': 'globe_life_field',
        'ARI': 'chase_field',
        'COL': 'coors_field',
        'PIT': 'pnc_park',
        'STL': 'busch_stadium',
        'BAL': 'camden_yards',
        'WSH': 'nationals_park',
        'ATH': 'sutter_health_park' # For Las Vegas/Oakland alternate
    }
    df['park'] = df['home_team'].map(team_code_to_park)
    df['park'] = df['park'].replace({'tmobile_park': 't-mobile_park', 't mobile park': 't-mobile_park'})
    df['batter_id'] = df['batter']
    df['pitcher_id'] = df['pitcher']
    df['home_team_code'] = df['home_team']

    # Context: park, altitude, roof
    df['park_hr_rate'] = df['park'].map(park_hr_rate_map).fillna(1.00)
    df['park_altitude'] = df['park'].map(park_altitude_map).fillna(0)
    df['roof_status'] = df['park'].map(roof_status_map).fillna("open")

    # --- WEATHER (city+date caching) ---
    st.write("Merging weather data (can be slow for big ranges)...")
    weather_features = ['temp', 'wind_mph', 'wind_dir', 'humidity', 'condition']
    df['weather_key'] = df['home_team_code'].astype(str) + "_" + df['game_date'].dt.strftime("%Y%m%d")

    @st.cache_data(show_spinner=False)
    def get_weather(city, date):
        api_key = st.secrets["weather"]["api_key"]
        url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={date}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and resp.text.strip():
                data = resp.json()
                best_hr = 19
                hours = data['forecast']['forecastday'][0]['hour']
                hour_data = min(hours, key=lambda h: abs(int(h['time'].split(' ')[1].split(':')[0]) - best_hr))
                return {
                    'temp': hour_data['temp_f'],
                    'wind_mph': hour_data['wind_mph'],
                    'wind_dir': hour_data['wind_dir'],
                    'humidity': hour_data['humidity'],
                    'condition': hour_data['condition']['text']
                }
        except Exception as e:
            return {k: None for k in weather_features}
        return {k: None for k in weather_features}

    progress = st.progress(0, text="Fetching weather...")
    unique_keys = df['weather_key'].unique()
    for i, key in enumerate(unique_keys):
        team = key.split('_')[0]
        city = mlb_team_city_map.get(team, "New York")
        date = df[df['weather_key'] == key].iloc[0]['game_date'].strftime("%Y-%m-%d")
        weather = get_weather(city, date)
        for feat in weather_features:
            df.loc[df['weather_key'] == key, feat] = weather[feat]
        progress.progress((i + 1) / len(unique_keys), text=f"Weather {i+1}/{len(unique_keys)}")
    progress.empty()

    # --------------- ADVANCED ROLLING STATS ---------------
    st.write("Engineering rolling stat features (batter/pitcher, 3/5/7/14)...")
    ROLL_WINDOWS = [3, 5, 7, 14]
    batter_stats = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value', 'iso_value']
    pitcher_stats = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value', 'iso_value']

    def add_rolling_features(df, id_col, stat_cols, windows, prefix):
        for stat in stat_cols:
            for w in windows:
                feat = f"{prefix}_{stat}_{w}"
                df[feat] = (
                    df.groupby(id_col)[stat]
                    .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                    .fillna(0)
                )
        return df

    df = add_rolling_features(df, 'batter_id', batter_stats, ROLL_WINDOWS, 'B')
    df = add_rolling_features(df, 'pitcher_id', pitcher_stats, ROLL_WINDOWS, 'P')

    # -------------- Contextual/stat features --------------
    df['handed_matchup'] = df['stand'].astype(str) + df['p_throws'].astype(str)
    df['primary_pitch'] = df['pitch_type'].astype(str).fillna('')
    df['platoon'] = (df['stand'] != df['p_throws']).astype(int)
    df['game_hour'] = df['game_date'].dt.hour
    df['is_day'] = (df['game_hour'] < 18).astype(int)
    df['pull_air'] = ((df['bb_type'] == 'fly_ball') & (df['hc_x'].fillna(128) < 125)).astype(int)
    df['flyball'] = (df['bb_type'] == 'fly_ball').astype(int)
    df['line_drive'] = (df['bb_type'] == 'line_drive').astype(int)
    df['groundball'] = (df['bb_type'] == 'ground_ball').astype(int)
    df['pull_side'] = (df['hc_x'].fillna(128) < 125).astype(int)
    df['hr_outcome'] = (df['events'] == 'home_run').astype(int)

    st.success("Feature engineering complete.")

    # ========== EVENT-LEVEL EXPORT ========== #
    export_cols = [
        'game_date', 'batter', 'batter_id', 'pitcher', 'pitcher_id', 'events', 'description',
        'stand', 'p_throws', 'park', 'park_hr_rate', 'park_altitude', 'roof_status',
        'temp', 'wind_mph', 'wind_dir', 'humidity', 'condition',
        'handed_matchup', 'primary_pitch', 'platoon', 'is_day',
        *[f"B_{stat}_{w}" for stat in batter_stats for w in ROLL_WINDOWS],
        *[f"P_{stat}_{w}" for stat in pitcher_stats for w in ROLL_WINDOWS],
        'pull_air', 'flyball', 'line_drive', 'groundball', 'pull_side',
        'hr_outcome'
    ]
    # ========== EVENT-LEVEL EXPORT ========== #
    event_df = df[export_cols].copy()
    st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
    st.dataframe(event_df.head(20))

    # --- PLAYER-LEVEL DATAFRAME ---
    player_cols = ['batter_id', 'batter'] + [f"B_{stat}_{w}" for stat in batter_stats for w in ROLL_WINDOWS]
    player_df = (
        event_df.groupby(['batter_id', 'batter'])
        .tail(1)[player_cols]
        .reset_index(drop=True)
    )
    st.dataframe(player_df.head(20))

    # --- SAVE CSVs IN SESSION STATE --- #
    if 'event_csv' not in st.session_state:
        st.session_state['event_csv'] = event_df.to_csv(index=False).encode()
    if 'player_csv' not in st.session_state:
        st.session_state['player_csv'] = player_df.to_csv(index=False).encode()

    st.download_button(
        "⬇️ Download Event-Level CSV",
        data=st.session_state['event_csv'],
        file_name="event_level_hr_features.csv"
    )
    st.download_button(
        "⬇️ Download Player-Level CSV",
        data=st.session_state['player_csv'],
        file_name="player_level_hr_features.csv"
    )

    # ========== LOGISTIC REGRESSION (with progress) ========== #
    st.markdown("#### Logistic Regression Weights (Standardized Features)")
    # Use all rolling and context features for logit
    logit_features = [
        c for c in event_df.columns if (
            any(stat in c for stat in ['launch_speed', 'launch_angle', 'hit_distance', 'woba_value', 'iso_value']) or
            c in ['park_hr_rate', 'platoon', 'temp', 'wind_mph', 'humidity', 'pull_air', 'flyball', 'line_drive', 'groundball', 'pull_side']
        )
    ]

    model_df = event_df.dropna(subset=logit_features + ['hr_outcome'])
    X = model_df[logit_features].astype(float)
    y = model_df['hr_outcome'].astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=500)
    with st.spinner("Training logistic regression..."):
        model.fit(X_scaled, y)
    weights = pd.Series(model.coef_[0], index=logit_features).sort_values(ascending=False)
    weights_df = pd.DataFrame({'feature': weights.index, 'weight': weights.values})
    st.dataframe(weights_df.head(40))
    auc = roc_auc_score(y, model.predict_proba(X_scaled)[:, 1])
    st.write(f"Model ROC-AUC: **{auc:.3f}**")

    st.success("Full analysis done! All context features and rolling stats included.")

else:
    st.info("Set your date range and click 'Fetch Statcast Data and Run Analyzer' to begin.")
