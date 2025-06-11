import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybaseball import statcast
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import pybaseball
import requests

pybaseball.cache.enable()  # Enable disk caching for Statcast!

# ------------ MLB Park/Team/City Mapping (robust) -------------------- #
team_code_to_park = {
    'PHI': 'citizens_bank_park', 'ATL': 'truist_park', 'NYM': 'citi_field', 'BOS': 'fenway_park',
    'NYY': 'yankee_stadium', 'CHC': 'wrigley_field', 'LAD': 'dodger_stadium', 'OAK': 'sutter_health_park',
    'CIN': 'great_american_ball_park', 'DET': 'comerica_park', 'HOU': 'minute_maid_park', 'MIA': 'loan_depot_park',
    'TB': 'tropicana_field', 'MIL': 'american_family_field', 'SD': 'petco_park', 'SF': 'oracle_park',
    'TOR': 'rogers_centre', 'CLE': 'progressive_field', 'MIN': 'target_field', 'KC': 'kauffman_stadium',
    'CWS': 'guaranteed_rate_field', 'LAA': 'angel_stadium', 'SEA': 't-mobile_park', 'TEX': 'globe_life_field',
    'ARI': 'chase_field', 'COL': 'coors_field', 'PIT': 'pnc_park', 'STL': 'busch_stadium',
    'BAL': 'camden_yards', 'WSH': 'nationals_park', 'ATH': 'sutter_health_park'
}
# Handle alternate spellings
park_hr_rate_map = {
    'angels_stadium': 1.05, 'angel_stadium': 1.05, 'minute_maid_park': 1.06, 'coors_field': 1.30,
    'yankee_stadium': 1.19, 'fenway_park': 0.97, 'rogers_centre': 1.10, 'tropicana_field': 0.85,
    'camden_yards': 1.13, 'guaranteed_rate_field': 1.18, 'progressive_field': 1.01,
    'comerica_park': 0.96, 'kauffman_stadium': 0.98, 'globe_life_field': 1.00, 'dodger_stadium': 1.10,
    'oakland_coliseum': 0.82, 't-mobile_park': 0.86, 'tmobile_park': 0.86, 'oracle_park': 0.82,
    'wrigley_field': 1.12, 'great_american_ball_park': 1.26, 'american_family_field': 1.17, 'pnc_park': 0.87, 'busch_stadium': 0.87,
    'truist_park': 1.06, 'loan_depot_park': 0.86, 'loandepot_park': 0.86, 'citi_field': 1.05, 'nationals_park': 1.05,
    'petco_park': 0.85, 'chase_field': 1.06, 'citizens_bank_park': 1.19, 'sutter_health_park': 1.12,
    'target_field': 1.02
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
st.title("⚾ Statcast MLB HR Analyzer — Full Context, Robust")
st.markdown("""
Fetches MLB Statcast batted ball events and auto-engineers advanced rolling, park, weather, pitch-mix, and matchup features for HR prediction. **No extra uploads needed. Handles missing advanced Statcast columns.**
""")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=7))
with col2:
    end_date = st.date_input("End Date", value=datetime.today())

if st.button("Fetch Statcast Data and Run Analyzer"):
    st.info("Pulling Statcast data... (can take a few min for big ranges)")
    df = statcast(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    st.success(f"Loaded {len(df)} events from {start_date} to {end_date}")

    # Map parks using home team (robust for spelling)
    df['park'] = df['home_team'].map(team_code_to_park).str.lower().replace({'tmobile_park': 't-mobile_park'})
    df['park_hr_rate'] = df['park'].map(park_hr_rate_map).fillna(1.00)
    df['batter_id'] = df['batter']
    df['pitcher_id'] = df['pitcher']
    df['game_date'] = pd.to_datetime(df['game_date'])

    # Only keep batted ball events of interest
    target_events = ['single', 'double', 'triple', 'home_run', 'field_out']
    df = df[df['events'].isin(target_events)].reset_index(drop=True)

    # --- Weather features (cache by city/date)
    st.write("Merging weather data (may take a while for large date ranges)...")
    weather_features = ['temp', 'wind_mph', 'wind_dir', 'humidity', 'condition']
    df['weather_key'] = df['home_team'] + "_" + df['game_date'].dt.strftime("%Y%m%d")
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
            st.warning(f"Weather API error for {city} {date}: {e}")
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

    # --- Advanced rolling stats: support missing columns robustly
    ROLL = [3, 5, 7, 14]
    # Define ALL batter & pitcher rolling stats (robust to missing advanced cols)
    ALL_BATTER_STATS = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value', 'iso_value', 'xba', 'xwoba', 'xslg']
    ALL_PITCHER_STATS = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value', 'iso_value', 'xba', 'xwoba', 'xslg']
    present_batter_stats = [s for s in ALL_BATTER_STATS if s in df.columns]
    present_pitcher_stats = [s for s in ALL_PITCHER_STATS if s in df.columns]

    # Rolling helper
    def add_rolling(df, group, stats, windows, prefix):
        for stat in stats:
            for w in windows:
                col = f"{prefix}_{stat}_{w}"
                df[col] = df.groupby(group)[stat].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean()) if stat in df.columns else np.nan
        return df
    df = add_rolling(df, 'batter_id', present_batter_stats, ROLL, 'B')
    df = add_rolling(df, 'pitcher_id', present_pitcher_stats, ROLL, 'P')

    # Rolling pitch mix % (by batter/pitcher, by pitch_type, all windows)
    if 'pitch_type' in df.columns:
        pitch_types = df['pitch_type'].dropna().unique()
        for pt in pitch_types:
            for w in ROLL:
                col_b = f"B_pitch_{pt}_{w}"
                col_p = f"P_pitch_{pt}_{w}"
                df[col_b] = (
                    df.groupby('batter_id')['pitch_type']
                      .transform(lambda x: x.shift(1).rolling(w, min_periods=1).apply(lambda y: (y == pt).mean() if len(y) else np.nan))
                )
                df[col_p] = (
                    df.groupby('pitcher_id')['pitch_type']
                      .transform(lambda x: x.shift(1).rolling(w, min_periods=1).apply(lambda y: (y == pt).mean() if len(y) else np.nan))
                )

    # --- Context features (robust: fillna(False).astype(int)) ---
    df['handed_matchup'] = df['stand'].astype(str) + df['p_throws'].astype(str)
    df['primary_pitch'] = df['pitch_type'] if 'pitch_type' in df.columns else np.nan
    df['platoon'] = (df['stand'] != df['p_throws']).fillna(False).astype(int)
    df['game_hour'] = df['game_date'].dt.hour
    df['is_day'] = (df['game_hour'] < 18).fillna(False).astype(int)
    df['pull_air'] = ((df['bb_type'] == 'fly_ball') & (df['hc_x'] < 125)).fillna(False).astype(int)
    df['flyball'] = (df['bb_type'] == 'fly_ball').fillna(False).astype(int)
    df['line_drive'] = (df['bb_type'] == 'line_drive').fillna(False).astype(int)
    df['groundball'] = (df['bb_type'] == 'ground_ball').fillna(False).astype(int)
    df['pull_side'] = (df['hc_x'] < 125).fillna(False).astype(int)
    df['hr_outcome'] = (df['events'] == 'home_run').fillna(False).astype(int)

    st.success("Feature engineering complete.")

    # ==== EXPORT EVENT-LEVEL DATA ====
    # All batter and pitcher rollings present
    event_export_cols = [
        'game_date', 'batter', 'batter_id', 'pitcher', 'pitcher_id', 'events', 'description',
        'stand', 'p_throws', 'park', 'park_hr_rate', 'temp', 'wind_mph', 'humidity', 'condition',
        'handed_matchup', 'primary_pitch', 'platoon', 'is_day', 'pull_air', 'flyball', 'line_drive', 'groundball', 'pull_side',
        *[f"B_{stat}_{w}" for stat in present_batter_stats for w in ROLL],
        *[f"P_{stat}_{w}" for stat in present_pitcher_stats for w in ROLL],
        'hr_outcome'
    ]
    # Add pitch mix cols if they exist
    if 'pitch_type' in df.columns:
        pitch_types = df['pitch_type'].dropna().unique()
        for pt in pitch_types:
            for w in ROLL:
                event_export_cols += [f"B_pitch_{pt}_{w}", f"P_pitch_{pt}_{w}"]

    event_df = df[event_export_cols].copy()
    st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
    st.dataframe(event_df.head(20))
    st.download_button("⬇️ Download Event-Level CSV", data=event_df.to_csv(index=False), file_name="event_level_hr_features.csv")

    # ==== PLAYER-LEVEL EXPORT ====
    st.markdown("#### Download Player-Level Rolling Feature CSV (1 row per batter):")
    player_cols = ['batter_id', 'batter'] + [f"B_{stat}_{w}" for stat in present_batter_stats for w in ROLL]
    player_df = (
        event_df.groupby(['batter_id', 'batter'])
        .tail(1)[player_cols]
        .reset_index(drop=True)
    )
    st.dataframe(player_df.head(20))
    st.download_button("⬇️ Download Player-Level CSV", data=player_df.to_csv(index=False), file_name="player_level_hr_features.csv")

    # ==== LOGISTIC REGRESSION WEIGHTS ====
    st.markdown("#### Logistic Regression Weights (Standardized Features)")
    # All rolling/context features + pitch mix cols
    logit_features = []
    logit_features += [c for c in event_df.columns if any(stat in c for stat in ['launch_speed', 'launch_angle', 'hit_distance', 'woba_value', 'iso_value', 'xba', 'xwoba', 'xslg'])]
    logit_features += ['park_hr_rate', 'platoon', 'temp', 'wind_mph', 'humidity', 'pull_air', 'flyball', 'line_drive', 'groundball', 'pull_side', 'is_day']
    if 'pitch_type' in df.columns:
        for pt in pitch_types:
            for w in ROLL:
                logit_features += [f"B_pitch_{pt}_{w}", f"P_pitch_{pt}_{w}"]
    # Remove dupes, keep only those in event_df
    logit_features = [f for f in pd.unique(logit_features) if f in event_df.columns]
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

    st.success("Full analysis done! All context, advanced stats, pitch mix, weather, and robust handling of missing data included.")

else:
    st.info("Set your date range and click 'Fetch Statcast Data and Run Analyzer' to begin.")
