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
pybaseball.cache.enable()  # Enable disk caching for all pybaseball requests!

# ------------------- CONFIGS & MAPS ------------------- #

park_hr_rate_map = {
    'angels_stadium': 1.05, 'angel_stadium': 1.05, 'minute_maid_park': 1.06, 'coors_field': 1.30,
    'yankee_stadium': 1.19, 'fenway_park': 0.97, 'rogers_centre': 1.10, 'tropicana_field': 0.85,
    'camden_yards': 1.13, 'guaranteed_rate_field': 1.18, 'progressive_field': 1.01,
    'comerica_park': 0.96, 'kauffman_stadium': 0.98, 'globe_life_field': 1.00, 'dodger_stadium': 1.10,
    'oakland_coliseum': 0.82, 't-mobile_park': 0.86, 'tmobile_park': 0.86, 'oracle_park': 0.82, 'wrigley_field': 1.12,
    'great_american_ball_park': 1.26, 'american_family_field': 1.17, 'pnc_park': 0.87, 'busch_stadium': 0.87,
    'truist_park': 1.06, 'loan_depot_park': 0.86, 'loandepot_park': 0.86, 'citi_field': 1.05, 'nationals_park': 1.05,
    'petco_park': 0.85, 'chase_field': 1.06, 'citizens_bank_park': 1.19, 'sutter_health_park': 1.12,
    'target_field': 0.98
}
# All the team code -> park name mapping
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
    'MIA': 'loandepot_park',
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
    'SEA': 't-mobile_park',
    'TEX': 'globe_life_field',
    'ARI': 'chase_field',
    'COL': 'coors_field',
    'PIT': 'pnc_park',
    'STL': 'busch_stadium',
    'BAL': 'camden_yards',
    'WSH': 'nationals_park',
    'ATH': 'sutter_health_park' # For Las Vegas/Oakland alternate
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
    'loan_depot_park': 'closed', 'loandepot_park': 'closed', 'globe_life_field': 'open', 'tropicana_field': 'closed',
    'american_family_field': 'open'
}

mlb_team_city_map = {
    'ANA': 'Anaheim', 'ARI': 'Phoenix', 'ATL': 'Atlanta', 'BAL': 'Baltimore', 'BOS': 'Boston',
    'CHC': 'Chicago', 'CIN': 'Cincinnati', 'CLE': 'Cleveland', 'COL': 'Denver', 'CWS': 'Chicago',
    'DET': 'Detroit', 'HOU': 'Houston', 'KC': 'Kansas City', 'LAA': 'Anaheim', 'LAD': 'Los Angeles',
    'MIA': 'Miami', 'MIL': 'Milwaukee', 'MIN': 'Minneapolis', 'NYM': 'New York', 'NYY': 'New York',
    'OAK': 'Oakland', 'PHI': 'Philadelphia', 'PIT': 'Pittsburgh', 'SD': 'San Diego', 'SEA': 'Seattle',
    'SF': 'San Francisco', 'STL': 'St. Louis', 'TB': 'St. Petersburg', 'TEX': 'Arlington', 'TOR': 'Toronto',
    'WSH': 'Washington', 'ATH': 'Las Vegas'
}

# ------------------- STREAMLIT UI ------------------- #
st.title("⚾ Statcast MLB HR Analyzer — All Features, Context & Rolling Stats")
st.markdown("""
Fetch MLB Statcast batted ball events, engineer **rolling, park, weather, matchup, pitch mix, and advanced context features for HR prediction**.
- **No extra uploads needed** — all from Statcast API & Weather API.
- Outputs: event-level & player-level CSVs, plus logistic weights for every feature.
""")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=60))
with col2:
    end_date = st.date_input("End Date", value=datetime.today())

run_query = st.button("Fetch Statcast Data and Run Analyzer")

if run_query:
    st.info("Pulling Statcast data... (can take a few min for big ranges)")
    df = statcast(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    st.success(f"Loaded {len(df)} events from {start_date} to {end_date}")

    # Target events only
    target_events = ['single', 'double', 'triple', 'home_run', 'field_out']
    df = df[df['events'].isin(target_events)].reset_index(drop=True)

    df['game_date'] = pd.to_datetime(df['game_date'])
    # Add park name based on home team code (using mapping)
    df['park'] = df['home_team'].map(team_code_to_park).str.lower()
    # Fallback: if park is still NaN, try the home_team name as a last resort
    df['park'] = df['park'].fillna(df['home_team'].str.lower().str.replace(' ', '_'))
    df['batter_id'] = df['batter']
    df['pitcher_id'] = df['pitcher']
    df['home_team_code'] = df['home_team']

    # Park context
    df['park_hr_rate'] = df['park'].map(park_hr_rate_map).fillna(1.00)
    df['park_altitude'] = df['park'].map(park_altitude_map).fillna(0)
    df['roof_status'] = df['park'].map(roof_status_map).fillna("open")

    # Weather cache (by city/date)
    st.write("Merging weather data (may take a while for large date ranges)...")
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

    # ------------- ADVANCED ROLLING STATS, X-STATS, PITCH MIX ---------------- #

    ROLL = [3, 5, 7, 14]
    # Dynamically include all advanced stats if present in the download
    candidate_batter_stats = [
        'launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value', 'iso_value',
        'xwoba', 'xba', 'xslg'
    ]
    candidate_pitcher_stats = [
        'launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value', 'iso_value',
        'xwoba', 'xba', 'xslg'
    ]
    batter_stats = [c for c in candidate_batter_stats if c in df.columns]
    pitcher_stats = [c for c in candidate_pitcher_stats if c in df.columns]

    # Warn if advanced stats missing
    missing_bstats = set(candidate_batter_stats) - set(df.columns)
    missing_pstats = set(candidate_pitcher_stats) - set(df.columns)
    if missing_bstats:
        st.warning(f"Missing advanced batter stats from Statcast: {missing_bstats}")
    if missing_pstats:
        st.warning(f"Missing advanced pitcher stats from Statcast: {missing_pstats}")

    # Rolling feature function
    def add_rolling(df, group, stats, windows, prefix):
        for stat in stats:
            for w in windows:
                col = f"{prefix}_{stat}_{w}"
                df[col] = df.groupby(group)[stat].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        return df

    df = add_rolling(df, 'batter_id', batter_stats, ROLL, 'B')
    df = add_rolling(df, 'pitcher_id', pitcher_stats, ROLL, 'P')

    # --- Rolling pitch mix % (by batter/pitcher, by pitch_type) ---
    for pitch_col in ['pitch_type', 'pitch_name']:
        if pitch_col in df.columns:
            pitch_types = df[pitch_col].dropna().unique()
            for pt in pitch_types:
                col_b = f"B_pitchmix_{pt}"
                col_p = f"P_pitchmix_{pt}"
                # % of last N pitches that were this type for batter/pitcher
                for w in ROLL:
                    b_col = f"{col_b}_{w}"
                    p_col = f"{col_p}_{w}"
                    df[b_col] = (
                        df.groupby('batter_id')[pitch_col]
                        .transform(lambda x: x.shift(1).rolling(w, min_periods=1).apply(lambda y: (y == pt).mean() if len(y) else np.nan))
                    )
                    df[p_col] = (
                        df.groupby('pitcher_id')[pitch_col]
                        .transform(lambda x: x.shift(1).rolling(w, min_periods=1).apply(lambda y: (y == pt).mean() if len(y) else np.nan))
                    )

    # -------------- CONTEXT FEATURES, HANDEDNESS, PARK/PITCH/PLATOON RATES -------------- #
    df['handed_matchup'] = df['stand'].astype(str) + df['p_throws'].astype(str)
    df['primary_pitch'] = df['pitch_type'] if 'pitch_type' in df.columns else df.get('pitch_name')
    df['platoon'] = (df['stand'] != df['p_throws']).astype(int)
    df['game_hour'] = df['game_date'].dt.hour
    df['is_day'] = (df['game_hour'] < 18).astype(int)
    df['pull_air'] = ((df['bb_type'] == 'fly_ball') & (df['hc_x'] < 125)).astype(int) if 'hc_x' in df.columns and 'bb_type' in df.columns else 0
    df['flyball'] = (df['bb_type'] == 'fly_ball').astype(int) if 'bb_type' in df.columns else 0
    df['line_drive'] = (df['bb_type'] == 'line_drive').astype(int) if 'bb_type' in df.columns else 0
    df['groundball'] = (df['bb_type'] == 'ground_ball').astype(int) if 'bb_type' in df.columns else 0
    df['pull_side'] = (df['hc_x'] < 125).astype(int) if 'hc_x' in df.columns else 0
    df['hr_outcome'] = (df['events'] == 'home_run').astype(int)

    # --- Park season-to-date HR rate ---
    park_hr_df = df.groupby('park').agg(
        park_hr_count = ('hr_outcome', 'sum'),
        park_event_count = ('hr_outcome', 'count')
    ).reset_index()
    park_hr_df['park_hr_rate_season'] = park_hr_df['park_hr_count'] / park_hr_df['park_event_count']
    df = df.merge(park_hr_df[['park', 'park_hr_rate_season']], on='park', how='left')

    # --- Handedness HR rate (by stand/p_throws, season to date) ---
    if 'handed_matchup' in df.columns:
        hand_hr_df = df.groupby('handed_matchup').agg(
            hand_hr_count = ('hr_outcome', 'sum'),
            hand_event_count = ('hr_outcome', 'count')
        ).reset_index()
        hand_hr_df['handed_hr_rate_season'] = hand_hr_df['hand_hr_count'] / hand_hr_df['hand_event_count']
        df = df.merge(hand_hr_df[['handed_matchup', 'handed_hr_rate_season']], on='handed_matchup', how='left')

    # --- Pitch type HR rate (by pitch_type, season to date) ---
    if 'pitch_type' in df.columns:
        pitch_hr_df = df.groupby('pitch_type').agg(
            pitch_hr_count = ('hr_outcome', 'sum'),
            pitch_event_count = ('hr_outcome', 'count')
        ).reset_index()
        pitch_hr_df['pitch_type_hr_rate_season'] = pitch_hr_df['pitch_hr_count'] / pitch_hr_df['pitch_event_count']
        df = df.merge(pitch_hr_df[['pitch_type', 'pitch_type_hr_rate_season']], on='pitch_type', how='left')
    elif 'pitch_name' in df.columns:
        pitch_hr_df = df.groupby('pitch_name').agg(
            pitch_hr_count = ('hr_outcome', 'sum'),
            pitch_event_count = ('hr_outcome', 'count')
        ).reset_index()
        pitch_hr_df['pitch_type_hr_rate_season'] = pitch_hr_df['pitch_hr_count'] / pitch_hr_df['pitch_event_count']
        df = df.merge(pitch_hr_df[['pitch_name', 'pitch_type_hr_rate_season']], on='pitch_name', how='left')

    st.success("Feature engineering complete.")

    # ----------- EVENT-LEVEL EXPORT ----------- #
    # Output all columns, but limit display to avoid crashes on download
    display_cols = [
        'game_date', 'batter', 'batter_id', 'pitcher', 'pitcher_id', 'events', 'description',
        'stand', 'p_throws', 'park_hr_rate', 'park_altitude', 'roof_status',
        'temp', 'wind_mph', 'wind_dir', 'humidity', 'condition',
        'handed_matchup', 'primary_pitch', 'platoon', 'is_day',
        'pull_air', 'flyball', 'line_drive', 'groundball', 'pull_side',
        'park_hr_rate_season', 'handed_hr_rate_season', 'pitch_type_hr_rate_season', 'hr_outcome'
    ]

    # Add all dynamically engineered stats
    for c in df.columns:
        if (c.startswith('B_') or c.startswith('P_') or c.startswith('pitchmix_')) and c not in display_cols:
            display_cols.append(c)
        if (c.startswith('B_pitchmix_') or c.startswith('P_pitchmix_')) and c not in display_cols:
            display_cols.append(c)

    # Remove duplicates while keeping order
    from collections import OrderedDict
    display_cols = list(OrderedDict.fromkeys([col for col in display_cols if col in df.columns]))

    event_df = df[display_cols].copy()

    st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
    st.dataframe(event_df.head(20))
    st.download_button(
        "⬇️ Download Event-Level CSV",
        data=event_df.to_csv(index=False),
        file_name="event_level_hr_features.csv"
    )

    # ----------- PLAYER-LEVEL EXPORT ----------- #
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

    # ========== LOGISTIC REGRESSION (with progress) ========== #
    st.markdown("#### Logistic Regression Weights (Standardized Features)")
    # Select all numerics except obvious non-predictive IDs/descriptions
    logit_features = [
        col for col in event_df.columns if (
            pd.api.types.is_numeric_dtype(event_df[col]) and
            col not in ['batter', 'pitcher', 'batter_id', 'pitcher_id', 'game_hour']
            and not col.endswith('_id')
            and col != 'hr_outcome'
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
