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

# --- Static Park/Team/Weather Maps --- #
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
# ------------------------------------------------------------------------ #

st.title("⚾ Statcast MLB HR Analyzer — All Batted Ball, Context, Rolling, Weather & Pitch Mix Features")
st.markdown("""
Fetches MLB Statcast batted ball events and auto-engineers **advanced rolling, park, weather, pitch-mix, custom batted ball, and matchup features for HR prediction**.
Handles all missing columns gracefully.
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
    df['home_team_code'] = df['home_team']
    df['park'] = df['home_team_code'].map(team_code_to_park).fillna(df['home_team'].str.lower().str.replace(' ', '_'))
    df['batter_id'] = df['batter']
    df['pitcher_id'] = df['pitcher']

    # Park/context features
    df['park_hr_rate'] = df['park'].map(park_hr_rate_map).fillna(1.00)
    df['park_altitude'] = df['park'].map(park_altitude_map).fillna(0)
    df['roof_status'] = df['park'].map(roof_status_map).fillna("open")

    # Weather cache (by city/date)
    st.write("Merging weather data (may take a while for large date ranges)...")
    weather_features = ['temp', 'wind_mph', 'wind_dir', 'humidity', 'condition']
    df['weather_key'] = df['home_team_code'] + "_" + df['game_date'].dt.strftime("%Y%m%d")

    @st.cache_data(show_spinner=False)
    def get_weather(city, date):
        # Place your API key in .streamlit/secrets.toml
        api_key = st.secrets["weather"]["api_key"]
        url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={date}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and resp.text.strip():
                data = resp.json()
                # Get the most common MLB game hour (19:00 local)
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

    # --- Advanced Statcast features: ensure always present --- #
    for col in ['xwoba', 'xslg', 'xba']:
        if col not in df.columns:
            st.warning(f"Missing advanced stat: {col}")
            df[col] = np.nan

    # --- Custom batted ball flags: robust to missing (use .fillna) ---
    df['is_barrel'] = (
        (df['launch_speed'].fillna(-999) >= 98) &
        (df['launch_angle'].fillna(-999).between(26, 30))
    ).astype('Int64')
    df['is_sweet_spot'] = df['launch_angle'].fillna(-999).between(8, 32).astype('Int64')
    df['is_hard_hit'] = (df['launch_speed'].fillna(-999) >= 95).astype('Int64')

    # Rolling max/median EV for batters/pitchers
    for w in ROLL:
        df[f'B_max_ev_{w}'] = (
            df.groupby('batter_id')['launch_speed']
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
        )
        df[f'B_median_ev_{w}'] = (
            df.groupby('batter_id')['launch_speed']
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).median())
        )
        df[f'P_max_ev_{w}'] = (
            df.groupby('pitcher_id')['launch_speed']
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
        )
        df[f'P_median_ev_{w}'] = (
            df.groupby('pitcher_id')['launch_speed']
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).median())
        )
        # Rolling flyball distance, only for flyball events
        df[f'B_flyball_dist_{w}'] = (
            df.assign(is_fb=(df['bb_type'] == 'fly_ball'))
            .groupby('batter_id')
            .apply(lambda g: g['hit_distance_sc'].where(g['is_fb']).shift(1).rolling(w, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        df[f'P_flyball_dist_{w}'] = (
            df.assign(is_fb=(df['bb_type'] == 'fly_ball'))
            .groupby('pitcher_id')
            .apply(lambda g: g['hit_distance_sc'].where(g['is_fb']).shift(1).rolling(w, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

    # --- Robust context/batted ball flags ---
    for flag_col, expr in {
        'platoon': lambda d: (d['stand'] != d['p_throws']).astype(int),
        'handed_matchup': lambda d: d['stand'].astype(str) + d['p_throws'].astype(str),
        'primary_pitch': lambda d: d['pitch_type'] if 'pitch_type' in d else np.nan,
        'game_hour': lambda d: d['game_date'].dt.hour,
        'is_day': lambda d: (d['game_date'].dt.hour < 18).astype(int),
        'pull_air': lambda d: (((d['bb_type'] == 'fly_ball') & (d['hc_x'].fillna(-1) < 125)).astype('Int64')),
        'flyball': lambda d: (d['bb_type'] == 'fly_ball').astype('Int64'),
        'line_drive': lambda d: (d['bb_type'] == 'line_drive').astype('Int64'),
        'groundball': lambda d: (d['bb_type'] == 'ground_ball').astype('Int64'),
        'pull_side': lambda d: (d['hc_x'].fillna(-1) < 125).astype('Int64'),
        'hr_outcome': lambda d: (d['events'] == 'home_run').astype(int)
    }.items():
        df[flag_col] = expr(df)

    # --- Rolling stats for both batters and pitchers --- #
    ROLL = [3, 5, 7, 14]
    # Key advanced stats
    batter_stats = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value', 'iso_value', 'xwoba', 'xslg', 'xba']
    pitcher_stats = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value', 'iso_value', 'xwoba', 'xslg', 'xba']
    custom_batted_flags = ['is_barrel', 'is_sweet_spot', 'is_hard_hit']

    # Only keep stats that exist
    batter_stats = [s for s in batter_stats if s in df.columns]
    pitcher_stats = [s for s in pitcher_stats if s in df.columns]
    custom_batted_flags = [s for s in custom_batted_flags if s in df.columns]

    def add_rolling(df, group, stats, windows, prefix):
        for stat in stats:
            for w in windows:
                col = f"{prefix}_{stat}_{w}"
                df[col] = (
                    df.groupby(group)[stat]
                      .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                )
        return df

    # Rolling mean stats (advanced)
    df = add_rolling(df, 'batter_id', batter_stats + custom_batted_flags, ROLL, 'B')
    df = add_rolling(df, 'pitcher_id', pitcher_stats + custom_batted_flags, ROLL, 'P')

    # Rolling max/median exit velocity, flyball distance
    for w in ROLL:
        df[f'B_max_ev_{w}'] = df.groupby('batter_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
        df[f'B_median_ev_{w}'] = df.groupby('batter_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).median())
        df[f'P_max_ev_{w}'] = df.groupby('pitcher_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
        df[f'P_median_ev_{w}'] = df.groupby('pitcher_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).median())

        # Rolling avg flyball distance (batters)
        df[f'B_flyball_dist_{w}'] = (
            df.groupby('batter_id')
            .apply(lambda g: g['hit_distance_sc'].shift(1).where(g['bb_type']=='fly_ball').rolling(w, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        df[f'P_flyball_dist_{w}'] = (
            df.groupby('pitcher_id')
            .apply(lambda g: g['hit_distance_sc'].shift(1).where(g['bb_type']=='fly_ball').rolling(w, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

    # ---- Rolling pitch mix % (object dtype, robust) ----
    if 'pitch_type' in df.columns:
        pitch_types = df['pitch_type'].dropna().unique()
        for pt in pitch_types:
            for w in ROLL:
                col_b = f"B_pitch_{pt}_{w}"
                col_p = f"P_pitch_{pt}_{w}"
                df[col_b] = (
                    df.groupby('batter_id')['pitch_type']
                      .transform(lambda x: x.shift(1).rolling(w, min_periods=1)
                      .apply(lambda y: pd.Series(y).eq(pt).mean() if len(y) else np.nan, raw=False))
                )
                df[col_p] = (
                    df.groupby('pitcher_id')['pitch_type']
                      .transform(lambda x: x.shift(1).rolling(w, min_periods=1)
                      .apply(lambda y: pd.Series(y).eq(pt).mean() if len(y) else np.nan, raw=False))
                )

    st.success("Feature engineering complete.")

    # ===== EVENT-LEVEL EXPORT ===== #
    # All new and original features, robust to missing cols
    export_cols = [
        'game_date', 'batter', 'batter_id', 'pitcher', 'pitcher_id', 'events', 'description',
        'stand', 'p_throws', 'park', 'park_hr_rate', 'park_altitude', 'roof_status',
        'temp', 'wind_mph', 'wind_dir', 'humidity', 'condition',
        'handed_matchup', 'primary_pitch', 'platoon', 'is_day'
    ]
    # All rolling/statcast/pitch-mix/ball flags
    extra_cols = []
    for stat in batter_stats + custom_batted_flags:
        for w in ROLL:
            extra_cols.append(f"B_{stat}_{w}")
    for stat in pitcher_stats + custom_batted_flags:
        for w in ROLL:
            extra_cols.append(f"P_{stat}_{w}")
    # All pitch type %s
    if 'pitch_type' in df.columns:
        pitch_types = df['pitch_type'].dropna().unique()
        for pt in pitch_types:
            for w in ROLL:
                extra_cols.append(f"B_pitch_{pt}_{w}")
                extra_cols.append(f"P_pitch_{pt}_{w}")

    # Custom rolling max/median/flyball features
    for w in ROLL:
        extra_cols += [
            f'B_max_ev_{w}', f'B_median_ev_{w}', f'B_flyball_dist_{w}',
            f'P_max_ev_{w}', f'P_median_ev_{w}', f'P_flyball_dist_{w}'
        ]

    # Ball-in-play flags and HR outcome
    extra_cols += ['is_barrel', 'is_sweet_spot', 'is_hard_hit', 'pull_air', 'flyball', 'line_drive', 'groundball', 'pull_side', 'hr_outcome']

    event_cols = export_cols + extra_cols
    event_cols = [c for c in event_cols if c in df.columns]
    event_df = df[event_cols].copy()
    st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
    st.dataframe(event_df.head(20))
    st.download_button("⬇️ Download Event-Level CSV", data=event_df.to_csv(index=False), file_name="event_level_hr_features.csv")

    # ===== PLAYER-LEVEL EXPORT ===== #
    st.markdown("#### Download Player-Level Rolling Feature CSV (1 row per batter):")
    player_cols = ['batter_id', 'batter'] + [c for c in event_df.columns if c.startswith('B_')]
    player_df = (
        event_df.groupby(['batter_id', 'batter'])
        .tail(1)[player_cols]
        .reset_index(drop=True)
    )
    st.dataframe(player_df.head(20))
    st.download_button("⬇️ Download Player-Level CSV", data=player_df.to_csv(index=False), file_name="player_level_hr_features.csv")

    # ===== LOGISTIC REGRESSION (with scaling/weights, robust) ===== #
    st.markdown("#### Logistic Regression Weights (Standardized Features)")
    logit_features = [
        c for c in event_df.columns if (
            any(s in c for s in [
                'launch_speed', 'launch_angle', 'hit_distance', 'woba_value', 'iso_value', 'xwoba', 'xslg', 'xba',
                'pitch_', 'is_barrel', 'is_sweet_spot', 'is_hard_hit', 'max_ev', 'median_ev', 'flyball_dist'
            ])
            or c in ['park_hr_rate', 'platoon', 'temp', 'wind_mph', 'humidity', 'pull_air', 'flyball', 'line_drive', 'groundball', 'pull_side']
        )
    ]
    model_df = event_df.dropna(subset=logit_features + ['hr_outcome'], how='any')
    if len(model_df) > 10:
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
    else:
        st.warning("Not enough data with complete features to fit logistic regression for HR prediction.")

    st.success("Full analysis done! All context, advanced rolling, pitch mix, custom batted ball, and robust weighting included.")

else:
    st.info("Set your date range and click 'Fetch Statcast Data and Run Analyzer' to begin.")
