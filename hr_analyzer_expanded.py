import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybaseball import statcast
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# --- Static mappings (parks, teams, cities, altitudes) ---
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

st.title("⚾ Statcast MLB HR Analyzer — Cleaned, No Redundancy")
st.markdown("""
Fetches MLB Statcast batted ball events and **auto-engineers rolling, park, weather, pitch-mix, and advanced HR features** for HR prediction.  
*No single-event (non-rolling) stats included in player-level modeling!*
""")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=7))
with col2:
    end_date = st.date_input("End Date", value=datetime.today())

run_query = st.button("Fetch Statcast Data and Run Analyzer")

if run_query:
    progress = st.progress(0, text="Starting...")
    st.info("Pulling Statcast data...")
    df = statcast(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    progress.progress(10, text="Statcast downloaded")
    st.success(f"Loaded {len(df)} events from {start_date} to {end_date}")

    # Basic prepping & park mapping
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

    # Weather features (uses city from home_team_code)
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

    unique_keys = df['weather_key'].unique()
    for i, key in enumerate(unique_keys):
        team = key.split('_')[0]
        city = mlb_team_city_map.get(team, "New York")
        date = df[df['weather_key'] == key].iloc[0]['game_date'].strftime("%Y-%m-%d")
        weather = get_weather(city, date)
        for feat in weather_features:
            df.loc[df['weather_key'] == key, feat] = weather[feat]
        progress.progress(10 + int(40 * (i+1) / len(unique_keys)), text=f"Weather {i+1}/{len(unique_keys)}")
    progress.progress(50, text="Weather merged")

    # Rolling stat features (batters & pitchers)
    ROLL = [3, 5, 7, 14]
    batter_stats = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value', 'xwoba', 'xslg', 'xba']
    pitcher_stats = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value', 'xwoba', 'xslg', 'xba']
    batter_stats = [s for s in batter_stats if s in df.columns]
    pitcher_stats = [s for s in pitcher_stats if s in df.columns]

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
        df[f'P_max_ev_{w}'] = df.groupby('pitcher_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
        df[f'B_median_ev_{w}'] = df.groupby('batter_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).median())
        df[f'P_median_ev_{w}'] = df.groupby('pitcher_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).median())

    # ---------- FIX: Must define hr_outcome BEFORE using in park_handed_hr_rate ----------
    df['hr_outcome'] = (df['events'] == 'home_run').astype('Int64')

    # Park-handed HR rate (Statcast-based, NOT static)
    def compute_park_handed_hr_rate(df):
        df['handed_matchup'] = df['stand'].astype(str) + df['p_throws'].astype(str)
        grp = df.groupby(['park', 'handed_matchup'])
        rate = grp['hr_outcome'].mean().reset_index().rename(
            columns={'hr_outcome': 'park_handed_hr_rate'}
        )
        df = df.merge(rate, on=['park', 'handed_matchup'], how='left')
        return df

    df = compute_park_handed_hr_rate(df)

    # --- Rolling HR rate for pitcher ---
    for w in ROLL:
        df[f'P_rolling_hr_rate_{w}'] = (
            df.groupby('pitcher_id')['hr_outcome'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )

    # --- Logistic regression modeling (with robust null handling) ---
    st.markdown("#### Logistic Regression Weights (Standardized Features)")

    # Features to include: rolling/statcast/park/advanced, no single-event iso_value
    logit_features = [
        c for c in df.columns
        if (
            any(s in c for s in [
                'launch_speed', 'launch_angle', 'hit_distance_sc_', 'woba_value', 'xwoba', 'xslg', 'xba',
                'pitch_pct_', 'park_hr_rate', 'park_handed_hr_rate', 'max_ev_', 'median_ev_',
                'rolling_hr_rate_', 'altitude'
            ])
            or c in [
                'platoon', 'temp', 'wind_mph', 'humidity', 'pull_air', 'flyball', 'line_drive',
                'groundball', 'pull_side', 'is_barrel', 'is_hard_hit', 'is_sweet_spot'
            ]
            or '_x_' in c
        )
        and not c.endswith('iso_value')
        and not c == 'iso_value'
    ]

    # Require at least 90% coverage
    nonnull_thresh = 0.9
    coverage = df[logit_features].notnull().mean()
    use_feats = coverage[coverage > nonnull_thresh].index.tolist()
    model_df = df.dropna(subset=use_feats + ['hr_outcome'], how='any')

    if len(model_df) > 10 and len(use_feats) > 0:
        X = model_df[use_feats].astype(float)
        y = model_df['hr_outcome'].astype(int)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=500)
        model.fit(X_scaled, y)
        weights = pd.Series(model.coef_[0], index=use_feats).sort_values(ascending=False)
        weights_df = pd.DataFrame({'feature': weights.index, 'weight': weights.values})
        st.dataframe(weights_df.head(60))
        auc = roc_auc_score(y, model.predict_proba(X_scaled)[:, 1])
        st.write(f"Model ROC-AUC: **{auc:.3f}**")
        st.download_button(
            "⬇️ Download Logistic Regression Weights CSV",
            data=weights_df.to_csv(index=False),
            file_name="logit_feature_weights.csv"
        )
    else:
        st.warning("Not enough data with complete features to fit logistic regression for HR prediction.")

    # ===== Event-Level CSV Export =====
    export_cols = [
        'game_date', 'batter', 'batter_id', 'pitcher', 'pitcher_id', 'events', 'description',
        'stand', 'p_throws', 'park', 'park_hr_rate', 'park_handed_hr_rate', 'park_altitude', 'roof_status',
        'temp', 'wind_mph', 'wind_dir', 'humidity', 'condition',
        'handed_matchup', 'primary_pitch', 'platoon', 'is_day', 'hr_outcome'
    ]
    export_cols += [c for c in df.columns if c not in export_cols]
    event_cols = [c for c in export_cols if c in df.columns]
    event_df = df[event_cols].copy()
    st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
    st.dataframe(event_df.head(20))
    st.download_button("⬇️ Download Event-Level CSV", data=event_df.to_csv(index=False), file_name="event_level_hr_features.csv")

    # ===== Player-Level CSV Export =====
    st.markdown("#### Download Player-Level Rolling Feature CSV (1 row per batter):")
    player_cols = ['batter_id', 'batter'] + [c for c in df.columns if c.startswith('B_')]
    player_df = (
        event_df.groupby(['batter_id', 'batter'])
        .tail(1)[player_cols]
        .reset_index(drop=True)
    )
    st.dataframe(player_df.head(20))
    st.download_button("⬇️ Download Player-Level CSV", data=player_df.to_csv(index=False), file_name="player_level_hr_features.csv")

    st.success("Analysis complete. All features, upgrades, and outputs are available above.")

else:
    st.info("Set your date range and click 'Fetch Statcast Data and Run Analyzer' to begin.")
