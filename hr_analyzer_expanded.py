import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
from pybaseball import statcast
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_selection import RFECV
import xgboost as xgb
import requests

# ========== CONTEXT MAPS ==========
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

# ---------- UTILITY FUNCTIONS ----------
def wind_dir_to_angle(wind_dir):
    directions = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5,
        'SE': 135, 'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    if pd.isna(wind_dir):
        return np.nan
    wind_dir = str(wind_dir).upper()
    for d, angle in directions.items():
        if d in wind_dir:
            return angle
    return np.nan

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
    return {'temp': None, 'wind_mph': None, 'wind_dir': None, 'humidity': None, 'condition': None}

def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

def rolling_pitch_type_hr(df, id_col, pitch_col, window):
    out = np.full(len(df), np.nan)
    df = df.reset_index(drop=True)
    grouped = df.groupby([id_col, pitch_col])
    for _, group_idx in grouped.groups.items():
        group_idx = list(group_idx)
        vals = df.loc[group_idx, 'hr_outcome'].shift(1).rolling(window, min_periods=1).mean()
        out[group_idx] = vals
    return out

def robust_numeric_columns(df):
    cols = []
    for c in df.columns:
        try:
            dt = pd.api.types.pandas_dtype(df[c].dtype)
            if (np.issubdtype(dt, np.number) or pd.api.types.is_numeric_dtype(df[c])) and not pd.api.types.is_bool_dtype(df[c]) and df[c].nunique() > 1:
                cols.append(c)
        except Exception:
            continue
    return cols

# ========== TAB 1 ==========
st.set_page_config("MLB HR Analyzer", layout="wide")
tab1, tab2 = st.tabs(["1️⃣ Fetch & Feature Engineer Data", "2️⃣ Upload & Analyze"])

with tab1:
    st.header("Fetch Statcast Data & Generate Features")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", value=datetime.today())

    # --------- Matchups (Lineup) Upload for context (optional) ----------
    st.markdown("##### (Optional) Upload Today's Matchups/Lineups CSV (for city, stadium, time, weather context)")
    uploaded_lineups = st.file_uploader("Upload Today's Matchups CSV", type="csv", key="lineupsup")

    fetch_btn = st.button("Fetch Statcast, Feature Engineer, and Download", type="primary")
    progress = st.empty()

    if fetch_btn:
        progress.progress(5, "Fetching Statcast data...")
        df = statcast(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        progress.progress(10, "Loaded Statcast")
        st.write(f"Loaded {len(df)} raw Statcast events.")
        if len(df) == 0:
            st.error("No data! Try different dates.")
            st.stop()

        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])

        # Filter events
        valid_events = [
            'single', 'double', 'triple', 'homerun', 'home_run', 'field_out',
            'force_out', 'grounded_into_double_play', 'fielders_choice_out',
            'pop_out', 'lineout', 'flyout', 'sac_fly', 'sac_fly_double_play'
        ]
        if 'events' in df.columns:
            df = df[df['events'].str.lower().str.replace(' ', '').isin(valid_events)].copy()

        # HR outcome
        if 'hr_outcome' not in df.columns:
            if 'events' in df.columns:
                df['hr_outcome'] = df['events'].astype(str).str.lower().str.replace(' ', '').isin(['homerun', 'home_run']).astype(int)
            else:
                df['hr_outcome'] = np.nan

        # Park/team mapping
        if 'home_team_code' in df.columns:
            df['home_team_code'] = df['home_team_code'].astype(str).str.upper()
        if 'park' not in df.columns:
            if 'home_team_code' in df.columns:
                df['park'] = df['home_team_code'].map(team_code_to_park)
        if 'park' in df.columns and df['park'].isnull().any() and 'home_team' in df.columns:
            df['park'] = df['park'].fillna(df['home_team'].str.lower().str.replace(' ', '_'))
        elif 'home_team' in df.columns and 'park' not in df.columns:
            df['park'] = df['home_team'].str.lower().str.replace(' ', '_')

        # -------- Integrate matchups CSV if uploaded --------
        # This will override city/stadium/park if present in your upload
        if uploaded_lineups:
            lineups = pd.read_csv(uploaded_lineups)
            # Try to merge by date/team
            if 'game_date' in df.columns and 'team code' in lineups.columns:
                df['game_date'] = df['game_date'].astype(str)
                lineups['game_date'] = lineups['game_date'].astype(str)
                if 'home_team_code' in df.columns:
                    merged = df.merge(
                        lineups[['team code', 'game_date', 'city', 'stadium', 'time', 'weather']],
                        left_on=['home_team_code', 'game_date'],
                        right_on=['team code', 'game_date'],
                        how='left'
                    )
                    df['city'] = merged['city']
                    df['stadium'] = merged['stadium']
                    df['time'] = merged['time']
                    df['weather'] = merged['weather']
            if 'stadium' in df.columns:
                df['park'] = df['stadium'].str.lower().str.replace(' ', '_')

        if 'park' not in df.columns:
            st.error("Could not determine ballpark from your data (missing 'park', 'home_team_code', and 'home_team').")
            st.stop()
        df['park_hr_rate'] = df['park'].map(park_hr_rate_map).fillna(1.0)
        df['park_altitude'] = df['park'].map(park_altitude_map).fillna(0)
        df['roof_status'] = df['park'].map(roof_status_map).fillna("open")
        progress.progress(20, "Park/team/stadium context merged")

        # -------- Weather Integration --------
        weather_features = ['temp', 'wind_mph', 'wind_dir', 'humidity', 'condition']
        if 'city' in df.columns and 'game_date' in df.columns:
            df['weather_key'] = df['city'].fillna('') + "_" + df['game_date'].astype(str)
            unique_keys = df['weather_key'].unique()
            for i, key in enumerate(unique_keys):
                if '_' not in key: continue
                city, date = key.split('_', 1)
                if not city or not date: continue
                if 'weather' in df.columns and pd.notnull(df.loc[df['weather_key'] == key, 'weather']).any():
                    continue  # Assume weather already filled
                weather = get_weather(city, date)
                for feat in weather_features:
                    df.loc[df['weather_key'] == key, feat] = weather[feat]
                progress.progress(20 + int(30 * (i+1) / len(unique_keys)), text=f"Weather {i+1}/{len(unique_keys)}")
            progress.progress(50, "Weather merged (city/date).")
        else:
            if 'home_team_code' in df.columns and 'game_date' in df.columns:
                df['weather_key'] = df['home_team_code'] + "_" + df['game_date'].astype(str)
                unique_keys = df['weather_key'].unique()
                for i, key in enumerate(unique_keys):
                    team = key.split('_')[0]
                    city = mlb_team_city_map.get(team, "New York")
                    date = df[df['weather_key'] == key].iloc[0]['game_date']
                    weather = get_weather(city, str(date))
                    for feat in weather_features:
                        df.loc[df['weather_key'] == key, feat] = weather[feat]
                    progress.progress(20 + int(30 * (i+1) / len(unique_keys)), text=f"Weather {i+1}/{len(unique_keys)}")
                progress.progress(50, "Weather merged (fallback).")
            else:
                for feat in weather_features:
                    df[feat] = None

        df['wind_dir_angle'] = df['wind_dir'].apply(wind_dir_to_angle)
        df['wind_dir_sin'] = np.sin(np.deg2rad(df['wind_dir_angle']))
        df['wind_dir_cos'] = np.cos(np.deg2rad(df['wind_dir_angle']))

        # Advanced Statcast metrics and rolling splits
        if 'batter_id' not in df.columns and 'batter' in df.columns:
            df['batter_id'] = df['batter']
        if 'pitcher_id' not in df.columns and 'pitcher' in df.columns:
            df['pitcher_id'] = df['pitcher']

        if 'barrel' in df.columns and 'batter_id' in df.columns:
            df['barrel_rate_20'] = df.groupby('batter_id')['barrel'].transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
        if 'launch_speed' in df.columns and 'batter_id' in df.columns:
            df['hard_hit_rate_20'] = df.groupby('batter_id')['launch_speed'].transform(lambda x: (x.shift(1) >= 95).rolling(20, min_periods=5).mean())
        if 'launch_angle' in df.columns and 'batter_id' in df.columns:
            df['sweet_spot_rate_20'] = df.groupby('batter_id')['launch_angle'].transform(lambda x: x.shift(1).between(8, 32).rolling(20, min_periods=5).mean())

        if 'stand' in df.columns and 'wind_dir_angle' in df.columns:
            def relative_wind_angle(row):
                try:
                    if row['stand'] == 'L':
                        return (row['wind_dir_angle'] - 45) % 360
                    else:
                        return (row['wind_dir_angle'] - 135) % 360
                except Exception:
                    return np.nan
            df['relative_wind_angle'] = df.apply(relative_wind_angle, axis=1)
            df['relative_wind_sin'] = np.sin(np.deg2rad(df['relative_wind_angle']))
            df['relative_wind_cos'] = np.cos(np.deg2rad(df['relative_wind_angle']))

        if 'park' in df.columns and 'stand' in df.columns:
            for w in [7, 14, 30]:
                col_name = f'park_hand_HR_{w}'
                df[col_name] = df.groupby(['park', 'stand'])['hr_outcome'].transform(lambda x: x.shift(1).rolling(w, min_periods=5).mean())

        roll_windows = [3, 5, 7, 14]
        batter_cols = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value',
                       'release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z']
        pitcher_cols = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value','release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z']

        batter_feat_dict = {}
        pitcher_feat_dict = {}
        for col in batter_cols:
            if col in df.columns:
                for w in roll_windows:
                    cname = f'B_{col}_{w}'
                    batter_feat_dict[cname] = df.groupby('batter_id')[col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        for col in pitcher_cols:
            if col in df.columns:
                for w in roll_windows:
                    cname = f'P_{col}_{w}'
                    pitcher_feat_dict[cname] = df.groupby('pitcher_id')[col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())

        if 'stand' in df.columns and 'p_throws' in df.columns:
            for w in roll_windows:
                df[f'B_vsP_hand_HR_{w}'] = df.groupby(['batter_id', 'p_throws'])['hr_outcome'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                df[f'P_vsB_hand_HR_{w}'] = df.groupby(['pitcher_id', 'stand'])['hr_outcome'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())

        if 'pitch_type' in df.columns:
            for w in roll_windows:
                df[f'B_pitchtype_HR_{w}'] = rolling_pitch_type_hr(df, 'batter_id', 'pitch_type', w)
                df[f'P_pitchtype_HR_{w}'] = rolling_pitch_type_hr(df, 'pitcher_id', 'pitch_type', w)

        df = pd.concat([df, pd.DataFrame(batter_feat_dict), pd.DataFrame(pitcher_feat_dict)], axis=1)
        df = df.copy()
        progress.progress(80, "Advanced Statcast & context features done")
        df = dedup_columns(df)
        st.success(f"Feature engineering complete! {len(df)} batted ball events.")
        st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
        st.dataframe(df.head(20))
        st.download_button("⬇️ Download Event-Level CSV", data=df.to_csv(index=False), file_name="event_level_hr_features.csv", key="download_statcast_event_level")

        # === LOGISTIC WEIGHTS DOWNLOAD (quick fit for weighting export) ===
        if 'hr_outcome' in df.columns and df['hr_outcome'].nunique() > 1:
            model_features = robust_numeric_columns(df)
            cat_context = [c for c in [
                'park_hr_rate', 'park_altitude', 'temp', 'humidity', 'wind_mph',
                'wind_dir_angle', 'wind_dir_sin', 'wind_dir_cos', 'relative_wind_angle', 'relative_wind_sin', 'relative_wind_cos'
            ] if c in df.columns]
            for c in df.columns:
                if c.startswith("park_hand_HR_"):
                    cat_context.append(c)
            model_features = [c for c in model_features + cat_context if c != 'hr_outcome']

            feature_na_fracs = {c: df[c].isna().mean() for c in model_features}
            model_features = [c for c in model_features if feature_na_fracs[c] < 0.3]
            model_df = df.dropna(subset=model_features + ['hr_outcome'], how='any')
            if model_df['hr_outcome'].nunique() < 2 or len(model_df) < 30:
                st.warning("Not enough HR/non-HR events for logistic regression weights (check date range or missing features).")
            else:
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import GridSearchCV
                X = model_df[model_features].fillna(0)
                y = model_df['hr_outcome'].astype(int)
                grid = GridSearchCV(LogisticRegression(max_iter=200, solver='liblinear'), param_grid={'C': [0.1, 1, 10]}, cv=3, scoring='roc_auc')
                grid.fit(X, y)
                logit = grid.best_estimator_
                weights = pd.DataFrame({
                    'feature': model_features,
                    'weight': logit.coef_[0],
                    'intercept': logit.intercept_[0]
                })
                st.dataframe(weights.sort_values('weight', ascending=False))
                st.download_button(
                    "⬇️ Download Logistic Weights CSV",
                    data=weights.to_csv(index=False),
                    file_name="logit_weights.csv"
                )

# --- (NEW) Generate One-Row-Per-Batter Event-Level CSV for Today's Slate ---
st.markdown("---")
st.subheader("🔮 Generate Today's Batter Event-Level CSV (one row per batter, for live HR picks)")

uploaded_today_lineups = st.file_uploader(
    "Upload Today's Lineups/Matchups CSV (Required for One-Row-Per-Batter Export)",
    type="csv", key="todaylineup"
)
generate_btn = st.button("Generate One-Row-Per-Batter Event-Level CSV")

if generate_btn:
    if not uploaded_today_lineups:
        st.warning("Please upload a matchups/lineups CSV for today (with columns like 'team code', 'player name', etc).")
        st.stop()
    today_lineups = pd.read_csv(uploaded_today_lineups)
    # Clean/normalize player and team columns
    lineup_cols = [c.strip().lower().replace(" ", "_") for c in today_lineups.columns]
    today_lineups.columns = lineup_cols

    # Basic checks
    required_cols = {'player_name', 'mlb_id', 'team_code', 'game_date'}
    if not required_cols.issubset(set(today_lineups.columns)):
        st.error(f"Missing columns: {required_cols - set(today_lineups.columns)}")
        st.stop()
    # Drop duplicate players
    today_batters = today_lineups.drop_duplicates(subset=['mlb_id'])

    # Add context columns (stadium, city, time, park, etc)
    today_batters['park'] = today_batters.get('stadium', today_batters.get('team_code', '')).str.lower().str.replace(' ', '_')
    today_batters['city'] = today_batters.get('city', '')
    today_batters['game_date'] = pd.to_datetime(today_batters['game_date']).dt.strftime("%Y-%m-%d")
    today_batters['park_hr_rate'] = today_batters['park'].map(park_hr_rate_map).fillna(1.0)
    today_batters['park_altitude'] = today_batters['park'].map(park_altitude_map).fillna(0)
    today_batters['roof_status'] = today_batters['park'].map(roof_status_map).fillna("open")
    today_batters['batter_id'] = today_batters['mlb_id'].astype(str)

    # Weather API or static weather values
    for feat in ['temp', 'wind_mph', 'wind_dir', 'humidity', 'condition']:
        today_batters[feat] = None
    unique_keys = today_batters['city'] + "_" + today_batters['game_date']
    for key in unique_keys.unique():
        city, date = key.split("_", 1)
        weather = get_weather(city, date)
        idx = (today_batters['city'] + "_" + today_batters['game_date']) == key
        for feat in ['temp', 'wind_mph', 'wind_dir', 'humidity', 'condition']:
            today_batters.loc[idx, feat] = weather[feat]
    today_batters['wind_dir_angle'] = today_batters['wind_dir'].apply(wind_dir_to_angle)
    today_batters['wind_dir_sin'] = np.sin(np.deg2rad(today_batters['wind_dir_angle']))
    today_batters['wind_dir_cos'] = np.cos(np.deg2rad(today_batters['wind_dir_angle']))

    # Add ALL rolling/stat columns used by backtest/historical CSVs
    full_stat_cols = get_all_stat_rolling_cols()
    for col in full_stat_cols:
        if col not in today_batters.columns:
            today_batters[col] = np.nan

    # Diagnostics: compare columns to historical event-level files
    sample_ev_file = st.file_uploader("Upload Sample Historical Event-Level CSV (for diagnostics/column order)", type="csv", key="colalign")
    hist_cols = None
    if sample_ev_file:
        ev = pd.read_csv(sample_ev_file, nrows=1)
        hist_cols = [c for c in ev.columns if not c.startswith("unnamed")]
        missing_cols = [c for c in hist_cols if c not in today_batters.columns]
        extra_cols = [c for c in today_batters.columns if c not in hist_cols]
        st.info(f"Columns in history but missing in today's export: {missing_cols}")
        st.info(f"Columns in today's export but not in history: {extra_cols}")
        today_batters = today_batters.reindex(columns=hist_cols + [c for c in today_batters.columns if c not in hist_cols])

    st.markdown("#### Sample of Today's One-Row-Per-Batter DataFrame:")
    st.dataframe(today_batters.head(25))
    st.success(f"Created today's event-level file: {len(today_batters)} batters. All rolling/stat columns now present.")

    st.download_button(
        "⬇️ Download Today's Event-Level CSV (for Prediction App)",
        data=today_batters.to_csv(index=False),
        file_name=f"event_level_today_{datetime.now().strftime('%Y_%m_%d')}.csv"
    )

    # Show null report for each rolling/stat col
    st.markdown("#### Null report for rolling/stat features in output:")
    roll_diag = today_batters[full_stat_cols].isnull().sum().sort_values(ascending=False)
    st.text(roll_diag.to_string())

    # Show weather/context diagnostics for output
    st.markdown("#### Weather/context columns in output:")
    context_cols = ['city', 'park', 'stadium', 'time', 'temp', 'wind_mph', 'wind_dir', 'humidity', 'condition']
    for col in context_cols:
        if col not in today_batters.columns:
            today_batters[col] = None
    st.dataframe(today_batters[context_cols].drop_duplicates())
    st.markdown("---")
    st.subheader("🔄 Merge Rolling/Stat Features Into Today's Batter CSV")

    uploaded_today_lineups_v2 = st.file_uploader(
        "Upload Today's Lineups/Matchups CSV (again, for full feature merge)", type="csv", key="todaylineup_v2"
    )
    uploaded_sample_hist = st.file_uploader(
        "Upload Sample Historical Event-Level CSV (for diagnostics/column order)", type="csv", key="samplehist"
    )

    merge_btn = st.button("🔗 Generate Today's Batter Event-Level CSV (with merged rolling/stat features)")

    if merge_btn:
        if not uploaded_today_lineups_v2 or not uploaded_sample_hist:
            st.warning("Please upload BOTH today’s lineup/matchups and a sample historical event-level CSV.")
            st.stop()
        # Load data
        today_lineups = pd.read_csv(uploaded_today_lineups_v2)
        hist = pd.read_csv(uploaded_sample_hist)
        st.info(f"Loaded {len(today_lineups)} batters for today, {len(hist)} historical batted ball events.")

        # Normalize col names
        tcols = [c.strip().lower().replace(" ", "_") for c in today_lineups.columns]
        today_lineups.columns = tcols

        # Key for join is MLB ID (batter_id, mlb_id, etc)
        id_col = None
        if "mlb_id" in today_lineups.columns:
            id_col = "mlb_id"
        elif "batter_id" in today_lineups.columns:
            id_col = "batter_id"
        else:
            st.error("Could not find mlb_id/batter_id in lineups file.")
            st.stop()

        # Key for hist = 'batter_id'
        if 'batter_id' not in hist.columns:
            st.error("No 'batter_id' column in historical CSV.")
            st.stop()
        # Normalize for join
        today_lineups[id_col] = today_lineups[id_col].astype(str)
        hist['batter_id'] = hist['batter_id'].astype(str)

        # --- Grab latest features for each batter
        if 'game_date' in hist.columns:
            hist['game_date'] = pd.to_datetime(hist['game_date'], errors='coerce')
        rolling_cols = [c for c in hist.columns if (
            c.startswith('B_') or c.startswith('P_') or 
            c.startswith('park_hand_HR_') or 
            c.endswith('_rate_20') or 
            c.startswith('B_vsP_hand_HR_') or 
            c.startswith('P_vsB_hand_HR_') or 
            c.startswith('B_pitchtype_HR_') or 
            c.startswith('P_pitchtype_HR_')
        )]

        last_feats = hist.sort_values('game_date').groupby('batter_id').tail(1)
        last_feats = last_feats[['batter_id'] + rolling_cols].copy()
        st.info(f"Found {len(last_feats)} batters with latest rolling/stat features.")

        merged = today_lineups.merge(last_feats, left_on=id_col, right_on='batter_id', how='left', suffixes=('', '_roll'))

        st.write("Sample of Today's One-Row-Per-Batter DataFrame (with merged rolling/stat features):")
        st.dataframe(merged.head(25))
        null_report = merged[rolling_cols].isnull().sum().sort_values(ascending=False)
        st.markdown("#### Null report for rolling/stat features in output:")
        st.text(null_report.to_string())

        context_cols = ['city', 'park', 'stadium', 'time', 'temp', 'wind_mph', 'wind_dir', 'humidity', 'condition']
        for col in context_cols:
            if col not in merged.columns:
                merged[col] = None
        st.markdown("#### Weather/context columns in output after merge:")
        st.dataframe(merged[context_cols].drop_duplicates())

        # Ensure historical column order if sample file was uploaded
        ref_file = sample_ev_file if sample_ev_file else uploaded_sample_hist
        if ref_file:
            ev_sample = pd.read_csv(ref_file, nrows=1)
            hist_cols = [c for c in ev_sample.columns if not c.startswith("unnamed")]
            extra_cols = [c for c in merged.columns if c not in hist_cols]
            merged = merged.reindex(columns=hist_cols + extra_cols)

        merged_out_cols = [c for c in merged.columns if not c.startswith("unnamed")]
        st.success(f"Created today's event-level file: {len(merged)} batters. All rolling/stat columns now present (where available).")
        st.download_button(
            "⬇️ Download Today's Event-Level CSV (for Prediction App, with merged features)",
            data=merged[merged_out_cols].to_csv(index=False),
            file_name=f"event_level_today_full_{datetime.now().strftime('%Y_%m_%d')}.csv"
        )

        merged['rolling_stat_count'] = merged[rolling_cols].notnull().sum(axis=1)
        st.markdown("#### Non-null rolling/stat feature count per batter:")
        p_name = 'player_name' if 'player_name' in merged.columns else merged.columns[0]
        st.dataframe(merged[[p_name, 'rolling_stat_count']].sort_values('rolling_stat_count', ascending=False).head(25))

        st.markdown("---")
        st.info("Tab 1 complete. Proceed to Tab 2 for uploading event-level files and running the HR prediction/analyzer pipeline.")
        
with tab2:
    st.subheader("2️⃣ Upload & Analyze")
    st.markdown(
        """
        Upload event-level, matchup, and logistic weights CSVs.  
        Use the threshold slider to adjust HR probability cutoff for predictions.  
        Audit report downloads available below.
        """
    )

    uploaded_events = st.file_uploader("Upload Event-Level Features CSV", type="csv", key="evup")
    uploaded_matchups = st.file_uploader("Upload Matchups CSV", type="csv", key="mup")
    uploaded_logit = st.file_uploader("Upload Logistic Weights CSV", type="csv", key="lup")

    threshold = st.slider(
        "Set HR Probability Threshold:",
        min_value=0.01,
        max_value=0.50,
        step=0.01,
        value=0.13,
        help="Only events with HR probability above this are counted as HR predictions."
    )

    analyze_btn = st.button("Run Analysis (Logit + XGBoost Leaderboard)", type="primary")

    if analyze_btn:
        progress = st.progress(0, "10%: Loading data...")

        if not uploaded_events or not uploaded_matchups or not uploaded_logit:
            st.warning("Please upload all 3 files to continue.")
            st.stop()

        event_df = pd.read_csv(uploaded_events)
        matchup_df = pd.read_csv(uploaded_matchups)
        logit_weights = pd.read_csv(uploaded_logit)
        progress.progress(10, "20%: Cleaning and merging data...")

        event_df['batter_id'] = event_df.get('batter_id', event_df.get('batter', None))
        event_df['batter_id'] = event_df['batter_id'].apply(clean_id)
        matchup_df['mlb id'] = matchup_df['mlb id'].apply(clean_id)

        merged = event_df.merge(
            matchup_df[['mlb id', 'player name', 'batting order', 'position']],
            left_on='batter_id', right_on='mlb id', how='left', indicator=True
        )
        st.write("Merge results:", merged['_merge'].value_counts())
        st.write(f"Total merged rows: {len(merged)}")
        st.write("Rows missing matchup:", (merged['_merge'] == 'left_only').sum())

        progress.progress(20, "30%: Filtering for hitters (batting order 1-9, not pitchers)...")

        bo_raw = merged['batting order'].astype(str).str.strip().str.upper()
        pos = merged['position'].astype(str).str.strip().str.upper()
        bo_int = pd.to_numeric(bo_raw, errors='coerce')

        st.write("Unique batting order values:", sorted(bo_raw.unique()))
        st.write("Unique position values:", sorted(pos.unique()))

        hitter_mask = (
            bo_int.between(1, 9) &
            (~pos.fillna("").isin(['SP', 'P', 'RP', 'LHP', 'RHP']))
        )

        n_pass = hitter_mask.sum()
        st.write(f"Rows passing hitter filter: {n_pass} of {len(merged)} ({n_pass/len(merged)*100:.1f}%)")
        if n_pass == 0:
            st.error("All merged rows missing batting order/position! Check your IDs for formatting issues and upload new files.")
            st.stop()
        hitters_df = merged[hitter_mask].copy()
        hitters_df = dedup_columns(hitters_df)
        progress.progress(40, "40%: Filtered for hitters.")

        hitters_df['batter_name'] = (
            hitters_df['player name']
            .fillna(hitters_df.get('player_name'))
            .fillna(hitters_df['batter_id'])
        )

        if 'hr_outcome' not in hitters_df.columns:
            if 'events' in hitters_df.columns:
                hitters_df['hr_outcome'] = hitters_df['events'].astype(str).str.lower().str.replace(' ', '').isin(['homerun', 'home_run']).astype(int)
            else:
                st.error("No HR outcome detected, and 'events' column not available for mapping!")
                st.stop()

        hitters_df = hitters_df.loc[:, ~hitters_df.columns.duplicated()]
        hitters_df.columns = [str(col).strip().lower() for col in hitters_df.columns]

        if 'feature' in logit_weights.columns:
            model_features = [str(f).strip().lower() for f in logit_weights['feature'].values if str(f).strip().lower() in hitters_df.columns and pd.api.types.is_numeric_dtype(hitters_df[str(f).strip().lower()])]
        else:
            model_features = robust_numeric_columns(hitters_df)
        if not model_features or 'hr_outcome' not in hitters_df.columns:
            st.error("Model features or hr_outcome missing from event-level data.")
            st.stop()

        X = hitters_df[model_features].fillna(0)
        X = dedup_columns(X)
        y = hitters_df['hr_outcome'].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = dedup_columns(X_train)
        X_test = dedup_columns(X_test)
        progress.progress(50, "50%: Logistic Regression feature selection...")

        try:
            lr = LogisticRegression(max_iter=120, solver='liblinear')
            rfecv = RFECV(
                estimator=lr,
                step=3,
                cv=2,
                min_features_to_select=max(5, int(0.1*X_train.shape[1])),
                scoring='roc_auc',
                n_jobs=-1
            )
            rfecv.fit(X_train, y_train)
            selected_feature_names = [str(f).strip().lower() for f in X_train.columns[rfecv.support_]]
        except Exception as e:
            st.error(f"RFECV feature selection failed: {e}")
            st.stop()

        progress.progress(60, "60%: Logistic Regression grid search...")
        try:
            grid = GridSearchCV(LogisticRegression(max_iter=120, solver='liblinear'), param_grid={'C': [0.1, 1, 10]}, cv=2, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train[selected_feature_names], y_train)
            best_logit = grid.best_estimator_
        except Exception as e:
            st.error(f"Logistic regression fitting failed: {e}")
            st.stop()

        # Logistic scoring (with deduplication)
        if hasattr(best_logit, "feature_names_in_"):
            selected_feature_names = [str(f).strip().lower() for f in best_logit.feature_names_in_]
        X_hitters = hitters_df.reindex(columns=selected_feature_names)
        X_hitters = dedup_columns(X_hitters)
        X_hitters = X_hitters.fillna(0).astype(float)
        actual = list(X_hitters.columns)
        expected = list(selected_feature_names)
        if expected != actual:
            st.error(f"""
            🚨 Feature names mismatch!
            \nExpected (model was trained with):\n{expected}
            \nActual (provided to predict_proba):\n{actual}
            """)
            st.stop()

        hitters_df['logit_prob'] = best_logit.predict_proba(X_hitters)[:, 1]
        hitters_df['logit_hr_pred'] = (hitters_df['logit_prob'] > threshold).astype(int)

        # ========== XGBoost Block ==========
        progress.progress(70, "70%: XGBoost feature prep...")
        X_xgb = X.copy()
        X_xgb = dedup_columns(X_xgb)
        X_xgb = X_xgb.select_dtypes(include=[np.number]).copy()
        X_xgb = X_xgb.dropna(axis=1, how='all')
        for col in list(X_xgb.columns):
            if not (np.issubdtype(X_xgb[col].dtype, np.floating) or np.issubdtype(X_xgb[col].dtype, np.integer)):
                st.warning(f"Column '{col}' in XGBoost input is not purely numeric (dtype={X_xgb[col].dtype}). Dropping this column.")
                X_xgb = X_xgb.drop(columns=[col])
            elif X_xgb[col].apply(lambda x: isinstance(x, pd.DataFrame)).any():
                st.warning(f"Column '{col}' in XGBoost input contains nested DataFrames. Dropping this column.")
                X_xgb = X_xgb.drop(columns=[col])
        X_xgb = X_xgb.fillna(0).astype(float)
        y_xgb = y

        X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_xgb, y_xgb, test_size=0.2, random_state=42)
        progress.progress(80, "80%: Fitting XGBoost...")

        xgb_params = {
            'max_depth': [3, 4],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
        best_xgb = None
        try:
            xgb_grid = GridSearchCV(
                xgb.XGBClassifier(n_estimators=50, eval_metric='logloss', n_jobs=-1, use_label_encoder=False),
                xgb_params, cv=2, scoring='roc_auc', n_jobs=-1
            )
            xgb_grid.fit(X_train_xgb, y_train_xgb)
            best_xgb = xgb_grid.best_estimator_
            hitters_df['xgb_prob'] = best_xgb.predict_proba(X_xgb)[:, 1]
            hitters_df['xgb_hr_pred'] = (hitters_df['xgb_prob'] > threshold).astype(int)
        except Exception as e:
            st.warning(f"XGBoost grid fit failed: {e}")
            hitters_df['xgb_prob'] = np.nan
            hitters_df['xgb_hr_pred'] = np.nan

        progress.progress(100, "100%: Done! See results below.")

        # --- Leaderboards ---
        st.markdown("## Side-by-Side HR Probability Leaderboards (Top 15 Hitters)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Logit Leaderboard (Auto-tuned)")
            logit_leaderboard = (
                hitters_df.groupby('batter_name')
                .agg(
                    n_events=('hr_outcome', 'count'),
                    n_predicted_HR=('logit_hr_pred', 'sum'),
                    mean_logit_prob=('logit_prob', 'mean')
                )
                .sort_values(['n_predicted_HR', 'mean_logit_prob'], ascending=False)
                .head(15)
            )
            st.dataframe(logit_leaderboard)
        with col2:
            st.markdown("#### XGBoost Leaderboard (Auto-tuned)")
            xgb_leaderboard = (
                hitters_df.groupby('batter_name')
                .agg(
                    n_events=('hr_outcome', 'count'),
                    n_predicted_HR=('xgb_hr_pred', 'sum'),
                    mean_xgb_prob=('xgb_prob', 'mean')
                )
                .sort_values(['n_predicted_HR', 'mean_xgb_prob'], ascending=False)
                .head(15)
            )
            st.dataframe(xgb_leaderboard)

        st.markdown("#### Download Full Event-Level Data with Model Scores:")
        st.download_button("⬇️ Download Scored Event CSV", data=hitters_df.to_csv(index=False), file_name="event_level_scored.csv")

        # ========== BLIND DATES TOGGLE - INSERTION POINT ==========
        st.markdown("#### Blind HR outcomes for selected dates to generate future-unbiased predictions?")
        if 'game_date' in hitters_df.columns:
            all_dates = sorted(hitters_df['game_date'].astype(str).unique())
            default_blind = [d for d in all_dates if d >= '2025-06-17']
            blind_dates = st.multiselect(
                "Omit `hr_outcome` for which dates? (for backtest/future prediction)",
                options=all_dates,
                default=default_blind,
                help="Select one or more dates to omit actual HR outcomes (for unbiased prediction)"
            )
            if blind_dates:
                df_blind = hitters_df.copy()
                df_blind['game_date'] = df_blind['game_date'].astype(str)
                df_blind.loc[df_blind['game_date'].isin(blind_dates), 'hr_outcome'] = np.nan
                st.download_button(
                    "⬇️ Download Blinded Event-Level CSV",
                    data=df_blind.to_csv(index=False),
                    file_name="event_level_hr_features_blind.csv"
                )
