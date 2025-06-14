import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybaseball import statcast
import requests
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import pickle

# ===================== Context Maps =====================
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

# ===================== Utility Functions =====================
def wind_dir_to_angle(wind_dir):
    directions = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5,
        'SE': 135, 'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    try:
        for d in directions:
            if d in str(wind_dir).upper():
                return directions[d]
    except:
        pass
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

def compute_park_handed_hr_rate(df):
    if all(col in df.columns for col in ['stand', 'p_throws', 'hr_outcome', 'park']):
        df['handed_matchup'] = df['stand'].astype(str) + df['p_throws'].astype(str)
        grp = df.groupby(['park', 'handed_matchup'])
        rate = grp['hr_outcome'].mean().reset_index().rename(
            columns={'hr_outcome': 'park_handed_hr_rate'}
        )
        df = df.merge(rate, on=['park', 'handed_matchup'], how='left')
    else:
        df['park_handed_hr_rate'] = np.nan
    return df

def remove_duplicate_columns(df):
    _, idx = np.unique(df.columns, return_index=True)
    return df.iloc[:, idx]

# ===================== Streamlit UI =====================
st.title("⚾ All-in-One MLB HR Analyzer & XGBoost Modeler")
st.markdown("""
Uploads, fetches and builds full Statcast + Weather + Park + Pitch Mix + Categorical + Advanced interaction features,
**and allows training and using XGBoost for HR event prediction with leaderboard!**
""")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=7))
with col2:
    end_date = st.date_input("End Date", value=datetime.today())

st.subheader("Step 1: Data Source")
load_source = st.radio("Load Data From", ['Fetch Statcast', 'Upload CSV'], horizontal=True)

if load_source == "Fetch Statcast":
    run_query = st.button("Fetch Statcast Data and Run Analyzer")
else:
    uploaded_event_csv = st.file_uploader("Upload Event-Level CSV", type=["csv"])

uploaded_matchups = st.file_uploader("Upload Daily Lineups / Matchups CSV (optional)", type=["csv"])

# ===================== Data Load/Prep =====================
df = None
if (load_source == "Fetch Statcast" and 'run_query' in locals() and run_query) or (load_source == "Upload CSV" and 'uploaded_event_csv' in locals() and uploaded_event_csv is not None):
    progress = st.progress(0, text="Starting...")
    if load_source == "Fetch Statcast":
        df = statcast(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        progress.progress(5, text="Statcast downloaded")
    else:
        df = pd.read_csv(uploaded_event_csv)
        progress.progress(5, text="CSV loaded")
    st.success(f"Loaded {len(df)} events.")

    # Minimal cleaning for robust merges
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])
    if 'batter_id' not in df.columns and 'batter' in df.columns:
        df['batter_id'] = df['batter']
    if 'pitcher_id' not in df.columns and 'pitcher' in df.columns:
        df['pitcher_id'] = df['pitcher']
    if 'home_team_code' not in df.columns and 'home_team' in df.columns:
        df['home_team_code'] = df['home_team']
    if 'home_team_code' in df.columns:
        df['home_team_code'] = df['home_team_code'].astype(str).str.upper()
    if 'park' not in df.columns and 'home_team_code' in df.columns:
        df['park'] = df['home_team_code'].map(team_code_to_park).fillna(df['home_team'].str.lower().str.replace(' ', '_'))
    df['park_hr_rate'] = df['park'].map(park_hr_rate_map).fillna(1.0)
    df['park_altitude'] = df['park'].map(park_altitude_map).fillna(0)
    df['roof_status'] = df['park'].map(roof_status_map).fillna("open")

    # ===================== Weather Integration =====================
    weather_features = ['temp', 'wind_mph', 'wind_dir', 'humidity', 'condition']
    if 'home_team_code' in df.columns and 'game_date' in df.columns:
        df['weather_key'] = df['home_team_code'] + "_" + df['game_date'].dt.strftime("%Y%m%d")
        unique_keys = df['weather_key'].unique()
        for i, key in enumerate(unique_keys):
            team = key.split('_')[0]
            city = mlb_team_city_map.get(team, "New York")
            date = df[df['weather_key'] == key].iloc[0]['game_date'].strftime("%Y-%m-%d")
            weather = get_weather(city, date)
            for feat in weather_features:
                df.loc[df['weather_key'] == key, feat] = weather[feat]
            progress.progress(5 + int(35 * (i+1) / len(unique_keys)), text=f"Weather {i+1}/{len(unique_keys)}")
        progress.progress(40, text="Weather merged")
    else:
        for feat in weather_features:
            df[feat] = None

    # ===================== Feature Engineering (Optimized) =====================

    # --- Wind Direction Encoding ---
    df['wind_dir_angle'] = df['wind_dir'].apply(wind_dir_to_angle)
    df['wind_dir_sin'] = np.sin(np.deg2rad(df['wind_dir_angle']))
    df['wind_dir_cos'] = np.cos(np.deg2rad(df['wind_dir_angle']))

    # --- Categorical/Indicator Features ---
    df['is_home'] = ((df['home_team_code'] == df['batter'].map(lambda x: str(x)[:3]))
                     if 'batter' in df.columns and 'home_team_code' in df.columns else False).astype(int)
    df['is_high_leverage'] = ((df['inning'] >= 8) & (df['bat_score_diff'].abs() <= 2)).astype(int) if 'inning' in df.columns and 'bat_score_diff' in df.columns else 0
    df['is_late_inning'] = (df['inning'] >= 8).astype(int) if 'inning' in df.columns else 0
    df['is_early_inning'] = (df['inning'] <= 3).astype(int) if 'inning' in df.columns else 0
    df['runners_on'] = (df[['on_1b', 'on_2b', 'on_3b']].fillna(0).astype(int).sum(axis=1) > 0).astype(int) if set(['on_1b','on_2b','on_3b']).issubset(df.columns) else 0

    # --- Batted Ball Type Binary Flags ---
    if 'bb_type' in df.columns:
        df['bb_type_fly_ball'] = (df['bb_type']=='fly_ball').astype(int)
        df['bb_type_line_drive'] = (df['bb_type']=='line_drive').astype(int)
        df['bb_type_ground_ball'] = (df['bb_type']=='ground_ball').astype(int)
        df['bb_type_popup'] = (df['bb_type']=='popup').astype(int)

    # --- Interactions for Weather/Batted Ball ---
    for feat in ['is_hard_hit','flyball','pull_air','is_barrel']:
        if feat in df.columns:
            df[f'{feat}_x_temp'] = df[feat] * df['temp']
            df[f'{feat}_x_humidity'] = df[feat] * df['humidity']
            df[f'{feat}_x_wind_mph'] = df[feat] * df['wind_mph']

    # ===================== OPTIMIZED ROLLING & PITCH MIX =====================

    # All rolling features are created in batches and added at once for efficiency!
    ROLL = [3, 5, 7, 14]
    batter_stats = ['launch_speed','launch_angle','hit_distance_sc','woba_value']
    pitcher_stats = ['launch_speed','launch_angle','hit_distance_sc','woba_value']

    new_batter_cols = {}
    new_pitcher_cols = {}

    if 'batter_id' in df.columns and 'game_date' in df.columns:
        df = df.sort_values(['batter_id', 'game_date'])

        for stat in batter_stats:
            if stat in df.columns:
                for w in ROLL:
                    # Rolling mean
                    new_batter_cols[f'B_{stat}_{w}'] = df.groupby('batter_id')[stat].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        if 'launch_speed' in df.columns:
            for w in ROLL:
                new_batter_cols[f'B_max_ev_{w}'] = df.groupby('batter_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
                new_batter_cols[f'B_median_ev_{w}'] = df.groupby('batter_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).median())
        df = pd.concat([df, pd.DataFrame(new_batter_cols)], axis=1)
        df = df.copy()  # De-fragment

    if 'pitcher_id' in df.columns and 'game_date' in df.columns:
        df = df.sort_values(['pitcher_id', 'game_date'])

        for stat in pitcher_stats:
            if stat in df.columns:
                for w in ROLL:
                    new_pitcher_cols[f'P_{stat}_{w}'] = df.groupby('pitcher_id')[stat].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        if 'launch_speed' in df.columns:
            for w in ROLL:
                new_pitcher_cols[f'P_max_ev_{w}'] = df.groupby('pitcher_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
                new_pitcher_cols[f'P_median_ev_{w}'] = df.groupby('pitcher_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).median())
                if 'hr_outcome' in df.columns:
                    new_pitcher_cols[f'P_rolling_hr_rate_{w}'] = df.groupby('pitcher_id')['hr_outcome'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df = pd.concat([df, pd.DataFrame(new_pitcher_cols)], axis=1)
        df = df.copy()  # De-fragment

    # --- Pitch type mix: create all at once ---
    pitch_types = ['SL','SI','FC','FF','ST','CH','CU','FS','FO','SV','KC','EP','FA','KN','CS','SC']
    batch_cols = {}

    for pt in pitch_types:
        if 'pitch_type' in df.columns:
            for w in ROLL:
                # Batter
                batch_cols[f'B_pitch_pct_{pt}_{w}'] = df.groupby('batter_id')['pitch_type'].transform(lambda x: x.shift(1).eq(pt).rolling(w, min_periods=1).mean())
                # Pitcher
                batch_cols[f'P_pitch_pct_{pt}_{w}'] = df.groupby('pitcher_id')['pitch_type'].transform(lambda x: x.shift(1).eq(pt).rolling(w, min_periods=1).mean())

    if batch_cols:
        df = pd.concat([df, pd.DataFrame(batch_cols)], axis=1)
        df = df.copy()  # De-fragment

    # --- Park-handed HR rate ---
    df = compute_park_handed_hr_rate(df)

    # --- Categorical dummies (no col duplication) ---
    cat_cols = [
        "stand", "p_throws", "pitch_type", "pitch_name", "bb_type",
        "condition", "roof_status", "handed_matchup"
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('Unknown')
    df = pd.get_dummies(df, columns=[col for col in cat_cols if col in df.columns], drop_first=False)
    df = remove_duplicate_columns(df)

    # --- Batted Ball Flags ---
    batted_ball_flags = [
        'is_barrel', 'is_sweet_spot', 'is_hard_hit', 'flyball', 'line_drive',
        'groundball', 'pull_air', 'pull_side'
    ]
    for col in batted_ball_flags:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # --- Weather interactions ---
    for col in ['is_barrel', 'is_hard_hit', 'flyball', 'pull_air']:
        if col in df.columns and all(x in df.columns for x in ['humidity', 'temp', 'wind_mph']):
            df[f'{col}_x_humidity'] = df[col] * df['humidity']
            df[f'{col}_x_temp'] = df[col] * df['temp']
            df[f'{col}_x_wind_mph'] = df[col] * df['wind_mph']

    # --- Situational Features ---
    if 'inning' in df.columns:
        df['is_early_inning'] = (df['inning'] <= 3).astype(int)
        df['is_late_inning'] = (df['inning'] >= 7).astype(int)
    else:
        df['is_early_inning'] = 0
        df['is_late_inning'] = 0
    if 'outs_when_up' in df.columns:
        df['is_high_leverage'] = (df['outs_when_up'] == 2).astype(int)
    else:
        df['is_high_leverage'] = 0
    if set(['on_1b', 'on_2b', 'on_3b']).issubset(df.columns):
        df['runners_on'] = (
            df['on_1b'].notnull().astype(int) +
            df['on_2b'].notnull().astype(int) +
            df['on_3b'].notnull().astype(int)
        )
    else:
        df['runners_on'] = 0

    # --- Home/Away feature ---
    if all(x in df.columns for x in ['home_team', 'batter']):
        df['is_home'] = (df['batter'].astype(str) == df['home_team'].astype(str)).astype(int)
    else:
        df['is_home'] = 0

    # --- Wind trigonometric encoding ---
    if 'wind_dir' in df.columns:
        df['wind_dir_angle'] = df['wind_dir'].apply(wind_dir_to_angle)
        df['wind_dir_sin'] = np.sin(np.deg2rad(df['wind_dir_angle']))
        df['wind_dir_cos'] = np.cos(np.deg2rad(df['wind_dir_angle']))
    else:
        df['wind_dir_sin'] = 0
        df['wind_dir_cos'] = 1

    df = remove_duplicate_columns(df)

    # ===================== Modeling Features Selection =====================
    model_features = [
        # Numeric/statcast/context
        'woba_value', 'launch_speed', 'launch_angle', 'hit_distance_sc', 'launch_speed_angle'
    ]
    # Add all rolling, pitch mix, event dummies, interaction features, pitcher features
    for col in df.columns:
        if (
            col.startswith('B_') or col.startswith('P_') or col.startswith('park_') or
            col.startswith('temp') or col.startswith('humidity') or col.startswith('wind_') or
            col.startswith('is_') or col.startswith('pull_') or col.startswith('flyball') or
            col.startswith('bb_type_') or col.startswith('pitch_type_') or col.startswith('pitch_name_') or
            col.startswith('condition_') or col.startswith('roof_status_') or col.startswith('handed_matchup_') or
            col.startswith('stand_') or col.startswith('p_throws_')
        ):
            model_features.append(col)
    model_features = list(dict.fromkeys([c for c in model_features if c in df.columns]))

    # ===================== XGBoost Modeling =====================
    if 'hr_outcome' in df.columns:
        model_df = df.dropna(subset=model_features + ['hr_outcome'], how='any')
        if len(model_df) > 50:
            X = model_df[model_features]
            y = model_df['hr_outcome'].astype(int)
            X = X.apply(pd.to_numeric, errors='coerce')
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.15, random_state=42, stratify=y
            )
            xgb_model = xgb.XGBClassifier(
                n_estimators=120, max_depth=5, learning_rate=0.15,
                subsample=0.9, colsample_bytree=0.8, eval_metric='auc', use_label_encoder=False
            )
            xgb_model.fit(X_train, y_train)
            y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_prob)
            st.success(f"XGBoost Model ROC-AUC: {auc:.3f}")
            st.write(classification_report(y_test, y_pred_prob > 0.5, digits=3))
            feature_importance = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.markdown("#### XGBoost Feature Importances")
            st.dataframe(feature_importance.head(50))
            st.download_button(
                "⬇️ Download XGBoost Model (.pkl)",
                data=pickle.dumps(xgb_model),
                file_name="xgboost_hr_model.pkl"
            )
        else:
            st.warning("Not enough complete events to fit XGBoost model. Try a wider date range.")
    else:
        st.warning("No 'hr_outcome' column found in data.")

    # ===================== CSV OUTPUTS =====================
    export_cols = [
        'game_date', 'batter', 'batter_id', 'pitcher', 'pitcher_id', 'events', 'description',
        'stand', 'p_throws', 'park', 'park_hr_rate', 'park_handed_hr_rate', 'park_altitude', 'roof_status',
        'temp', 'wind_mph', 'wind_dir', 'humidity', 'condition',
        'handed_matchup', 'platoon', 'is_day', 'hr_outcome'
    ]
    export_cols += [c for c in df.columns if c not in export_cols]
    event_cols = [c for c in export_cols if c in df.columns]
    event_df = df[event_cols].copy()
    st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
    st.dataframe(event_df.head(20))
    st.download_button("⬇️ Download Event-Level CSV", data=event_df.to_csv(index=False), file_name="event_level_hr_features.csv")

    # Player-Level CSV
    player_cols = ['batter_id', 'batter'] + [c for c in df.columns if c.startswith('B_')]
    if any(c.startswith('P_') for c in df.columns):
        player_cols += [c for c in df.columns if c.startswith('P_')]
    player_df = (
        event_df.groupby(['batter_id', 'batter'])
        .tail(1)[player_cols]
        .reset_index(drop=True)
    )
    st.markdown("#### Download Player-Level Rolling Feature CSV (1 row per batter):")
    st.dataframe(player_df.head(20))
    st.download_button("⬇️ Download Player-Level CSV", data=player_df.to_csv(index=False), file_name="player_level_hr_features.csv")

    st.success("All-in-one MLB HR Analyzer is complete! 🚀")
