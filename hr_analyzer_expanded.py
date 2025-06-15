import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
import warnings

# ========= Context Maps ==========
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

# ========== Utility Functions ==========
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

def add_hr_outcome(df):
    if 'hr_outcome' not in df.columns:
        if 'events' in df.columns:
            df['hr_outcome'] = df['events'].astype(str).str.lower().str.contains('home_run|homerun|home run').astype(int)
        elif 'description' in df.columns:
            df['hr_outcome'] = df['description'].astype(str).str.lower().str.contains('home_run|homerun|home run').astype(int)
        else:
            df['hr_outcome'] = 0
    return df

@st.cache_data(show_spinner=False)
def get_weather(city, date):
    api_key = st.secrets.get("weather", {}).get("api_key", None)
    if not api_key:
        return {'temp': None, 'wind_mph': None, 'wind_dir': None, 'humidity': None, 'condition': None}
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

def robust_numeric_columns(df):
    numeric_types = (np.number, np.float32, np.float64, np.int32, np.int64, float, int)
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return cols

def rolling_stat_pct(grp, num_mask, denom_mask, w):
    # grp: pd.Series, num_mask/denom_mask: Boolean Series
    return num_mask.shift(1).rolling(w, min_periods=1).sum() / denom_mask.shift(1).rolling(w, min_periods=1).sum()

# ========== Streamlit UI ==========
st.set_page_config(layout="wide")
st.title("⚾ All-in-One MLB HR Analyzer & XGBoost Modeler")
st.markdown("""
Uploads, fetches and builds full Statcast + Weather + Park + Pitch Mix + Physics + Categorical + Advanced rolling features,  
**and allows training and using Logistic/XGBoost for HR event prediction with leaderboard and feature weights!**
""")

tab1, tab2 = st.tabs(["1️⃣ Fetch & Feature Engineer", "2️⃣ Analyze & Model"])

# ========== FETCH DATA TAB ==========
with tab1:
    st.header("Step 1: Data Fetch & Feature Engineering")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", value=datetime.today())

    st.subheader("Fetch Statcast Batted Ball Data")
    run_query = st.button("Fetch Statcast Data and Feature Engineer")

    uploaded_event_csv = st.file_uploader("Or Upload Event-Level CSV", type=["csv"], key="eventcsv1")
    uploaded_matchups = st.file_uploader("Upload Daily Lineups / Matchups CSV (optional)", type=["csv"], key="matchupcsv1")

    if run_query or (uploaded_event_csv is not None):
        if run_query:
            from pybaseball import statcast
            df = statcast(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            st.success(f"Loaded {len(df)} events from Statcast.")
        else:
            df = pd.read_csv(uploaded_event_csv)
            st.success(f"Loaded {len(df)} events from uploaded CSV.")

        # Minimal cleaning
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
            df['park'] = df['home_team_code'].map(team_code_to_park).fillna(
                df['home_team'].str.lower().str.replace(' ', '_') if 'home_team' in df.columns else None
            )
        df['park_hr_rate'] = df['park'].map(park_hr_rate_map).fillna(1.0)
        df['park_altitude'] = df['park'].map(park_altitude_map).fillna(0)
        df['roof_status'] = df['park'].map(roof_status_map).fillna("open")

        # Weather API Integration (event-level, de-dupe by home_team_code+date)
        weather_features = ['temp', 'wind_mph', 'wind_dir', 'humidity', 'condition']
        if 'home_team_code' in df.columns and 'game_date' in df.columns:
            df['weather_key'] = df['home_team_code'] + "_" + df['game_date'].dt.strftime("%Y%m%d")
            unique_keys = df['weather_key'].unique()
            weather_dict = {}
            for key in unique_keys:
                team = key.split('_')[0]
                city = mlb_team_city_map.get(team, "New York")
                date = df[df['weather_key'] == key].iloc[0]['game_date'].strftime("%Y-%m-%d")
                weather_dict[key] = get_weather(city, date)
            for feat in weather_features:
                df[feat] = df['weather_key'].map(lambda k: weather_dict[k][feat] if k in weather_dict else None)
            st.success("Weather merged")

        # Wind Direction Encoding
        df['wind_dir_angle'] = df['wind_dir'].apply(wind_dir_to_angle)
        df['wind_dir_sin'] = np.sin(np.deg2rad(df['wind_dir_angle']))
        df['wind_dir_cos'] = np.cos(np.deg2rad(df['wind_dir_angle']))

        # Home/Away feature
        if all(x in df.columns for x in ['home_team', 'batter']):
            df['is_home'] = (df['batter'].astype(str) == df['home_team'].astype(str)).astype(int)
        else:
            df['is_home'] = 0

        # ========== HR OUTCOME ENCODING ==========
        df = add_hr_outcome(df)

        # ========== PARK-HANDED HR RATE ==========
        df = compute_park_handed_hr_rate(df)

        # ========== ROLLING & ADVANCED FEATURES ==========
        ROLL = [3, 5, 7, 14]
        df = df.sort_values(['batter_id', 'game_date']) if 'batter_id' in df.columns and 'game_date' in df.columns else df
        batter_stats = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value']
        pitcher_stats = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value']

        # Use lists to avoid fragmentation warnings
        batter_features = {}
        pitcher_features = {}

        # Rolling for batter
        for stat in batter_stats:
            if stat in df.columns:
                for w in ROLL:
                    batter_features[f'B_{stat}_{w}'] = df.groupby('batter_id')[stat].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        for w in ROLL:
            if 'launch_speed' in df.columns:
                batter_features[f'B_max_ev_{w}'] = df.groupby('batter_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
                batter_features[f'B_median_ev_{w}'] = df.groupby('batter_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).median())

        # Rolling for pitcher
        for stat in pitcher_stats:
            if stat in df.columns:
                for w in ROLL:
                    pitcher_features[f'P_{stat}_{w}'] = df.groupby('pitcher_id')[stat].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        for w in ROLL:
            if 'launch_speed' in df.columns:
                pitcher_features[f'P_max_ev_{w}'] = df.groupby('pitcher_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
                pitcher_features[f'P_median_ev_{w}'] = df.groupby('pitcher_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).median())
            if 'hr_outcome' in df.columns:
                pitcher_features[f'P_rolling_hr_rate_{w}'] = df.groupby('pitcher_id')['hr_outcome'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())

        # Advanced Statcast Physics Rolling
        physics_cols = [
            'release_spin_rate', 'spin_axis', 'spin_dir', 'spin_rate_deprecated',
            'attack_angle', 'attack_direction', 'bat_speed'
        ]
        for col in physics_cols:
            if col in df.columns:
                for w in ROLL:
                    pitcher_features[f'P_{col}_{w}'] = df.groupby('pitcher_id')[col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                    batter_features[f'B_{col}_{w}'] = df.groupby('batter_id')[col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())

        # Combine all new features at once (avoid fragmentation)
        df = pd.concat([df, pd.DataFrame(batter_features, index=df.index), pd.DataFrame(pitcher_features, index=df.index)], axis=1)

        # Pitch type mix for batters/pitchers (percent of each type in last X PA or BF)
        pitch_types = ['SL', 'SI', 'FC', 'FF', 'ST', 'CH', 'CU', 'FS', 'FO', 'SV', 'KC', 'EP', 'FA', 'KN', 'CS', 'SC']
        for pt in pitch_types:
            if 'pitch_type' in df.columns:
                for w in ROLL:
                    colname_b = f'B_pitch_pct_{pt}_{w}'
                    colname_p = f'P_pitch_pct_{pt}_{w}'
                    # Avoid fragmentation: collect as dict, then join
                    batter_features[colname_b] = df.groupby('batter_id')['pitch_type'].transform(lambda x: x.shift(1).eq(pt).rolling(w, min_periods=1).mean())
                    pitcher_features[colname_p] = df.groupby('pitcher_id')['pitch_type'].transform(lambda x: x.shift(1).eq(pt).rolling(w, min_periods=1).mean())
        df = pd.concat([df, pd.DataFrame(batter_features, index=df.index), pd.DataFrame(pitcher_features, index=df.index)], axis=1)

        # Categorical dummies
        cat_cols = [
            "stand", "p_throws", "pitch_type", "pitch_name", "bb_type",
            "condition", "roof_status", "handed_matchup"
        ]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('Unknown')
        df = pd.get_dummies(df, columns=[col for col in cat_cols if col in df.columns], drop_first=False)

        # Batted Ball Flags (fillna, ensure int)
        batted_ball_flags = [
            'is_barrel', 'is_sweet_spot', 'is_hard_hit', 'flyball', 'line_drive',
            'groundball', 'pull_air', 'pull_side'
        ]
        for col in batted_ball_flags:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        # More weather interactions
        for col in ['is_barrel', 'is_hard_hit', 'flyball', 'pull_air']:
            if col in df.columns and all(x in df.columns for x in ['humidity', 'temp', 'wind_mph']):
                df[f'{col}_x_humidity'] = df[col] * df['humidity']
                df[f'{col}_x_temp'] = df[col] * df['temp']
                df[f'{col}_x_wind_mph'] = df[col] * df['wind_mph']

        # Situational Features
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

        # ========== OUTPUTS ==========
        export_cols = [
            'game_date', 'batter', 'batter_id', 'pitcher', 'pitcher_id', 'events', 'description',
            'stand', 'p_throws', 'park', 'park_hr_rate', 'park_handed_hr_rate', 'park_altitude', 'roof_status',
            'temp', 'wind_mph', 'wind_dir', 'humidity', 'condition',
            'handed_matchup', 'is_home', 'is_high_leverage', 'is_early_inning', 'is_late_inning', 'runners_on', 'hr_outcome'
        ]
        export_cols += [c for c in df.columns if c not in export_cols]
        event_cols = [c for c in export_cols if c in df.columns]
        event_df = df[event_cols].copy()
        st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
        st.dataframe(event_df.head(20))
        st.download_button("⬇️ Download Event-Level CSV", data=event_df.to_csv(index=False), file_name="event_level_hr_features.csv")

        # ========== Logistic Regression Weights (if possible) ==========
        numeric_cols = robust_numeric_columns(df)
        model_features = [c for c in numeric_cols if c not in ['hr_outcome']]
        model_df = df.dropna(subset=model_features + ['hr_outcome'], how='any')
        logit_weights = pd.DataFrame()
        auc = None
        if 'hr_outcome' in df.columns and model_df['hr_outcome'].nunique() > 1:
            X = model_df[model_features]
            y = model_df['hr_outcome'].astype(int)
            logit = LogisticRegression(max_iter=200, solver='liblinear')
            logit.fit(X, y)
            weights = pd.DataFrame({'feature': model_features, 'weight': logit.coef_[0]})
            st.markdown("#### Download Logistic Regression Weights")
            st.dataframe(weights.sort_values('weight', ascending=False).head(50))
            st.download_button(
                "⬇️ Download Logistic Weights CSV",
                data=weights.to_csv(index=False),
                file_name="logistic_weights.csv"
            )
            # Compute AUC for info
            y_pred_prob = logit.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, y_pred_prob)
            st.info(f"Logistic Regression ROC-AUC: {auc:.3f}")
        elif 'hr_outcome' in df.columns and model_df['hr_outcome'].nunique() == 1:
            st.warning("Only one class found in hr_outcome. Logistic Regression not run.")
        else:
            st.warning("No 'hr_outcome' column detected! Add or compute this before downstream modeling.")

# ========== ANALYSIS TAB ==========
with tab2:
    st.header("Step 2: Upload CSVs & Run Models")
    st.markdown("Upload your event-level CSV (with hr_outcome), your daily matchups CSV, and (optionally) your logistic weights CSV to run analysis and leaderboard.")

    col1, col2, col3 = st.columns(3)
    uploaded_event = col1.file_uploader("Upload Event-Level CSV", type=["csv"], key="eventcsv2")
    uploaded_matchup = col2.file_uploader("Upload Matchups CSV", type=["csv"], key="matchupcsv2")
    uploaded_logit = col3.file_uploader("Upload Logistic Weights CSV", type=["csv"], key="logitcsv2")

    ready = uploaded_event is not None and uploaded_matchup is not None
    if ready:
        df = pd.read_csv(uploaded_event)
        matchup_df = pd.read_csv(uploaded_matchup)
        if uploaded_logit:
            logit_weights = pd.read_csv(uploaded_logit)
        else:
            logit_weights = None

        df = add_hr_outcome(df)
        # Features already engineered in previous step

        # ========== MODELING ==========
        numeric_cols = robust_numeric_columns(df)
        model_features = [c for c in numeric_cols if c not in ['hr_outcome']]
        model_df = df.dropna(subset=model_features + ['hr_outcome'], how='any')
        if 'hr_outcome' in df.columns and model_df['hr_outcome'].nunique() > 1:
            X = model_df[model_features]
            y = model_df['hr_outcome'].astype(int)
            # Logistic Regression (if user uploads weights, skip fit)
            if logit_weights is None:
                logit = LogisticRegression(max_iter=200, solver='liblinear')
                logit.fit(X, y)
                weights = pd.DataFrame({'feature': model_features, 'weight': logit.coef_[0]})
            else:
                weights = logit_weights
            st.markdown("#### Logistic Regression Feature Weights")
            st.dataframe(weights.sort_values('weight', ascending=False).head(50))
            # XGBoost
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
            st.markdown("#### XGBoost Feature Importances")
            feat_import = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.dataframe(feat_import.head(50))
            st.download_button(
                "⬇️ Download XGBoost Model (.pkl)",
                data=pickle.dumps(xgb_model),
                file_name="xgboost_hr_model.pkl"
            )
        else:
            st.warning("No 'hr_outcome' column detected or only one class present. Cannot run models.")

        # ========== OUTPUTS ==========
        export_cols = [
            'game_date', 'batter', 'batter_id', 'pitcher', 'pitcher_id', 'events', 'description',
            'stand', 'p_throws', 'park', 'park_hr_rate', 'park_handed_hr_rate', 'park_altitude', 'roof_status',
            'temp', 'wind_mph', 'wind_dir', 'humidity', 'condition',
            'handed_matchup', 'is_home', 'is_high_leverage', 'is_early_inning', 'is_late_inning', 'runners_on', 'hr_outcome'
        ]
        export_cols += [c for c in df.columns if c not in export_cols]
        event_cols = [c for c in export_cols if c in df.columns]
        event_df = df[event_cols].copy()
        st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
        st.dataframe(event_df.head(20))
        st.download_button("⬇️ Download Event-Level CSV", data=event_df.to_csv(index=False), file_name="event_level_hr_features.csv")

st.info("🚀 HR Analyzer — All features, physics, rolling splits, robust modeling, UI, and CSV management included.")
