import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybaseball import statcast
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle

# ========= Context maps ==========
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

# ========= Utility functions ==========
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
    except Exception:
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

def optimize_rolling_features(df, id_col, stats, roll_windows, prefix):
    features = {}
    df = df.sort_values([id_col, 'game_date'])
    for stat in stats:
        for w in roll_windows:
            colname = f"{prefix}{stat}_{w}"
            if stat in df.columns:
                features[colname] = df.groupby(id_col)[stat].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        # Add max/median EV for launch_speed
        if stat == 'launch_speed':
            for w in roll_windows:
                features[f"{prefix}max_ev_{w}"] = df.groupby(id_col)[stat].transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
                features[f"{prefix}median_ev_{w}"] = df.groupby(id_col)[stat].transform(lambda x: x.shift(1).rolling(w, min_periods=1).median())
    return features

def optimize_pitch_type_pct(df, id_col, roll_windows, prefix):
    pitch_types = ['SL','SI','FC','FF','ST','CH','CU','FS','FO','SV','KC','EP','FA','KN','CS','SC']
    features = {}
    for pt in pitch_types:
        for w in roll_windows:
            features[f"{prefix}pitch_pct_{pt}_{w}"] = (
                df.groupby(id_col)['pitch_type'].transform(
                    lambda x: x.shift(1).eq(pt).rolling(w, min_periods=1).mean()
                ) if 'pitch_type' in df.columns else 0
            )
    return features

# ========= App UI ==========
st.title("⚾ All-in-One MLB HR Analyzer & XGBoost/Logistic Modeler")
st.markdown("""
Upload/fetch and build full Statcast + Weather + Park + Pitch Mix + Categorical + Interaction features,
**then train and use XGBoost and Logistic Regression for HR event prediction and leaderboard (starters only)!**
""")

app_mode = st.radio("Choose App Mode:", ["Fetch Data & Feature Engineer", "Upload & Analyze (Logit/XGB)"], horizontal=True)

# ====== FETCH DATA & FEATURE ENGINEERING MODE ======
if app_mode == "Fetch Data & Feature Engineer":
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", value=datetime.today())

    fetch_button = st.button("Fetch Statcast & Feature Engineer")
    df = None
    if fetch_button:
        progress = st.progress(0, text="Starting fetch...")
        df = statcast(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        st.success(f"Loaded {len(df)} events.")
        progress.progress(10, text="Minimal cleaning")

        # === Minimal cleaning and types ===
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
                df.get('home_team', pd.Series('')).str.lower().str.replace(' ', '_'))
        df['park_hr_rate'] = df['park'].map(park_hr_rate_map).fillna(1.0)
        df['park_altitude'] = df['park'].map(park_altitude_map).fillna(0)
        df['roof_status'] = df['park'].map(roof_status_map).fillna("open")
        progress.progress(12, text="Park/venue added")

        # === Weather merge ===
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
                progress.progress(12 + int(28 * (i+1) / len(unique_keys)), text=f"Weather {i+1}/{len(unique_keys)}")
            progress.progress(40, text="Weather merged")
        else:
            for feat in weather_features:
                df[feat] = None

        # === Wind Direction Encoding ===
        df['wind_dir_angle'] = df['wind_dir'].apply(wind_dir_to_angle)
        df['wind_dir_sin'] = np.sin(np.deg2rad(df['wind_dir_angle']))
        df['wind_dir_cos'] = np.cos(np.deg2rad(df['wind_dir_angle']))

        # === Categorical/Indicator Features ===
        df['is_home'] = (df['home_team_code'] == df['batter'].map(lambda x: str(x)[:3]) if 'batter' in df.columns and 'home_team_code' in df.columns else False).astype(int)
        df['is_high_leverage'] = ((df['inning'] >= 8) & (df['bat_score_diff'].abs() <= 2)).astype(int) if 'inning' in df.columns and 'bat_score_diff' in df.columns else 0
        df['is_late_inning'] = (df['inning'] >= 8).astype(int) if 'inning' in df.columns else 0
        df['is_early_inning'] = (df['inning'] <= 3).astype(int) if 'inning' in df.columns else 0
        df['runners_on'] = (df[['on_1b', 'on_2b', 'on_3b']].fillna(0).astype(int).sum(axis=1) > 0).astype(int) if set(['on_1b','on_2b','on_3b']).issubset(df.columns) else 0

        # === Batted Ball Type Binary Flags ===
        if 'bb_type' in df.columns:
            df['bb_type_fly_ball'] = (df['bb_type']=='fly_ball').astype(int)
            df['bb_type_line_drive'] = (df['bb_type']=='line_drive').astype(int)
            df['bb_type_ground_ball'] = (df['bb_type']=='ground_ball').astype(int)
            df['bb_type_popup'] = (df['bb_type']=='popup').astype(int)

        # === Interaction Features ===
        for feat in ['is_hard_hit','flyball','pull_air','is_barrel']:
            if feat in df.columns:
                df[f'{feat}_x_temp'] = df[feat] * df['temp']
                df[f'{feat}_x_humidity'] = df[feat] * df['humidity']
                df[f'{feat}_x_wind_mph'] = df[feat] * df['wind_mph']

        # === OPTIMIZED ROLLING FEATURES: batter and pitcher ===
        roll_windows = [3,5,7,14]
        batter_stats = ['launch_speed','launch_angle','hit_distance_sc','woba_value']
        pitcher_stats = ['launch_speed','launch_angle','hit_distance_sc','woba_value']

        batter_features = optimize_rolling_features(df, 'batter_id', batter_stats, roll_windows, 'B_')
        pitcher_features = optimize_rolling_features(df, 'pitcher_id', pitcher_stats, roll_windows, 'P_')

        # === OPTIMIZED pitch type pct features ===
        batter_pitch_features = optimize_pitch_type_pct(df, 'batter_id', roll_windows, 'B_')
        pitcher_pitch_features = optimize_pitch_type_pct(df, 'pitcher_id', roll_windows, 'P_')
        # Ensure unique index for concat
        df = df.reset_index(drop=True)
        # === Add all new columns at once to avoid fragmentation ===
        df = pd.concat([df, 
            pd.DataFrame(batter_features, index=df.index),
            pd.DataFrame(pitcher_features, index=df.index),
            pd.DataFrame(batter_pitch_features, index=df.index),
            pd.DataFrame(pitcher_pitch_features, index=df.index)
            ], axis=1)

        # === Park-Handed HR Rate ===
        df = compute_park_handed_hr_rate(df)

        # === Categorical dummies for modeling ===
        cat_cols = [
            "stand", "p_throws", "pitch_type", "pitch_name", "bb_type",
            "condition", "roof_status", "handed_matchup"
        ]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('Unknown')
        df = pd.get_dummies(df, columns=[col for col in cat_cols if col in df.columns], drop_first=False)

        # === Batted Ball Flags (fillna, ensure int) ===
        batted_ball_flags = [
            'is_barrel', 'is_sweet_spot', 'is_hard_hit', 'flyball', 'line_drive',
            'groundball', 'pull_air', 'pull_side'
        ]
        for col in batted_ball_flags:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        # === More weather interactions ===
        for col in ['is_barrel', 'is_hard_hit', 'flyball', 'pull_air']:
            if col in df.columns and all(x in df.columns for x in ['humidity', 'temp', 'wind_mph']):
                df[f'{col}_x_humidity'] = df[col] * df['humidity']
                df[f'{col}_x_temp'] = df[col] * df['temp']
                df[f'{col}_x_wind_mph'] = df[col] * df['wind_mph']

        # ========== DOWNLOADS ==========
        # Ensure 'hr_outcome' present for modeling
        if 'hr_outcome' not in df.columns:
            st.warning("No 'hr_outcome' column detected. Please ensure this is present before analysis.")
        else:
            st.success("Feature engineering complete! Download event-level CSV for analysis or scoring.")
            # Download event-level CSV
            st.download_button(
                "⬇️ Download Event-Level CSV",
                data=df.to_csv(index=False),
                file_name="event_level_hr_features.csv"
            )

            # === Optional: Fit logistic regression, save weights CSV ===
            model_features = [c for c in df.columns if c not in ['hr_outcome', 'game_date', 'batter', 'pitcher', 'events', 'description']]
            # Filter numeric only
            model_features = [c for c in model_features if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]
            model_df = df.dropna(subset=model_features + ['hr_outcome'], how='any')
            if len(model_df) > 50:
                X = model_df[model_features]
                y = model_df['hr_outcome'].astype(int)
                logit = LogisticRegression(max_iter=500, solver='liblinear')
                logit.fit(X, y)
                coefs = pd.DataFrame({'feature': X.columns, 'weight': logit.coef_[0]})
                st.download_button(
                    "⬇️ Download Logistic Weights CSV",
                    data=coefs.to_csv(index=False),
                    file_name="logit_weights.csv"
                )
                st.success("Logistic weights fitted and downloadable for analysis!")
            else:
                st.info("Not enough data for logistic regression weight fitting. Save event-level CSV and fit in analysis mode.")
        st.stop()

# ====== UPLOAD & ANALYZE MODE (LOGIT/XGBOOST/LEADERBOARD) ======
else:
    st.markdown("#### Upload Event-Level CSV, Daily Matchups, and Logistic Weights (all required):")
    col1, col2, col3 = st.columns(3)
    with col1:
        events_csv = st.file_uploader("Event-Level CSV", type=['csv'], key="event_csv")
    with col2:
        matchups_csv = st.file_uploader("Daily Matchups CSV", type=['csv'], key="matchups_csv")
    with col3:
        logit_weights_csv = st.file_uploader("Logistic Weights CSV", type=['csv'], key="logit_csv")
    analyze_btn = st.button("Analyze & Score")

    # All files are required!
    if analyze_btn:
        if not (events_csv and matchups_csv and logit_weights_csv):
            st.error("Please upload all 3 required CSV files to proceed.")
            st.stop()

        # --- Load data ---
        df = pd.read_csv(events_csv)
        matchups = pd.read_csv(matchups_csv)
        logit_weights = pd.read_csv(logit_weights_csv)

        # --- Check for hr_outcome ---
        if 'hr_outcome' not in df.columns:
            st.error("Event-level CSV is missing 'hr_outcome'. Please regenerate your features and ensure this column is present.")
            st.stop()

        # --- Merge for starters/leaderboard ---
        # Assume matchups has 'mlb id', 'player name', 'team code', 'game_date'
        # Normalize matchups to match event data player id/batter columns
        starter_ids = set(matchups['mlb id'].astype(str).unique())
        day = pd.to_datetime(matchups['game_date'].iloc[0], errors='coerce')
        # Filter for that day's events and starters only
        df['batter_id'] = df['batter_id'].astype(str)
        daily_events = df[df['batter_id'].isin(starter_ids) & (pd.to_datetime(df['game_date'], errors='coerce') == day)]
        st.info(f"Filtered to {len(daily_events)} daily starter events for leaderboard scoring.")

        # --- Logistic regression scoring (using user-uploaded weights) ---
        weights_map = dict(zip(logit_weights['feature'], logit_weights['weight']))
        feats = [c for c in daily_events.columns if c in weights_map]
        X = daily_events[feats].fillna(0)
        logits = X.values.dot(np.array([weights_map[f] for f in feats]))
        daily_events['logit_score'] = 1 / (1 + np.exp(-logits))
        st.markdown("#### Top 20 Starters by Logistic HR Score:")
        st.dataframe(daily_events[['batter_id', 'logit_score']].sort_values('logit_score', ascending=False).head(20))

        # --- XGBoost (if enough data) ---
        model_features = [c for c in daily_events.columns if pd.api.types.is_numeric_dtype(daily_events[c]) and c not in ['hr_outcome']]
        model_df = daily_events.dropna(subset=model_features + ['hr_outcome'], how='any')
        if len(model_df) > 50:
            X = model_df[model_features]
            y = model_df['hr_outcome'].astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
            xgb_model = xgb.XGBClassifier(n_estimators=120, max_depth=5, learning_rate=0.15,
                                          subsample=0.9, colsample_bytree=0.8, eval_metric='auc', use_label_encoder=False)
            xgb_model.fit(X_train, y_train)
            y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_prob)
            st.success(f"XGBoost Model ROC-AUC: {auc:.3f}")
            st.write(classification_report(y_test, y_pred_prob > 0.5, digits=3))
            st.markdown("#### XGBoost Feature Importances")
            st.dataframe(pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(30))
            st.download_button("⬇️ Download XGBoost Model (.pkl)",
                              data=pickle.dumps(xgb_model), file_name="xgboost_hr_model.pkl")
        else:
            st.warning("Not enough starter events for XGBoost modeling.")
        
        # --- Download leaderboard with both scores ---
        leaderboard = daily_events[['batter_id', 'logit_score'] + ([c for c in daily_events.columns if c.startswith('logit_score') or c.startswith('xgb_')])]
        st.download_button("⬇️ Download Scored Starter Leaderboard", data=leaderboard.to_csv(index=False), file_name="starter_leaderboard.csv")
