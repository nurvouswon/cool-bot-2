import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybaseball import statcast
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

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
# ================== UTILITY FUNCTIONS ==============

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

def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

# ========== Pitch type context ==============
def rolling_pitch_type_hr(df, group_col, pitch_type_col, window=5):
    # Returns rolling HR rate by pitch type for each group (batter or pitcher)
    def rolling_func(x):
        return x.shift(1).rolling(window, min_periods=1).mean()
    # Group by both group_col and pitch_type_col for correct split
    return (
        df.groupby([group_col, pitch_type_col])['hr_outcome']
        .transform(rolling_func)
    )

# ========== APP MAIN UI ==========

st.set_page_config("MLB HR Analyzer", layout="wide")
st.title("⚾ All-in-One MLB HR Analyzer & XGBoost Modeler")
st.caption("Statcast + Physics + Weather + Park + Hand Splits + Pitch Type + Contextual Factors")

tab1, tab2 = st.tabs(["1️⃣ Fetch & Feature Engineer Data", "2️⃣ Upload & Analyze"])

with tab1:
    st.header("Fetch Statcast Data & Generate Features")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", value=datetime.today())

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

        # Ensure all IDs and event names are clean
        for id_col in ['batter', 'batter_id', 'pitcher', 'pitcher_id']:
            if id_col in df.columns:
                df[id_col] = df[id_col].astype(str)

        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])

        # FILTER EVENTS to only batted ball events of interest
        valid_events = ['single', 'double', 'triple', 'homerun', 'home_run', 'field_out']
        if 'events' in df.columns:
            df = df[df['events'].astype(str).str.lower().str.replace(' ', '').isin(valid_events)].copy()

        # Robust HR OUTCOME recognition
        if 'hr_outcome' not in df.columns:
            if 'events' in df.columns:
                df['hr_outcome'] = df['events'].astype(str).str.lower().str.replace(' ', '').isin(['homerun', 'home_run']).astype(int)
            else:
                df['hr_outcome'] = np.nan

        # PARK/TEAM/CONTEXT MAPPING -- robust fallback
        if 'home_team_code' in df.columns:
            df['home_team_code'] = df['home_team_code'].astype(str).str.upper()
        if 'park' not in df.columns:
            if 'home_team_code' in df.columns:
                df['park'] = df['home_team_code'].map(team_code_to_park)
        if 'park' in df.columns and df['park'].isnull().any() and 'home_team' in df.columns:
            df['park'] = df['park'].fillna(df['home_team'].str.lower().str.replace(' ', '_'))
        elif 'home_team' in df.columns and 'park' not in df.columns:
            df['park'] = df['home_team'].str.lower().str.replace(' ', '_')
        if 'park' not in df.columns:
            st.error("Could not determine ballpark from your data (missing 'park', 'home_team_code', and 'home_team').")
            st.stop()
        df['park_hr_rate'] = df['park'].map(park_hr_rate_map).fillna(1.0)
        df['park_altitude'] = df['park'].map(park_altitude_map).fillna(0)
        df['roof_status'] = df['park'].map(roof_status_map).fillna("open")

        # Advanced: park hand HR rate (you must maintain this map if needed)
        if 'park' in df.columns and 'stand' in df.columns:
            if 'park_hand_hr_rate' not in df.columns:
                df['park_hand_hr_rate'] = (
                    df['park'].astype(str) + '_' + df['stand'].astype(str)
                ).map({})  # <-- fill with your park_hand_hr_rate_map or set as needed
                df['park_hand_hr_rate'] = df['park_hand_hr_rate'].fillna(1.0)
        else:
            df['park_hand_hr_rate'] = 1.0

        progress.progress(20, "Park/team context merged")

        # WEATHER (event-level) -- see above for weather API function
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
                progress.progress(20 + int(30 * (i+1) / len(unique_keys)), text=f"Weather {i+1}/{len(unique_keys)}")
            progress.progress(50, "Weather merged")
        else:
            for feat in weather_features:
                df[feat] = None

        df['wind_dir_angle'] = df['wind_dir'].apply(wind_dir_to_angle)
        df['wind_dir_sin'] = np.sin(np.deg2rad(df['wind_dir_angle']))
        df['wind_dir_cos'] = np.cos(np.deg2rad(df['wind_dir_angle']))

        # Advanced rolling physics/Statcast features + hand splits + pitch type
        progress.progress(60, "Advanced rolling/statcast/hand/pitch features...")
        roll_windows = [3, 5, 7, 14]
        batter_cols = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value',
                       'release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z']
        pitcher_cols = batter_cols

        # Add missing batter_id, pitcher_id if only have batter/pitcher
        if 'batter_id' not in df.columns and 'batter' in df.columns:
            df['batter_id'] = df['batter']
        if 'pitcher_id' not in df.columns and 'pitcher' in df.columns:
            df['pitcher_id'] = df['pitcher']

        # Batters
        batter_feat_dict = {}
        for col in batter_cols:
            if col in df.columns:
                for w in roll_windows:
                    cname = f'B_{col}_{w}'
                    batter_feat_dict[cname] = df.groupby('batter_id')[col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        # Pitchers
        pitcher_feat_dict = {}
        for col in pitcher_cols:
            if col in df.columns:
                for w in roll_windows:
                    cname = f'P_{col}_{w}'
                    pitcher_feat_dict[cname] = df.groupby('pitcher_id')[col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        # Batter-vs-Pitcher Hand Splits
        if 'stand' in df.columns and 'p_throws' in df.columns:
            for w in roll_windows:
                df[f'B_vsP_hand_HR_{w}'] = df.groupby(['batter_id', 'p_throws'])['hr_outcome'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                df[f'P_vsB_hand_HR_{w}'] = df.groupby(['pitcher_id', 'stand'])['hr_outcome'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        # Rolling HR rate by pitch type (batter and pitcher)
        if 'pitch_type' in df.columns:
            for w in roll_windows:
                df[f'B_pitchtype_HR_{w}'] = rolling_pitch_type_hr(df, 'batter_id', 'pitch_type', window=w)
                df[f'P_pitchtype_HR_{w}'] = rolling_pitch_type_hr(df, 'pitcher_id', 'pitch_type', window=w)

        # Merge features
        df = pd.concat([df, pd.DataFrame(batter_feat_dict), pd.DataFrame(pitcher_feat_dict)], axis=1)
        df = df.copy()
        progress.progress(80, "All rolling/physics/context features complete")

        # Weather x Statcast interactions
        for col in ['is_barrel', 'is_hard_hit']:
            if col in df.columns and all(x in df.columns for x in ['humidity', 'temp', 'wind_mph']):
                df[f'{col}_x_humidity'] = df[col] * df['humidity']
                df[f'{col}_x_temp'] = df[col] * df['temp']
                df[f'{col}_x_wind_mph'] = df[col] * df['wind_mph']

        # Context/categorical encoding
        context_features = [
            'park_hr_rate', 'park_altitude', 'park_hand_hr_rate', 'temp', 'humidity', 'wind_mph',
            'wind_dir_angle', 'wind_dir_sin', 'wind_dir_cos'
        ]
        for c in context_features:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        if 'roof_status' in df.columns:
            df = pd.get_dummies(df, columns=['roof_status'], drop_first=True)

        df = dedup_columns(df)

        progress.progress(95, "All features complete")
        st.success(f"Feature engineering complete! {len(df)} batted ball events.")
        st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
        st.dataframe(df.head(20))
        st.download_button("⬇️ Download Event-Level CSV", data=df.to_csv(index=False), file_name="event_level_hr_features.csv")
        progress.empty()

        # --- Robust Logistic Regression Weights Calculation ---

        if 'hr_outcome' in df.columns:
        # Show distribution of HR outcomes
            hr_counts = df['hr_outcome'].value_counts(dropna=False)
            st.write(f"HR outcome value counts: {hr_counts.to_dict()}")

        # Build feature list: numeric + valid context features
            model_features = robust_numeric_columns(df) + [c for c in context_features if c in df.columns and c != 'hr_outcome']
            model_features = [c for c in model_features if c != 'hr_outcome']

        # Diagnostic: see what features are present/missing
            model_features_initial = model_features.copy()
            feature_na_fracs = {c: df[c].isna().mean() for c in model_features}

        # Only keep features with less than 20% missing values
            model_features = [c for c in model_features if feature_na_fracs[c] < 0.2]
            dropped_feats = set(model_features_initial) - set(model_features)
            if dropped_feats:
                st.info(f"Dropped features due to NAs: {dropped_feats}")
            if not model_features:
                st.error("No usable model features (all too many missing values).")
                st.stop()

        # Drop rows with NA ONLY on the model features (and hr_outcome)
            model_df = df.dropna(subset=model_features + ['hr_outcome'], how='any')
            st.write(f"After NA drop: {dict(model_df['hr_outcome'].value_counts())}, n={len(model_df)}")
        # Require at least 30 rows, and at least 2 classes
            if model_df['hr_outcome'].nunique() < 2 or len(model_df) < 30:
                st.warning("Not enough HR/non-HR events for logistic regression weights (check date range or missing features).")
                st.stop()

        # Fit logistic regression
            X = model_df[model_features].fillna(0)
            y = model_df['hr_outcome'].astype(int)
            logit = LogisticRegression(max_iter=200, solver='liblinear')
            logit.fit(X, y)
            weights = pd.DataFrame({
                'feature': model_features,
                'weight': logit.coef_[0],
                'intercept': logit.intercept_[0]
            })
            st.dataframe(weights.sort_values('weight', ascending=False))
            st.download_button("⬇️ Download Logistic Weights CSV", data=weights.to_csv(index=False), file_name="logit_weights.csv")

with tab2:
    st.header("Upload Event, Matchup, and Analyze")
    st.markdown("**All 3 uploads required!**")
    uploaded_events = st.file_uploader("Upload Event-Level Features CSV", type="csv", key="evup")
    uploaded_matchups = st.file_uploader("Upload Matchups CSV", type="csv", key="mup")
    uploaded_logit = st.file_uploader("Upload Logistic Weights CSV", type="csv", key="lup")
    analyze_btn = st.button("Run Analysis (Logit + XGBoost Leaderboard)", type="primary")

    if analyze_btn:
        # ========== LOAD DATA ==========
        if not uploaded_events or not uploaded_matchups or not uploaded_logit:
            st.warning("Please upload event-level, matchup, and logistic weights CSVs before running analysis.")
            st.stop()
        df = pd.read_csv(uploaded_events)
        matchups = pd.read_csv(uploaded_matchups)
        logit_weights = pd.read_csv(uploaded_logit)

        st.write(f"Loaded {len(df)} events, {len(matchups)} matchup rows, {len(logit_weights)} logistic weights.")

        # ========== FILTER EVENTS TO ONLY VALID BATTERS ==========
        # Use 'mlb id' or 'player name' to match
        if 'batter_id' in df.columns and 'mlb id' in matchups.columns:
            merge_on = ('batter_id', 'mlb id')
        elif 'batter' in df.columns and 'player name' in matchups.columns:
            merge_on = ('batter', 'player name')
        else:
            st.error("Could not match batters between event-level data and matchup file. Check columns.")
            st.stop()

        # Merge events with today's matchups to get only today's batters
        today_batters = matchups[merge_on[1]].unique()
        event_df = df[df[merge_on[0]].isin(today_batters)].copy()
        if event_df.empty:
            st.error("No matching batters found in event-level data for today's lineup.")
            st.stop()
        # Merge in matchup info for display
        event_df = event_df.merge(matchups, left_on=merge_on[0], right_on=merge_on[1], how='left', suffixes=('', '_mu'))

        # ========== PREPARE FEATURES ==========
        model_features = [f for f in logit_weights['feature'].values if f in event_df.columns and pd.api.types.is_numeric_dtype(event_df[f])]
        if not model_features:
            st.error("None of the model features from weights file are present in your data.")
            st.stop()
        if 'hr_outcome' not in event_df.columns:
            st.warning("No hr_outcome column in the event-level data! Scoring only.")

        X = event_df[model_features].fillna(0)
        coef = logit_weights.set_index('feature')['weight'].reindex(model_features).fillna(0).values
        intercept = logit_weights['intercept'].iloc[0] if 'intercept' in logit_weights.columns else 0

        # ========== LOGISTIC REGRESSION SCORING ==========
        event_df['logit_score'] = intercept + np.dot(X, coef)
        event_df['logit_prob'] = 1 / (1 + np.exp(-event_df['logit_score']))

        # ========== XGBOOST MODEL (TRAIN/INFER) ==========
        # If hr_outcome present, train XGBoost, otherwise skip
        if 'hr_outcome' in event_df.columns and event_df['hr_outcome'].nunique() > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, event_df['hr_outcome'], test_size=0.2, random_state=42
            )
            xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.13, use_label_encoder=False, eval_metric='logloss')
            xgb_model.fit(X_train, y_train)
            event_df['xgb_prob'] = xgb_model.predict_proba(X)[:, 1]
            # Show XGB AUC
            auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
            st.metric("XGBoost ROC-AUC", round(auc, 4))
            st.markdown("### Classification Reports")
            st.code(classification_report(y_test, xgb_model.predict(X_test)), language='text')
        else:
            event_df['xgb_prob'] = np.nan
            st.info("Not enough HR/non-HR events to fit XGBoost. Only logit scoring shown.")

        # ========== LEADERBOARD ==========
        st.markdown("### HR Leaderboard (Today's Batters)")
        # Get best row per batter for leaderboard (use max logit_prob)
        leaderboard = (
            event_df.groupby(merge_on[0])
            .agg(
                batter_name=('batter' if 'batter' in event_df.columns else merge_on[0], 'first'),
                team=('team code' if 'team code' in event_df.columns else 'team', 'first'),
                batting_order=('batting order' if 'batting order' in event_df.columns else 'batting_order', 'first'),
                logit_prob=('logit_prob', 'max'),
                xgb_prob=('xgb_prob', 'max'),
            )
            .reset_index()
            .sort_values('logit_prob', ascending=False)
        )
        st.dataframe(leaderboard.head(15), use_container_width=True)

        # ========== DOWNLOADABLES ==========
        st.markdown("#### Download Full Event-Level Data with Model Scores:")
        st.download_button("⬇️ Download Scored Event CSV", data=event_df.to_csv(index=False), file_name="event_level_scored.csv")

        # CLASSIFICATION REPORTS
        st.markdown("### Classification Reports")
        st.code(classification_report(y_test, xgb_model.predict(X_test)), language='text')

        # ROC AUC
        auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
        st.metric("XGBoost ROC-AUC", round(auc, 4))
        # === Logistic Regression Classifier Performance ===

        from sklearn.linear_model import LogisticRegression

        # Use the same train/test split as for XGBoost
        logit_model = LogisticRegression(max_iter=200, solver='liblinear', class_weight='balanced')
        logit_model.fit(X_train, y_train)
        logit_pred = logit_model.predict(X_test)
        logit_proba = logit_model.predict_proba(X_test)[:, 1]

        st.markdown("### Logistic Regression Performance")
        st.metric("Logistic Regression ROC-AUC", round(roc_auc_score(y_test, logit_proba), 4))
        st.code(classification_report(y_test, logit_pred), language='text')

        st.success("Analysis complete!")
