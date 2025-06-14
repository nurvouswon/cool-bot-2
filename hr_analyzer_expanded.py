import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb

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

# ========== Utility Functions ==========
def wind_dir_to_angle(wind_dir):
    directions = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5,
        'SE': 135, 'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    for d in directions:
        if pd.notna(wind_dir) and d in str(wind_dir).upper():
            return directions[d]
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

# ========== Streamlit UI ==========
st.title("⚾ All-in-One MLB HR Analyzer & XGBoost/Logit Modeler")
st.markdown("""
Uploads, fetches and builds full Statcast + Weather + Park + Pitch Mix + Categorical + Advanced interaction features,
**and allows training and using Logistic Regression and XGBoost for HR event prediction!**
""")

mode = st.radio("Choose mode", ["Fetch & Feature Engineer", "Upload & Analyze"], horizontal=True)

if mode == "Fetch & Feature Engineer":
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", value=datetime.today())
    run_query = st.button("Fetch Statcast Data and Engineer Features")

    if run_query:
        with st.spinner("Fetching Statcast..."):
            from pybaseball import statcast
            df = statcast(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        st.success(f"Loaded {len(df)} events.")

        # ------- Ensure hr_outcome is present -------
        if 'hr_outcome' not in df.columns:
            if 'events' in df.columns:
                df['hr_outcome'] = df['events'].astype(str).str.lower().eq('home_run').astype(int)
            elif 'description' in df.columns:
                df['hr_outcome'] = df['description'].astype(str).str.contains('home run', case=False, na=False).astype(int)
            else:
                df['hr_outcome'] = 0

        # ------- Minimal Cleaning -------
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

        # ------- Weather API Integration -------
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
                st.progress(int(100 * (i+1) / len(unique_keys)), text=f"Weather {i+1}/{len(unique_keys)}")
            st.success("Weather merged.")

        # ------- Feature Engineering -------
        # Wind direction encoding
        df['wind_dir_angle'] = df['wind_dir'].apply(wind_dir_to_angle)
        df['wind_dir_sin'] = np.sin(np.deg2rad(df['wind_dir_angle']))
        df['wind_dir_cos'] = np.cos(np.deg2rad(df['wind_dir_angle']))

        # Indicator features
        df['is_home'] = ((df['home_team_code'] == df['batter'].map(lambda x: str(x)[:3]) if 'batter' in df.columns and 'home_team_code' in df.columns else False)).astype(int)
        df['is_high_leverage'] = ((df['inning'] >= 8) & (df['bat_score_diff'].abs() <= 2)).astype(int) if 'inning' in df.columns and 'bat_score_diff' in df.columns else 0
        df['is_late_inning'] = (df['inning'] >= 8).astype(int) if 'inning' in df.columns else 0
        df['is_early_inning'] = (df['inning'] <= 3).astype(int) if 'inning' in df.columns else 0
        df['runners_on'] = (df[['on_1b', 'on_2b', 'on_3b']].fillna(0).astype(int).sum(axis=1) > 0).astype(int) if set(['on_1b','on_2b','on_3b']).issubset(df.columns) else 0

        # Batted Ball Type Binary Flags
        if 'bb_type' in df.columns:
            df['bb_type_fly_ball'] = (df['bb_type']=='fly_ball').astype(int)
            df['bb_type_line_drive'] = (df['bb_type']=='line_drive').astype(int)
            df['bb_type_ground_ball'] = (df['bb_type']=='ground_ball').astype(int)
            df['bb_type_popup'] = (df['bb_type']=='popup').astype(int)

        # Weather/Batted Ball Interactions
        for feat in ['is_hard_hit','flyball','pull_air','is_barrel']:
            if feat in df.columns:
                df[f'{feat}_x_temp'] = df[feat] * df['temp']
                df[f'{feat}_x_humidity'] = df[feat] * df['humidity']
                df[f'{feat}_x_wind_mph'] = df[feat] * df['wind_mph']

        # Batch rolling features (optimized, no fragmentation)
        ROLL = [3,5,7,14]
        new_cols = {}
        batter_stats = ['launch_speed','launch_angle','hit_distance_sc','woba_value']
        for stat in batter_stats:
            for w in ROLL:
                cname = f'B_{stat}_{w}'
                if cname not in df.columns and stat in df.columns:
                    new_cols[cname] = df.groupby('batter_id')[stat].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        for w in ROLL:
            if 'launch_speed' in df.columns:
                new_cols[f'B_max_ev_{w}'] = df.groupby('batter_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
                new_cols[f'B_median_ev_{w}'] = df.groupby('batter_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).median())
        pitcher_stats = ['launch_speed','launch_angle','hit_distance_sc','woba_value']
        for stat in pitcher_stats:
            for w in ROLL:
                cname = f'P_{stat}_{w}'
                if cname not in df.columns and stat in df.columns:
                    new_cols[cname] = df.groupby('pitcher_id')[stat].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        for w in ROLL:
            if 'launch_speed' in df.columns:
                new_cols[f'P_max_ev_{w}'] = df.groupby('pitcher_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
                new_cols[f'P_median_ev_{w}'] = df.groupby('pitcher_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).median())
            if 'hr_outcome' in df.columns:
                new_cols[f'P_rolling_hr_rate_{w}'] = df.groupby('pitcher_id')['hr_outcome'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())

        # Pitch type mix, one-hot encoding (optimized)
        pitch_types = ['SL','SI','FC','FF','ST','CH','CU','FS','FO','SV','KC','EP','FA','KN','CS','SC']
        for pt in pitch_types:
            if 'pitch_type' in df.columns and 'batter_id' in df.columns:
                for w in ROLL:
                    new_cols[f'B_pitch_pct_{pt}_{w}'] = df.groupby('batter_id')['pitch_type'].transform(lambda x: x.shift(1).eq(pt).rolling(w, min_periods=1).mean())
            if 'pitch_type' in df.columns and 'pitcher_id' in df.columns:
                for w in ROLL:
                    new_cols[f'P_pitch_pct_{pt}_{w}'] = df.groupby('pitcher_id')['pitch_type'].transform(lambda x: x.shift(1).eq(pt).rolling(w, min_periods=1).mean())

        # Batch concat all rolling/stat cols
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
            df = df.copy()  # Defragment

        # Park-Handed HR Rate
        df = compute_park_handed_hr_rate(df)

        # Dummies (categorical features)
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

        # Final cleanup to remove duplicate columns (if any)
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # ========== DOWNLOAD EVENT-LEVEL CSV ==========
        st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
        st.dataframe(df.head(20))
        st.download_button("⬇️ Download Event-Level CSV", data=df.to_csv(index=False), file_name="event_level_hr_features.csv")

        # ========== OPTIONAL: FIT LOGISTIC REGRESSION FOR WEIGHTS & DOWNLOAD ==========
        model_features = [c for c in df.columns if c not in ['hr_outcome', 'batter', 'batter_id', 'pitcher', 'pitcher_id', 'game_date', 'events', 'description']]
        model_features = [c for c in model_features if np.issubdtype(df[c].dtype, np.number) and df[c].nunique() > 1]
        model_df = df.dropna(subset=model_features + ['hr_outcome'], how='any')
        logit_weights = pd.DataFrame()
        logit_model = None
        if len(model_df) > 30:
            X = model_df[model_features]
            y = model_df['hr_outcome'].astype(int)
            logit_model = LogisticRegression(max_iter=1000)
            logit_model.fit(X, y)
            logit_weights = pd.DataFrame({"feature": model_features, "weight": logit_model.coef_[0]})
            st.success("Logistic regression weights fit!")
            st.download_button("⬇️ Download Logistic Weights CSV", data=logit_weights.to_csv(index=False), file_name="logit_weights.csv")
        else:
            st.warning("Not enough complete events for logistic regression weights. Still safe to download event-level CSV.")

# ========== ANALYZE MODE ==========
elif mode == "Upload & Analyze":
    st.markdown("**Upload all three files to begin:**")
    uploaded_event_csv = st.file_uploader("Upload Event-Level CSV", type=["csv"], key="evt")
    uploaded_matchups = st.file_uploader("Upload Matchups CSV", type=["csv"], key="mtc")
    uploaded_logit_weights = st.file_uploader("Upload Logistic Weights CSV", type=["csv"], key="lwt")
    analyze_btn = st.button("Analyze & Score Models")

    if analyze_btn:
        # Require all three
        if not (uploaded_event_csv and uploaded_matchups and uploaded_logit_weights):
            st.error("Please upload all three files before analysis.")
            st.stop()
        # Load all
        df = pd.read_csv(uploaded_event_csv)
        matchups_df = pd.read_csv(uploaded_matchups)
        logit_weights = pd.read_csv(uploaded_logit_weights)

        # Ensure hr_outcome
        if 'hr_outcome' not in df.columns:
            if 'events' in df.columns:
                df['hr_outcome'] = df['events'].astype(str).str.lower().eq('home_run').astype(int)
            elif 'description' in df.columns:
                df['hr_outcome'] = df['description'].astype(str).str.contains('home run', case=False, na=False).astype(int)
            else:
                st.error("No 'hr_outcome' column detected. Please include this in your event-level CSV.")
                st.stop()

        # Merge matchups if needed (optional, only if you want to join columns)
        # Example: df = pd.merge(df, matchups_df, on=['game_date', 'batter'], how='left')

        # Clean dups
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # Align logit weights with event-level columns
        features_avail = [f for f in logit_weights['feature'] if f in df.columns]
        if not features_avail:
            st.error("None of the logistic regression features are present in the event-level data. Please check your CSVs.")
            st.stop()
        model_df = df.dropna(subset=features_avail + ['hr_outcome'], how='any')
        if len(model_df) < 30:
            st.error("Not enough complete events for analysis. Check your event-level CSV and filters.")
            st.stop()

        # --------- Logistic Regression Scoring ---------
        X_logit = model_df[features_avail]
        weights = logit_weights.set_index('feature').loc[features_avail]['weight'].values
        logit_scores = np.dot(X_logit, weights)
        # Scale scores 0-1 with sigmoid
        model_df['logit_hr_prob'] = 1 / (1 + np.exp(-logit_scores))

        # --------- XGBoost Scoring ---------
        st.markdown("#### Fitting XGBoost Classifier for HR Likelihood")
        X_xgb = model_df[features_avail]
        y_xgb = model_df['hr_outcome'].astype(int)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_xgb, y_xgb, test_size=0.15, random_state=42, stratify=y_xgb
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
            feature_importance = pd.Series(xgb_model.feature_importances_, index=X_xgb.columns).sort_values(ascending=False)
            st.markdown("#### XGBoost Feature Importances")
            st.dataframe(feature_importance.head(50))
            st.download_button(
                "⬇️ Download XGBoost Model (.pkl)",
                data=pickle.dumps(xgb_model),
                file_name="xgboost_hr_model.pkl"
            )
            # Add XGBoost predictions to df
            model_df['xgb_hr_prob'] = xgb_model.predict_proba(X_xgb)[:, 1]
        except Exception as e:
            st.error(f"XGBoost model error: {e}")
            model_df['xgb_hr_prob'] = np.nan

        # --------- Leaderboard Output ---------
        leaderboard_cols = ['game_date', 'batter', 'hr_outcome', 'logit_hr_prob', 'xgb_hr_prob']
        leaderboard_cols = [col for col in leaderboard_cols if col in model_df.columns]
        leaderboard = model_df.sort_values('logit_hr_prob', ascending=False)[leaderboard_cols].head(100)
        st.markdown("### HR Prediction Leaderboard (Top 100 by Logit Model)")
        st.dataframe(leaderboard)

        # Download full analysis CSV
        st.markdown("#### Download Scored Event-Level CSV")
        st.download_button(
            "⬇️ Download Scored Event-Level CSV",
            data=model_df.to_csv(index=False),
            file_name="event_level_hr_features_scored.csv"
        )

        st.success("Analysis complete! Both logistic regression and XGBoost HR scores are included.")
