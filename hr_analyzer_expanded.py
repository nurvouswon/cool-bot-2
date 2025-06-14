import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from pybaseball import statcast
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
import pickle

# ========= CONTEXT MAPS ==========
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

# ================= UTILITY FUNCTIONS =================
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

def robust_hr_outcome_column(df):
    if 'hr_outcome' not in df.columns:
        for col in ['events', 'description']:
            if col in df.columns:
                mask = df[col].astype(str).str.lower().str.contains("home_run|homer|homers|homered")
                df['hr_outcome'] = np.where(mask, 1, 0)
                return df
        df['hr_outcome'] = 0
    return df

def robust_numeric_columns(df):
    # Return only columns that are robust numeric for modeling
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]

# =============== ADVANCED ROLLING FEATURE ENGINEERING ===============
def rolling_features(df):
    df = df.copy()
    if not all(x in df.columns for x in ['batter_id', 'pitcher_id', 'game_date']):
        return df
    df = df.sort_values(['batter_id', 'game_date']).reset_index(drop=True)
    ROLL = [3, 5, 7, 14]
    batter_stats = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value']
    pitcher_stats = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value', 'hr_outcome']
    pitch_types = ['SL','SI','FC','FF','ST','CH','CU','FS','FO','SV','KC','EP','FA','KN','CS','SC']

    batter_feats, pitcher_feats, pitchmix_feats = {}, {}, {}

    # Rolling means/max/median for batter
    for stat in batter_stats:
        if stat in df.columns:
            s = df.groupby('batter_id')[stat]
            for w in ROLL:
                batter_feats[f'B_{stat}_{w}'] = s.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                if stat == 'launch_speed':
                    batter_feats[f'B_max_ev_{w}'] = s.transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
                    batter_feats[f'B_median_ev_{w}'] = s.transform(lambda x: x.shift(1).rolling(w, min_periods=1).median())
    # Pitcher rolling means/max/median, rolling HR rate
    for stat in pitcher_stats:
        if stat in df.columns:
            s = df.groupby('pitcher_id')[stat]
            for w in ROLL:
                pitcher_feats[f'P_{stat}_{w}'] = s.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                if stat == 'launch_speed':
                    pitcher_feats[f'P_max_ev_{w}'] = s.transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
                    pitcher_feats[f'P_median_ev_{w}'] = s.transform(lambda x: x.shift(1).rolling(w, min_periods=1).median())
    # Pitch mix: batter and pitcher
    for pt in pitch_types:
        if 'pitch_type' in df.columns:
            batter_group = df.groupby('batter_id')['pitch_type']
            pitcher_group = df.groupby('pitcher_id')['pitch_type']
            for w in ROLL:
                pitchmix_feats[f'B_pitch_pct_{pt}_{w}'] = batter_group.transform(
                    lambda x: x.shift(1).eq(pt).rolling(w, min_periods=1).mean())
                pitchmix_feats[f'P_pitch_pct_{pt}_{w}'] = pitcher_group.transform(
                    lambda x: x.shift(1).eq(pt).rolling(w, min_periods=1).mean())

    # Apply all at once for perf
    roll_df = pd.concat(
        [df] +
        [pd.DataFrame(batter_feats, index=df.index)] +
        [pd.DataFrame(pitcher_feats, index=df.index)] +
        [pd.DataFrame(pitchmix_feats, index=df.index)],
        axis=1
    )
    roll_df = roll_df.copy()  # Defragment
    return roll_df

# ================= STREAMLIT APP ===================
st.title("⚾️ All-in-One MLB HR Analyzer & Modeler")

mode = st.radio("Select mode:", ["Fetch Data (Feature Engineering)", "Analyze/Model (Logistic & XGBoost)"], horizontal=True)

if mode == "Fetch Data (Feature Engineering)":
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", value=datetime.today())

    run_query = st.button("Fetch Statcast & Run Feature Engineering")
    if run_query:
        df = statcast(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        st.success(f"Loaded {len(df)} events.")
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
            if 'home_team' in df.columns:
                df['park'] = df['home_team_code'].map(team_code_to_park).fillna(df['home_team'].str.lower().str.replace(' ', '_'))
            else:
                df['park'] = df['home_team_code'].map(team_code_to_park)
        df['park_hr_rate'] = df['park'].map(park_hr_rate_map).fillna(1.0)
        df['park_altitude'] = df['park'].map(park_altitude_map).fillna(0)
        df['roof_status'] = df['park'].map(roof_status_map).fillna("open")

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
                st.progress(5 + int(35 * (i+1) / len(unique_keys)), text=f"Weather {i+1}/{len(unique_keys)}")
            st.progress(40, text="Weather merged")
        else:
            for feat in weather_features:
                df[feat] = None

        # Wind trig encoding
        df['wind_dir_angle'] = df['wind_dir'].apply(wind_dir_to_angle)
        df['wind_dir_sin'] = np.sin(np.deg2rad(df['wind_dir_angle']))
        df['wind_dir_cos'] = np.cos(np.deg2rad(df['wind_dir_angle']))

        # Add hr_outcome if needed
        df = robust_hr_outcome_column(df)

        # Add advanced rolling features (modular, no fragmentation)
        df = rolling_features(df)

        # Output CSVs
        st.markdown("### Download Event-Level CSV (All Features + hr_outcome)")
        st.dataframe(df.head(25))
        st.download_button("⬇️ Download Event-Level CSV", data=df.to_csv(index=False), file_name="event_level_hr_features.csv")

elif mode == "Analyze/Model (Logistic & XGBoost)":
    uploaded_event_csv = st.file_uploader("Upload Event-Level CSV (required, includes hr_outcome)", type=["csv"])
    uploaded_matchups = st.file_uploader("Upload Daily Lineups / Matchups CSV (required)", type=["csv"])
    uploaded_logit = st.file_uploader("Upload Logistic Weights CSV (optional, for leaderboard blend)", type=["csv"])

    if uploaded_event_csv and uploaded_matchups:
        df = pd.read_csv(uploaded_event_csv)
        if 'hr_outcome' not in df.columns:
            st.error("No 'hr_outcome' column detected! Add or compute this before modeling.")
            st.stop()
        # Add advanced rolling features if not present
        if 'B_launch_speed_7' not in df.columns and 'batter_id' in df.columns:
            with st.spinner("Adding advanced rolling features..."):
                df = rolling_features(df)
        # Model features
        model_features = [c for c in robust_numeric_columns(df) if c != 'hr_outcome']
        model_df = df.dropna(subset=model_features + ['hr_outcome'], how='any')
        # Logistic Regression
        if len(model_df) > 50:
            X = model_df[model_features]
            y = model_df['hr_outcome'].astype(int)
            # Logistic Regression model
            logit_model = LogisticRegression(max_iter=500)
            logit_model.fit(X, y)
            y_pred_prob = logit_model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, y_pred_prob)
            st.success(f"Logistic Regression ROC-AUC: {auc:.3f}")
            st.write(classification_report(y, y_pred_prob > 0.5, digits=3))

            # Logistic regression feature weights
            logit_weights = pd.DataFrame({
                'feature': model_features,
                'weight': logit_model.coef_[0]
            }).sort_values('weight', key=abs, ascending=False)
            st.markdown("#### Download Logistic Weights CSV")
            st.dataframe(logit_weights.head(40))
            st.download_button(
                "⬇️ Download Logistic Weights CSV",
                data=logit_weights.to_csv(index=False),
                file_name="logistic_weights.csv"
            )
        else:
            st.warning("Not enough events for logistic regression.")

        # XGBoost Model
        if len(model_df) > 50:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.15, random_state=42, stratify=y
            )
            xgb_model = xgb.XGBClassifier(
                n_estimators=120, max_depth=5, learning_rate=0.15,
                subsample=0.9, colsample_bytree=0.8, eval_metric='auc', use_label_encoder=False
            )
            xgb_model.fit(X_train, y_train)
            y_pred_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
            auc_xgb = roc_auc_score(y_test, y_pred_prob_xgb)
            st.success(f"XGBoost ROC-AUC: {auc_xgb:.3f}")
            st.write(classification_report(y_test, y_pred_prob_xgb > 0.5, digits=3))
            xgb_feat_import = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.markdown("#### XGBoost Feature Importances")
            st.dataframe(xgb_feat_import.head(40))
            st.download_button(
                "⬇️ Download XGBoost Model (.pkl)",
                data=pickle.dumps(xgb_model),
                file_name="xgboost_hr_model.pkl"
            )
        else:
            st.warning("Not enough complete events for XGBoost modeling.")

        # ===== Output event-level CSV (now with model columns) =====
        st.markdown("#### Download Event-Level CSV (All Features + hr_outcome):")
        st.dataframe(df.head(25))
        st.download_button("⬇️ Download Event-Level CSV", data=df.to_csv(index=False), file_name="event_level_hr_features.csv")

        # ===== OPTIONAL: Load matchup file for leaderboard filtering =====
        matchups = pd.read_csv(uploaded_matchups)
        if 'mlb id' in matchups.columns or 'batter_id' in matchups.columns:
            # Accept both formats, standardize
            matchup_batters = set(matchups.get('mlb id', pd.Series()).dropna().astype(str).tolist())
            if 'batter_id' in df.columns:
                leaderboard = df[df['batter_id'].astype(str).isin(matchup_batters)].copy()
            else:
                leaderboard = df.copy()
        else:
            leaderboard = df.copy()

        # Leaderboard scoring (Logit + XGB, if models fit)
        if len(model_df) > 0 and 'batter_id' in leaderboard.columns:
            leaderboard['logit_score'] = logit_model.predict_proba(leaderboard[model_features].fillna(0))[:, 1]
            leaderboard['xgb_score'] = xgb_model.predict_proba(leaderboard[model_features].fillna(0))[:, 1]
            st.markdown("### Leaderboard (Top 20, Sorted by XGBoost Score):")
            st.dataframe(leaderboard.sort_values("xgb_score", ascending=False).head(20)[
                ['batter_id', 'logit_score', 'xgb_score', 'hr_outcome'] +
                [c for c in leaderboard.columns if c.startswith('B_max_ev_')]
            ])
            st.download_button(
                "⬇️ Download Leaderboard CSV",
                data=leaderboard.sort_values("xgb_score", ascending=False).to_csv(index=False),
                file_name="leaderboard_scored.csv"
            )
        else:
            st.warning("Leaderboard not available. Check for required columns and sufficient data.")
    else:
        st.warning("You must upload both event-level CSV and matchup CSV to start analysis.")

# End of file
