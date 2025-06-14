import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybaseball import statcast
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
import pickle

# ======= Static Context Maps =======
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

def wind_dir_to_angle(wind_dir):
    compass = {'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5,
        'SE': 135, 'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5}
    if pd.isna(wind_dir): return np.nan
    for k in compass:
        if k in str(wind_dir).upper(): return compass[k]
    return np.nan

@st.cache_data(show_spinner=False)
def get_weather(city, date):
    api_key = st.secrets["weather"]["api_key"]
    url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={date}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200 and resp.text.strip():
            data = resp.json()
            hours = data['forecast']['forecastday'][0]['hour']
            hour_data = min(hours, key=lambda h: abs(int(h['time'].split(' ')[1].split(':')[0]) - 19))
            return { 'temp': hour_data['temp_f'], 'wind_mph': hour_data['wind_mph'],
                    'wind_dir': hour_data['wind_dir'], 'humidity': hour_data['humidity'],
                    'condition': hour_data['condition']['text']}
    except Exception as e:
        st.warning(f"Weather API error for {city} {date}: {e}")
    return {k: None for k in ['temp','wind_mph','wind_dir','humidity','condition']}

def compute_park_handed_hr_rate(df):
    if all(col in df.columns for col in ['stand', 'p_throws', 'hr_outcome', 'park']):
        df['handed_matchup'] = df['stand'].astype(str) + df['p_throws'].astype(str)
        rate = df.groupby(['park', 'handed_matchup'])['hr_outcome'].mean().reset_index().rename(
            columns={'hr_outcome':'park_handed_hr_rate'})
        df = df.merge(rate, on=['park','handed_matchup'], how='left')
    else:
        df['park_handed_hr_rate'] = np.nan
    return df

def batch_rolling(df, group_col, stats, windows):
    results = []
    for stat in stats:
        if stat in df.columns:
            arrs = []
            for w in windows:
                arrs.append(df.groupby(group_col)[stat].shift(1).rolling(w, min_periods=1).mean().rename(f"{group_col[0]}_{stat}_{w}"))
                arrs.append(df.groupby(group_col)[stat].shift(1).rolling(w, min_periods=1).max().rename(f"{group_col[0]}_max_{stat}_{w}"))
                arrs.append(df.groupby(group_col)[stat].shift(1).rolling(w, min_periods=1).median().rename(f"{group_col[0]}_median_{stat}_{w}"))
            results.append(pd.concat(arrs, axis=1))
    return pd.concat(results, axis=1) if results else pd.DataFrame(index=df.index)

def feature_engineer(df):
    df = df.copy()
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
        df['park'] = df['home_team_code'].map(team_code_to_park).fillna(df['home_team'].str.lower().str.replace(' ', '_'))
    df['park_hr_rate'] = df['park'].map(park_hr_rate_map).fillna(1.0)
    df['park_altitude'] = df['park'].map(park_altitude_map).fillna(0)
    df['roof_status'] = df['park'].map(roof_status_map).fillna("open")

    # Wind features
    if 'wind_dir' in df.columns:
        df['wind_dir_angle'] = df['wind_dir'].apply(wind_dir_to_angle)
    else:
        df['wind_dir_angle'] = np.nan
    df['wind_dir_sin'] = np.sin(np.deg2rad(df['wind_dir_angle']))
    df['wind_dir_cos'] = np.cos(np.deg2rad(df['wind_dir_angle']))

    # Indicator/categorical features
    df['is_home'] = ((df['home_team_code'] == df['batter'].map(lambda x: str(x)[:3])) if 'batter' in df.columns and 'home_team_code' in df.columns else False).astype(int)
    df['is_high_leverage'] = ((df['inning'] >= 8) & (df['bat_score_diff'].abs() <= 2)).astype(int) if 'inning' in df.columns and 'bat_score_diff' in df.columns else 0
    df['is_late_inning'] = (df['inning'] >= 8).astype(int) if 'inning' in df.columns else 0
    df['is_early_inning'] = (df['inning'] <= 3).astype(int) if 'inning' in df.columns else 0
    df['runners_on'] = (df[['on_1b', 'on_2b', 'on_3b']].fillna(0).astype(int).sum(axis=1) > 0).astype(int) if set(['on_1b','on_2b','on_3b']).issubset(df.columns) else 0

    # Batted Ball Type Flags
    if 'bb_type' in df.columns:
        df['bb_type_fly_ball'] = (df['bb_type']=='fly_ball').astype(int)
        df['bb_type_line_drive'] = (df['bb_type']=='line_drive').astype(int)
        df['bb_type_ground_ball'] = (df['bb_type']=='ground_ball').astype(int)
        df['bb_type_popup'] = (df['bb_type']=='popup').astype(int)

    # Batted Ball Flags (fillna, ensure int)
    for col in ['is_barrel', 'is_sweet_spot', 'is_hard_hit', 'flyball', 'line_drive','groundball', 'pull_air', 'pull_side']:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Weather/Batted Ball Interactions
    for feat in ['is_barrel', 'is_hard_hit', 'flyball', 'pull_air']:
        for feat2 in ['temp', 'humidity', 'wind_mph']:
            if feat in df.columns and feat2 in df.columns:
                df[f"{feat}_x_{feat2}"] = df[feat] * df[feat2]

    # Rolling features (batters and pitchers)
    rolling_windows = [3,5,7,14]
    batter_stats = ['launch_speed','launch_angle','hit_distance_sc','woba_value']
    pitcher_stats = ['launch_speed','launch_angle','hit_distance_sc','woba_value']

    # Batch-rolling (efficient, single assignment)
    if 'batter_id' in df.columns:
        batch_bat = batch_rolling(df, 'batter_id', batter_stats, rolling_windows)
        df = pd.concat([df, batch_bat], axis=1)
    if 'pitcher_id' in df.columns:
        batch_pit = batch_rolling(df, 'pitcher_id', pitcher_stats, rolling_windows)
        df = pd.concat([df, batch_pit], axis=1)
        # Pitcher HR rolling
        if 'hr_outcome' in df.columns:
            for w in rolling_windows:
                df[f'P_rolling_hr_rate_{w}'] = df.groupby('pitcher_id')['hr_outcome'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())

    # Pitch type mix/one-hot (efficient)
    pitch_types = ['SL','SI','FC','FF','ST','CH','CU','FS','FO','SV','KC','EP','FA','KN','CS','SC']
    for pt in pitch_types:
        if 'pitch_type' in df.columns:
            df[f'pitch_type_{pt}'] = (df['pitch_type'] == pt).astype(int)
            for w in rolling_windows:
                df[f'B_pitch_pct_{pt}_{w}'] = (
                    df.groupby('batter_id')['pitch_type'].transform(
                        lambda x: x.shift(1).eq(pt).rolling(w, min_periods=1).mean()
                    )
                )
                df[f'P_pitch_pct_{pt}_{w}'] = (
                    df.groupby('pitcher_id')['pitch_type'].transform(
                        lambda x: x.shift(1).eq(pt).rolling(w, min_periods=1).mean()
                    )
                )

    # Park-handed HR rate
    df = compute_park_handed_hr_rate(df)

    # Dummies (cats)
    cat_cols = [
        "stand", "p_throws", "pitch_type", "pitch_name", "bb_type",
        "condition", "roof_status", "handed_matchup"
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('Unknown')
    df = pd.get_dummies(df, columns=[col for col in cat_cols if col in df.columns], drop_first=False)
    return df

# ===== Streamlit UI =====
st.title("⚾ All-in-One MLB HR Analyzer & XGBoost Modeler")

mode = st.radio("Choose Mode:", ['Fetch Data & Export', 'Upload & Analyze/Score'], horizontal=True)

if mode == "Fetch Data & Export":
    # 1. Date picker, source select
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", value=datetime.today())
    src = st.radio("Data Source", ["Fetch Statcast", "Upload CSV"], horizontal=True)
    run_query = st.button("Fetch/Load & Feature Engineer")
    uploaded_event_csv = st.file_uploader("Or: Upload Event-Level CSV", type=["csv"]) if src == "Upload CSV" else None

    if run_query or (uploaded_event_csv is not None):
        with st.status("Loading & Engineering..."):
            if src == "Fetch Statcast" and run_query:
                df = statcast(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            else:
                df = pd.read_csv(uploaded_event_csv)
            st.success(f"Loaded {len(df)} rows.")

            # WEATHER (merge batch for all unique home_team_code + game_date)
            if 'home_team_code' in df.columns and 'game_date' in df.columns:
                df['weather_key'] = df['home_team_code'] + "_" + pd.to_datetime(df['game_date']).dt.strftime("%Y%m%d")
                unique_keys = df['weather_key'].unique()
                for i, key in enumerate(unique_keys):
                    team = key.split('_')[0]
                    city = mlb_team_city_map.get(team, "New York")
                    date = df[df['weather_key'] == key].iloc[0]['game_date']
                    weather = get_weather(city, date.strftime("%Y-%m-%d"))
                    for feat in ['temp', 'wind_mph', 'wind_dir', 'humidity', 'condition']:
                        df.loc[df['weather_key'] == key, feat] = weather[feat]
                st.info("Weather merged.")

            # FULL FEATURE ENGINEERING
            df = feature_engineer(df)

            # Remove problematic duplicate columns if any (caused by pd.get_dummies sometimes)
            df = df.loc[:,~df.columns.duplicated()].copy()

            # Ensure 'hr_outcome' present for next steps
            if 'hr_outcome' not in df.columns:
                st.warning("No 'hr_outcome' column detected. Please make sure this is included for downstream model training.")
            else:
                # LOGISTIC REGRESSION WEIGHTS (auto-calculated for all features)
                logit_model = None
                model_features = [c for c in df.columns if c not in [
                    'game_date', 'batter', 'pitcher', 'description', 'events', 'hr_outcome'
                ] and np.issubdtype(df[c].dtype, np.number) and df[c].nunique() > 1]
                X = df[model_features].fillna(0)
                y = df['hr_outcome'].astype(int)
                if len(df) > 30:
                    logit_model = LogisticRegression(max_iter=1000)
                    logit_model.fit(X, y)
                    logit_weights = pd.DataFrame({
                        'feature': X.columns,
                        'logit_weight': logit_model.coef_[0]
                    }).sort_values('logit_weight', ascending=False)
                    st.markdown("#### Logistic Regression Feature Weights")
                    st.dataframe(logit_weights.head(30))
                    st.download_button("⬇️ Download Logistic Weights CSV", data=logit_weights.to_csv(index=False), file_name="logit_weights.csv")
                else:
                    st.info("Not enough events for meaningful logistic regression.")

            # EVENT-LEVEL CSV DOWNLOAD (always, so user can export for later analysis)
            st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
            st.dataframe(df.head(20))
            st.download_button("⬇️ Download Event-Level CSV", data=df.to_csv(index=False), file_name="event_level_hr_features.csv")
            st.success("Event-level feature engineering complete! You can now use these for analysis/scoring.")

elif mode == "Upload & Analyze/Score":
    st.info("**All three uploads are REQUIRED to start analysis.**")
    uploaded_event = st.file_uploader("Upload Event-Level Features CSV", type=["csv"])
    uploaded_matchups = st.file_uploader("Upload Matchups CSV", type=["csv"])
    uploaded_logit_weights = st.file_uploader("Upload Logistic Weights CSV", type=["csv"])

    ready = (uploaded_event is not None) and (uploaded_matchups is not None) and (uploaded_logit_weights is not None)

    if not ready:
        st.warning("You must upload all three CSV files before analysis can begin.")
    else:
        # --- LOAD ALL UPLOADS ---
        event_df = pd.read_csv(uploaded_event)
        matchups_df = pd.read_csv(uploaded_matchups)
        logit_weights = pd.read_csv(uploaded_logit_weights)

        # --- JOIN event/matchups on game_date, home_team, etc as needed ---
        # (You can customize this join as needed for your format)
        # For most use cases, event_df already has sufficient info

        # --- FULL FEATURE ENGINEERING ---
        df = feature_engineer(event_df)
        df = df.loc[:,~df.columns.duplicated()].copy()

        # --- MODEL FEATURES SYNC ---
        model_features = [f for f in logit_weights['feature'].values if f in df.columns and df[f].nunique() > 1]
        if not model_features:
            st.error("No matching features found between logistic weights and event data.")
        elif 'hr_outcome' not in df.columns:
            st.error("No 'hr_outcome' column found in your event-level upload.")
        else:
            st.success("Analysis ready: Running Logistic Regression and XGBoost models.")

            X = df[model_features].fillna(0)
            y = df['hr_outcome'].astype(int)

            # --- Logistic Regression (using uploaded weights, refit for evaluation) ---
            logit_model = LogisticRegression(max_iter=1000)
            logit_model.fit(X, y)
            y_logit_prob = logit_model.predict_proba(X)[:, 1]
            auc_logit = roc_auc_score(y, y_logit_prob)
            st.info(f"Logistic Regression ROC-AUC: {auc_logit:.3f}")
            st.write(classification_report(y, y_logit_prob > 0.5, digits=3))
            logit_weights_new = pd.DataFrame({
                'feature': X.columns,
                'logit_weight': logit_model.coef_[0]
            }).sort_values('logit_weight', ascending=False)
            st.dataframe(logit_weights_new.head(30))

            # --- XGBoost Classifier ---
            if len(df) > 50:
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
                st.info(f"XGBoost ROC-AUC: {auc:.3f}")
                st.write(classification_report(y_test, y_pred_prob > 0.5, digits=3))
                feature_importance = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
                st.markdown("#### XGBoost Feature Importances")
                st.dataframe(feature_importance.head(30))
                st.download_button(
                    "⬇️ Download XGBoost Model (.pkl)",
                    data=pickle.dumps(xgb_model),
                    file_name="xgboost_hr_model.pkl"
                )
            else:
                st.warning("Not enough complete events to fit XGBoost model.")

            # --- Scored Event Output CSV ---
            df['logit_score'] = y_logit_prob
            if len(df) > 50:
                df['xgb_score'] = xgb_model.predict_proba(X)[:,1]
            st.markdown("#### Download Scored Event-Level CSV")
            st.dataframe(df.head(20))
            st.download_button("⬇️ Download Scored Event-Level CSV", data=df.to_csv(index=False), file_name="scored_event_level_hr_features.csv")

st.success("All steps, maps, and logic included. 🚀")
