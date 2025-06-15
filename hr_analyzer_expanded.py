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

# ================== Context Maps ==================
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
    if pd.isna(wind_dir):
        return np.nan
    wind_dir = str(wind_dir).upper()
    for d in directions:
        if d in wind_dir:
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

# ========== Advanced Feature Engineering ==========

def add_batted_ball_event_filter(df):
    # Only keep certain batted ball events
    allowed = ["single", "double", "triple", "home_run", "homerun", "home run", "field_out"]
    def normalize_event(ev):
        return str(ev).strip().lower().replace(" ", "_") if pd.notna(ev) else ""
    allowed_norm = [e.replace(" ", "_") for e in allowed]
    return df[df['events'].apply(lambda ev: normalize_event(ev) in allowed_norm)].reset_index(drop=True)

def add_hr_outcome_column(df):
    hr_labels = ["home_run", "homerun", "home run"]
    def is_hr(ev):
        return str(ev).strip().lower().replace(" ", "_") in hr_labels
    df["hr_outcome"] = df["events"].apply(is_hr).astype(int)
    return df

def create_rolling_features(df):
    # All rolling features at once to avoid fragmentation
    batter_rolls = {}
    pitcher_rolls = {}
    ROLL = [3, 5, 7, 14]
    batter_stats = ['launch_speed','launch_angle','hit_distance_sc','woba_value']
    pitcher_stats = batter_stats.copy()
    for stat in batter_stats:
        if stat in df.columns:
            for w in ROLL:
                batter_rolls[f'B_{stat}_{w}'] = (
                    df.groupby('batter_id')[stat]
                    .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                )
    for stat in pitcher_stats:
        if stat in df.columns:
            for w in ROLL:
                pitcher_rolls[f'P_{stat}_{w}'] = (
                    df.groupby('pitcher_id')[stat]
                    .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                )
    # Pitch type rolling for batter/pitcher
    pitch_types = ['SL','SI','FC','FF','ST','CH','CU','FS','FO','SV','KC','EP','FA','KN','CS','SC']
    for pt in pitch_types:
        if 'pitch_type' in df.columns:
            for w in ROLL:
                batter_rolls[f'B_pitch_pct_{pt}_{w}'] = (
                    df.groupby('batter_id')['pitch_type'].transform(
                        lambda x: x.shift(1).eq(pt).rolling(w, min_periods=1).mean()
                    )
                )
                pitcher_rolls[f'P_pitch_pct_{pt}_{w}'] = (
                    df.groupby('pitcher_id')['pitch_type'].transform(
                        lambda x: x.shift(1).eq(pt).rolling(w, min_periods=1).mean()
                    )
                )
    # Add all at once to prevent fragmentation
    df = pd.concat([df, pd.DataFrame(batter_rolls), pd.DataFrame(pitcher_rolls)], axis=1)
    return df

def robust_numeric_columns(df):
    # Only real numeric columns, not bool or categorical
    nums = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1 and not pd.api.types.is_bool_dtype(df[c]):
            nums.append(c)
    return nums

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
st.set_page_config(page_title="MLB HR Analyzer", layout="wide")
st.title("⚾ All-in-One MLB HR Analyzer & XGBoost Modeler")
st.markdown("""
**Fetch and build event-level feature sets, run advanced modeling, and export all artifacts.**
""")

tab1, tab2 = st.tabs(["Fetch & Feature Engineering", "Analyze & Model"])
with tab1:
    st.header("Fetch Raw Statcast & Engineer Features")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", value=datetime.today())

    run_query = st.button("Fetch Statcast & Engineer Features")
    df = None
    if run_query:
        progress = st.progress(0, text="Fetching Statcast...")
        df = statcast(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        progress.progress(10, text="Filtering batted ball events...")
        df = add_batted_ball_event_filter(df)
        progress.progress(20, text="Adding HR outcome column...")
        df = add_hr_outcome_column(df)
        progress.progress(25, text="Minimal cleaning...")
        # minimal cleaning
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
                df['home_team'].str.lower().str.replace(' ', '_') if 'home_team' in df.columns else ""
            )
        df['park_hr_rate'] = df['park'].map(park_hr_rate_map).fillna(1.0)
        df['park_altitude'] = df['park'].map(park_altitude_map).fillna(0)
        df['roof_status'] = df['park'].map(roof_status_map).fillna("open")
        progress.progress(30, text="Getting weather features...")
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
                pct = 30 + int(30 * (i+1) / len(unique_keys))
                progress.progress(pct, text=f"Weather {i+1}/{len(unique_keys)}")
        else:
            for feat in weather_features:
                df[feat] = None
        progress.progress(60, text="Rolling features and advanced engineering...")
        df = create_rolling_features(df)
        progress.progress(70, text="Park-handed HR rates...")
        df = compute_park_handed_hr_rate(df)
        progress.progress(80, text="Final feature cleaning...")
        # add other engineered features (add yours as needed)
        df['wind_dir_angle'] = df['wind_dir'].apply(wind_dir_to_angle)
        df['wind_dir_sin'] = np.sin(np.deg2rad(df['wind_dir_angle']))
        df['wind_dir_cos'] = np.cos(np.deg2rad(df['wind_dir_angle']))
        batted_ball_flags = [
            'is_barrel', 'is_sweet_spot', 'is_hard_hit', 'flyball', 'line_drive',
            'groundball', 'pull_air', 'pull_side'
        ]
        for col in batted_ball_flags:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
        # interactions
        for col in ['is_barrel', 'is_hard_hit', 'flyball', 'pull_air']:
            if col in df.columns and all(x in df.columns for x in ['humidity', 'temp', 'wind_mph']):
                df[f'{col}_x_humidity'] = df[col] * df['humidity']
                df[f'{col}_x_temp'] = df[col] * df['temp']
                df[f'{col}_x_wind_mph'] = df[col] * df['wind_mph']
        progress.progress(85, text="Removing duplicate columns...")
        df = df.loc[:, ~df.columns.duplicated()]
        progress.progress(90, text="Logistic weights calculation...")
        # ==== Logistic weights ====
        model_features = [c for c in robust_numeric_columns(df) if c != "hr_outcome"]
        model_df = df.dropna(subset=model_features + ['hr_outcome'], how='any')
        logistic_weights = pd.DataFrame()
        auc = None
        if model_df['hr_outcome'].nunique() > 1 and len(model_df) >= 30:
            X = model_df[model_features]
            y = model_df['hr_outcome'].astype(int)
            logit = LogisticRegression(max_iter=200, solver='liblinear')
            logit.fit(X, y)
            logistic_weights = pd.DataFrame({
                'feature': model_features,
                'weight': logit.coef_[0]
            }).sort_values('weight', ascending=False)
            auc = roc_auc_score(y, logit.predict_proba(X)[:, 1])
            st.success(f"Logistic regression weights computed (AUC: {auc:.3f})")
        elif model_df['hr_outcome'].nunique() == 1:
            st.warning("Not enough HR events (at least two classes needed) to compute logistic regression weights.")
        else:
            st.warning("Insufficient data for logistic regression.")
        st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
        st.dataframe(df.head(20))
        st.download_button("Download Event CSV", data=df.to_csv(index=False), file_name="event_level.csv", mime="text/csv")
        if not logistic_weights.empty:
            st.markdown("#### Download Logistic Regression Weights CSV:")
            st.dataframe(logistic_weights)
            st.download_button("Download Logistic Weights", data=logistic_weights.to_csv(index=False), file_name="logit_weights.csv", mime="text/csv")
        st.progress(100, text="Feature engineering complete!")
        st.success(f"Done. {len(df)} events processed.")
        st.session_state["feature_df"] = df
        st.session_state["logistic_weights"] = logistic_weights

with tab2:
    st.header("Analyze: Logistic & XGBoost Model")
    st.markdown("Upload all required artifacts for this date (Event-Level CSV, Matchups CSV, Logistic Weights CSV):")
    upcol1, upcol2, upcol3 = st.columns(3)
    with upcol1:
        event_file = st.file_uploader("Upload Event-Level CSV", type="csv", key="events")
    with upcol2:
        matchup_file = st.file_uploader("Upload Matchups CSV", type="csv", key="matchups")
    with upcol3:
        logit_file = st.file_uploader("Upload Logistic Weights CSV", type="csv", key="logits")

    ready = all([event_file, matchup_file, logit_file])
    if st.button("Analyze and Score (Logistic & XGBoost)", disabled=not ready):
        # Load all data
        df = pd.read_csv(event_file)
        matchups = pd.read_csv(matchup_file)
        logit_weights = pd.read_csv(logit_file)
        # Ensure correct columns, dedupe
        df = df.loc[:, ~df.columns.duplicated()]
        if 'hr_outcome' not in df.columns:
            df = add_hr_outcome_column(df)
        df = add_batted_ball_event_filter(df)
        st.write(f"{len(df)} filtered batted ball events loaded.")
        # Merge with matchups on player id, game_date
        if 'batter_id' in df.columns and 'mlb id' in matchups.columns:
            matchups['mlb id'] = matchups['mlb id'].astype(str)
            df['batter_id'] = df['batter_id'].astype(str)
            df = df.merge(
                matchups.rename(columns={'mlb id': 'batter_id'}),
                how='left', on='batter_id', suffixes=('', '_mu')
            )
        else:
            st.warning("Could not merge matchups, check CSV formats.")
        # -- Logistic regression scoring --
        model_features = list(logit_weights['feature'].values)
        if 'hr_outcome' in df.columns and set(model_features).issubset(df.columns):
            X = df[model_features].fillna(0)
            coefs = logit_weights.set_index('feature')['weight']
            logit_score = (X * coefs).sum(axis=1)
            df['logit_score'] = logit_score
        else:
            st.warning("Logistic features missing from event CSV.")
        # -- XGBoost Model --
        if df['hr_outcome'].nunique() > 1 and len(df) >= 30:
            X_train, X_test, y_train, y_test = train_test_split(
                df[model_features], df['hr_outcome'], test_size=0.2, random_state=42)
            xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=5, n_jobs=-1, eval_metric='logloss', use_label_encoder=False)
            xgb_model.fit(X_train, y_train)
            xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, xgb_preds)
            st.success(f"XGBoost validation AUC: {auc:.3f}")
            # Predict all events
            df['xgb_score'] = xgb_model.predict_proba(df[model_features].fillna(0))[:, 1]
        else:
            st.warning("XGBoost: Not enough HR events for training (need at least 2 classes).")
            df['xgb_score'] = np.nan
        # Show leaderboard (top 50 by XGB, then logit score fallback)
        lb_cols = ['batter_id','player name','batting order','position','game_date','logit_score','xgb_score','hr_outcome']
        leaderboard = df.sort_values(['xgb_score','logit_score'], ascending=False)
        st.markdown("### Home Run Leaderboard")
        st.dataframe(leaderboard[lb_cols].head(50))
        st.download_button("Download Scored Leaderboard", data=leaderboard.to_csv(index=False), file_name="scored_leaderboard.csv", mime="text/csv")
