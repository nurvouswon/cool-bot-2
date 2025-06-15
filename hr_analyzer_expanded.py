import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybaseball import statcast
import requests
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
import pickle

# ======= Context Maps =======
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

# ====== Utility Functions ======
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

def robust_numeric_columns(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]

def add_hr_outcome(df):
    # Recognize home run using statcast event text
    if 'hr_outcome' in df.columns:
        return df
    possible_cols = ['events', 'description']
    for col in possible_cols:
        if col in df.columns:
            df['hr_outcome'] = df[col].astype(str).str.lower().apply(
                lambda x: 1 if ('home_run' in x or 'homerun' in x or 'home run' in x) else 0
            )
            if df['hr_outcome'].sum() > 0:
                return df
    df['hr_outcome'] = 0
    return df

def safe_dataframe(df, n=20):
    try:
        if df.shape[1] != len(set(df.columns)):
            st.warning("Duplicate columns detected in event data. Removing duplicates for display.")
            df = df.loc[:, ~df.columns.duplicated()]
        st.dataframe(df.head(n))
    except Exception as e:
        st.error(f"Error displaying dataframe: {e}")

# ====== Streamlit UI and Pipeline ======
st.set_page_config(page_title="⚾ MLB HR Analyzer", layout="wide")
st.title("⚾ All-in-One MLB HR Analyzer & Modeler")
st.markdown("""
Uploads, fetches and builds full Statcast + Weather + Park + Pitch Mix + Categorical + Advanced interaction features,
and allows **training and using Logistic Regression and XGBoost for HR event prediction with leaderboard!**
""")

tab1, tab2 = st.tabs(["1️⃣ Fetch & Feature Engineer", "2️⃣ Upload & Analyze"])

with tab1:
    st.header("Fetch Statcast, Add Weather, Engineer All Features")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", value=datetime.today())
    run_query = st.button("Fetch Statcast Data and Run Feature Engineering")
    if run_query:
        st.info("Fetching Statcast data and running all feature engineering (this may take several minutes)...")
        progress = st.progress(0, text="Starting...")
        df = statcast(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        progress.progress(10, text="Statcast downloaded")

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
        progress.progress(18, text="Park & context added")

        # Weather API Integration (event-level)
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
                pct = 18 + int(40 * (i+1) / max(1, len(unique_keys)))
                progress.progress(min(pct, 60), text=f"Weather {i+1}/{len(unique_keys)}")
            progress.progress(60, text="Weather merged")
        else:
            for feat in weather_features:
                df[feat] = None

        # Wind Direction Encoding
        df['wind_dir_angle'] = df['wind_dir'].apply(wind_dir_to_angle)
        df['wind_dir_sin'] = np.sin(np.deg2rad(df['wind_dir_angle']))
        df['wind_dir_cos'] = np.cos(np.deg2rad(df['wind_dir_angle']))
        progress.progress(62, text="Wind direction encoded")

        # Feature Engineering
        df = add_hr_outcome(df)
        progress.progress(65, text="HR outcome column engineered")

        # === Advanced Rolling Feature Engineering ===
        ROLL = [3, 5, 7, 14]
        batter_features = {}
        pitcher_features = {}

        batter_stats = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value']
        for stat in batter_stats:
            if stat in df.columns:
                for w in ROLL:
                    batter_features[f'B_{stat}_{w}'] = df.groupby('batter_id')[stat].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                    batter_features[f'B_max_ev_{w}'] = df.groupby('batter_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
                    batter_features[f'B_median_ev_{w}'] = df.groupby('batter_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).median())
        pitcher_stats = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value']
        for stat in pitcher_stats:
            if stat in df.columns:
                for w in ROLL:
                    pitcher_features[f'P_{stat}_{w}'] = df.groupby('pitcher_id')[stat].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                    pitcher_features[f'P_max_ev_{w}'] = df.groupby('pitcher_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
                    pitcher_features[f'P_median_ev_{w}'] = df.groupby('pitcher_id')['launch_speed'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).median())
                    if 'hr_outcome' in df.columns:
                        pitcher_features[f'P_rolling_hr_rate_{w}'] = df.groupby('pitcher_id')['hr_outcome'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())

        # Advanced Statcast physics features (example: add more as needed)
        physics_cols = ['release_spin_rate', 'spin_axis', 'spin_dir', 'attack_angle', 'attack_direction', 'bat_speed']
        for col in physics_cols:
            if col in df.columns:
                for w in ROLL:
                    pitcher_features[f'P_{col}_{w}'] = df.groupby('pitcher_id')[col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                    batter_features[f'B_{col}_{w}'] = df.groupby('batter_id')[col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())

        pitch_types = ['SL','SI','FC','FF','ST','CH','CU','FS','FO','SV','KC','EP','FA','KN','CS','SC']
        for pt in pitch_types:
            if 'pitch_type' in df.columns:
                for w in ROLL:
                    batter_features[f'B_pitch_pct_{pt}_{w}'] = df.groupby('batter_id')['pitch_type'].transform(lambda x: x.shift(1).eq(pt).rolling(w, min_periods=1).mean())
                    pitcher_features[f'P_pitch_pct_{pt}_{w}'] = df.groupby('pitcher_id')['pitch_type'].transform(lambda x: x.shift(1).eq(pt).rolling(w, min_periods=1).mean())

        # Combine all new features at once to avoid DataFrame fragmentation
        df = pd.concat([df, pd.DataFrame(batter_features, index=df.index), pd.DataFrame(pitcher_features, index=df.index)], axis=1)
        progress.progress(85, text="All rolling/statcast features merged")

        # Categorical dummies
        cat_cols = ["stand", "p_throws", "pitch_type", "pitch_name", "bb_type", "condition", "roof_status", "handed_matchup"]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('Unknown')
        df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns if any
        df = pd.get_dummies(df, columns=[col for col in cat_cols if col in df.columns], drop_first=True)
        progress.progress(90, text="Categorical features one-hot encoded")

        # Final clean-up: ensure no duplicate columns, index reset
        df = df.loc[:, ~df.columns.duplicated()]
        df.reset_index(drop=True, inplace=True)
        progress.progress(92, text="Clean-up & deduplication")

        # Save final event-level CSV
        st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
        event_csv = df.to_csv(index=False).encode()
        st.download_button("Download Event-Level CSV", event_csv, file_name="event_level_features.csv", mime="text/csv")

        # Show preview
        st.write(f"Weather merged<br>Loaded {len(df)} events.", unsafe_allow_html=True)
        safe_dataframe(df, n=25)
        progress.progress(100, text="Feature engineering complete!")

        # ===== Logistic Regression Weights Computation and Download =====
        # Only run if sufficient HRs are present
        if 'hr_outcome' in df.columns and df['hr_outcome'].sum() >= 5:
            model_features = [c for c in robust_numeric_columns(df) if c != 'hr_outcome']
            model_df = df.dropna(subset=model_features + ['hr_outcome'], how='any')
            X = model_df[model_features]
            y = model_df['hr_outcome'].astype(int)
            if y.nunique() > 1:  # Make sure there are both classes
                logit = LogisticRegression(max_iter=200, solver='liblinear')
                logit.fit(X, y)
                weights = pd.DataFrame({'feature': model_features, 'weight': logit.coef_[0]})
                st.markdown("#### Download Logistic Regression Weights")
                logit_csv = weights.to_csv(index=False).encode()
                st.download_button("Download Logistic Weights CSV", logit_csv, file_name="logistic_weights.csv", mime="text/csv")
                st.dataframe(weights.sort_values('weight', ascending=False).head(25))
            else:
                st.info("Not enough HR events (at least two classes needed) to compute logistic regression weights.")
        else:
            st.info("Not enough HR events to compute logistic regression weights (need at least 5).")
            
with tab2:
    st.header("Upload & Analyze — Model Fitting and Leaderboard")
    st.markdown("""
    - **All three files are now required:**
      - Event-level feature CSV
      - Daily Matchups CSV (starting lineups)
      - Logistic Regression Weights CSV
    """)
    event_csv = st.file_uploader("Upload Event-Level Features CSV", type=['csv'], key="events_upload")
    matchup_csv = st.file_uploader("Upload Daily Matchups CSV", type=['csv'], key="matchups_upload")
    logit_csv = st.file_uploader("Upload Logistic Weights CSV", type=['csv'], key="logit_upload")
    run_model = st.button("Run Logistic + XGBoost Analysis and Leaderboard")

    if run_model:
        missing = []
        if not event_csv: missing.append("Event-Level Features")
        if not matchup_csv: missing.append("Daily Matchups")
        if not logit_csv: missing.append("Logistic Weights")
        if missing:
            st.error("Please upload: " + ", ".join(missing))
        else:
            st.success("All files uploaded. Processing...")

            event_df = pd.read_csv(event_csv, low_memory=False)
            matchups_df = pd.read_csv(matchup_csv)
            logit_weights = pd.read_csv(logit_csv)
            st.write(f"Loaded {len(event_df)} event-level rows, {len(matchups_df)} daily matchups.")

            # Safety check: dedupe columns
            event_df = event_df.loc[:, ~event_df.columns.duplicated()]
            # Ensure hr_outcome column exists
            if 'hr_outcome' not in event_df.columns:
                event_df = add_hr_outcome(event_df)

            # --- Filter for today's lineup players ---
            # Merge on player (batter) id or name
            id_cols = ['batter_id', 'mlb id', 'player name', 'batter', 'player_name']
            found_id = None
            for c in id_cols:
                if c in event_df.columns and c in matchups_df.columns:
                    found_id = c
                    break
            if not found_id:
                st.warning("Could not match lineup/batter IDs between event-level data and matchups.")
                leader_df = event_df.copy()
            else:
                if "mlb id" in matchups_df.columns:
                    matchups_df["mlb id"] = matchups_df["mlb id"].astype(str)
                event_df[found_id] = event_df[found_id].astype(str)
                lineup_ids = matchups_df["mlb id"].astype(str).unique()
                leader_df = event_df[event_df[found_id].isin(lineup_ids)].copy()
                st.info(f"Filtered to {len(leader_df)} events for today's starting lineup.")

            # --- Apply logistic regression scoring ---
            model_features = [f for f in logit_weights['feature'] if f in leader_df.columns]
            if not model_features:
                st.error("No matching model features between logistic weights and uploaded event-level CSV.")
            else:
                X = leader_df[model_features].fillna(0)
                # Score: dot product
                leader_df['logit_score'] = np.dot(X.values, logit_weights.set_index('feature').loc[model_features, 'weight'].values)
                leader_df['logit_prob'] = 1 / (1 + np.exp(-leader_df['logit_score']))

                # --- Fit XGBoost on events with outcome ---
                if 'hr_outcome' in leader_df.columns and leader_df['hr_outcome'].sum() > 3:
                    model_df = leader_df.dropna(subset=model_features + ['hr_outcome'])
                    X_train, X_test, y_train, y_test = train_test_split(
                        model_df[model_features], model_df['hr_outcome'], test_size=0.3, random_state=42, stratify=model_df['hr_outcome']
                    )
                    xgb_model = xgb.XGBClassifier(n_estimators=60, max_depth=3, learning_rate=0.11, use_label_encoder=False, eval_metric='logloss')
                    xgb_model.fit(X_train, y_train)
                    leader_df['xgb_prob'] = xgb_model.predict_proba(X.fillna(0))[:,1]
                    auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1])
                    st.success(f"XGBoost ROC AUC: {auc:.3f}")
                else:
                    leader_df['xgb_prob'] = np.nan
                    st.warning("Not enough HR events for XGBoost training (at least 4 needed). Only logistic scores shown.")

                # --- Leaderboard ---
                st.markdown("### Home Run Leaderboard")
                sort_cols = []
                if 'xgb_prob' in leader_df.columns:
                    sort_cols.append('xgb_prob')
                if 'logit_prob' in leader_df.columns:
                    sort_cols.append('logit_prob')
                if not sort_cols:
                    st.warning("No model probabilities available for leaderboard!")
                else:
                    display_cols = ['batter_id', 'player_name', 'team', 'batting_order', 'logit_prob', 'xgb_prob', 'hr_outcome']
                    display_cols = [c for c in display_cols if c in leader_df.columns]
                    safe_dataframe(leader_df.sort_values(sort_cols, ascending=False)[display_cols], n=25)
                    csv = leader_df.sort_values(sort_cols, ascending=False)[display_cols].to_csv(index=False).encode()
                    st.download_button("Download Leaderboard CSV", csv, file_name="hr_leaderboard.csv", mime="text/csv")
