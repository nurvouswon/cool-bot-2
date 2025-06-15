import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pickle

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

# ========== UTILITY FUNCTIONS ==========
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
    try:
        api_key = st.secrets["weather"]["api_key"]
        url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={date}"
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
    # Return columns that are reliably numeric and not boolean, object, etc.
    return [
        c for c in df.columns
        if (np.issubdtype(df[c].dtype, np.number) or pd.api.types.is_numeric_dtype(df[c]))
        and not pd.api.types.is_bool_dtype(df[c])
        and df[c].nunique() > 1
    ]

# ========== ADVANCED STATCAST PHYSICS ==========

statcast_physics = [
    'launch_speed', 'launch_angle', 'hit_distance_sc', 'plate_x', 'plate_z',
    'release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z',
    'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'release_pos_x', 'release_pos_y', 'release_pos_z'
]
ROLL_WINDOWS = [3, 5, 7, 14]

def add_rolling_features(df):
    batter_id = 'batter_id' if 'batter_id' in df.columns else 'batter'
    pitcher_id = 'pitcher_id' if 'pitcher_id' in df.columns else 'pitcher'
    batter_roll = {}
    pitcher_roll = {}
    for stat in statcast_physics:
        if stat in df.columns:
            for w in ROLL_WINDOWS:
                batter_roll[f'B_{stat}_{w}'] = (
                    df.groupby(batter_id)[stat].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                )
                pitcher_roll[f'P_{stat}_{w}'] = (
                    df.groupby(pitcher_id)[stat].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                )
    # Add to dataframe all at once
    df = pd.concat([df, pd.DataFrame(batter_roll), pd.DataFrame(pitcher_roll)], axis=1)
    return df

def add_pitch_mix_features(df):
    # Create rolling pitch type % for batters and pitchers
    pitch_types = ['SL','SI','FC','FF','ST','CH','CU','FS','FO','SV','KC','EP','FA','KN','CS','SC']
    batter_id = 'batter_id' if 'batter_id' in df.columns else 'batter'
    pitcher_id = 'pitcher_id' if 'pitcher_id' in df.columns else 'pitcher'
    batter_pitch = {}
    pitcher_pitch = {}
    for pt in pitch_types:
        if 'pitch_type' in df.columns:
            for w in ROLL_WINDOWS:
                batter_pitch[f'B_pitch_pct_{pt}_{w}'] = (
                    df.groupby(batter_id)['pitch_type'].transform(
                        lambda x: x.shift(1).eq(pt).rolling(w, min_periods=1).mean()
                    )
                )
                pitcher_pitch[f'P_pitch_pct_{pt}_{w}'] = (
                    df.groupby(pitcher_id)['pitch_type'].transform(
                        lambda x: x.shift(1).eq(pt).rolling(w, min_periods=1).mean()
                    )
                )
    df = pd.concat([df, pd.DataFrame(batter_pitch), pd.DataFrame(pitcher_pitch)], axis=1)
    return df

def compute_park_handed_hr_rate(df):
    # Estimate park-handedness HR rate from event data
    if all(col in df.columns for col in ['stand', 'p_throws', 'events', 'park']):
        df['handed_matchup'] = df['stand'].astype(str) + df['p_throws'].astype(str)
        df['hr_outcome'] = df['events'].astype(str).str.lower().isin(['home_run','homerun','home run']).astype(int)
        grp = df.groupby(['park', 'handed_matchup'])
        rate = grp['hr_outcome'].mean().reset_index().rename(
            columns={'hr_outcome': 'park_handed_hr_rate'}
        )
        df = df.merge(rate, on=['park', 'handed_matchup'], how='left')
    else:
        df['park_handed_hr_rate'] = np.nan
    return df

# ========== APP START ==========
st.set_page_config(page_title="MLB HR Analyzer", layout="wide")
st.title("⚾ All-in-One MLB HR Analyzer & XGBoost Modeler")

tab1, tab2 = st.tabs(["Fetch & Engineer Data", "Upload & Analyze / Model"])

# ========== TAB 1: FETCH & ENGINEER ==========
with tab1:
    st.header("Step 1: Data Source and Feature Engineering")
    st.info("This step is for fetching raw Statcast CSV data, engineering features, and downloading a ready-to-analyze event-level CSV with 'hr_outcome'.")
    # User uploads or fetches data
    uploaded_event_csv = st.file_uploader("Upload Statcast CSV (if not fetching)", type=["csv"])
    start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=7))
    end_date = st.date_input("End Date", value=datetime.today())

    fetch_button = st.button("Fetch Statcast Data and Run Engineering", type="primary")
    progress = st.progress(0, text="Ready")

    df = None
    if fetch_button or uploaded_event_csv is not None:
        if fetch_button:
            from pybaseball import statcast
            df = statcast(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            progress.progress(10, text="Statcast downloaded")
        else:
            df = pd.read_csv(uploaded_event_csv)
            progress.progress(10, text="CSV loaded")

        # Standardize columns
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # ========== FILTER TO KEY EVENTS ==========
        allowed_events = ['single', 'double', 'triple', 'home_run', 'homerun', 'home run', 'field_out']
        if 'events' in df.columns:
            df = df[df['events'].astype(str).str.lower().isin(allowed_events)].copy()
        progress.progress(20, text="Filtered to batted ball events")

        # ========== LABEL HR OUTCOME ==========
        if 'events' in df.columns:
            df['hr_outcome'] = df['events'].astype(str).str.lower().isin(['home_run', 'homerun', 'home run']).astype(int)
        progress.progress(30, text="Labeled home runs")

        # ========== PARK/TEAM MAPPING ==========
        if 'home_team_code' in df.columns:
            df['home_team_code'] = df['home_team_code'].astype(str).str.upper()

        # Step 1: Try to create 'park' from home_team_code if not present
        if 'park' not in df.columns and 'home_team_code' in df.columns:
            df['park'] = df['home_team_code'].map(team_code_to_park)

        # Step 2: If 'park' still has missing values and home_team is present, fill those
        if 'park' in df.columns and df['park'].isnull().any() and 'home_team' in df.columns:
            df['park'] = df['park'].fillna(df['home_team'].str.lower().str.replace(' ', '_'))

        # Step 3: If 'park' still not in columns but home_team is, create from home_team
        if 'park' not in df.columns and 'home_team' in df.columns:
            df['park'] = df['home_team'].str.lower().str.replace(' ', '_')

        # Step 4: Final check for 'park'
        if 'park' not in df.columns:
            st.error("Could not determine ballpark from your data (missing 'park', 'home_team_code', and 'home_team').")
            st.stop()

        # Add park context maps (always present from here)
        df['park_hr_rate'] = df['park'].map(park_hr_rate_map).fillna(1.0)
        df['park_altitude'] = df['park'].map(park_altitude_map).fillna(0)
        df['roof_status'] = df['park'].map(roof_status_map).fillna("open")
        progress.progress(40, text="Park/team context merged")

        # ========== WEATHER MERGE ==========
        weather_features = ['temp', 'wind_mph', 'wind_dir', 'humidity', 'condition']
        if 'home_team_code' in df.columns and 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
            df['weather_key'] = df['home_team_code'] + "_" + df['game_date'].dt.strftime("%Y%m%d")
            unique_keys = df['weather_key'].unique()
            for i, key in enumerate(unique_keys):
                team = key.split('_')[0]
                city = mlb_team_city_map.get(team, "New York")
                date = df[df['weather_key'] == key].iloc[0]['game_date'].strftime("%Y-%m-%d")
                weather = get_weather(city, date)
                for feat in weather_features:
                    df.loc[df['weather_key'] == key, feat] = weather[feat]
                percent = 40 + int(15 * (i+1) / len(unique_keys))
                progress.progress(percent, text=f"Weather {i+1}/{len(unique_keys)}")
            progress.progress(55, text="Weather merged")
        else:
            for feat in weather_features:
                df[feat] = None

        # ========== WIND ENCODING ==========
        df['wind_dir_angle'] = df['wind_dir'].apply(wind_dir_to_angle)
        df['wind_dir_sin'] = np.sin(np.deg2rad(df['wind_dir_angle']))
        df['wind_dir_cos'] = np.cos(np.deg2rad(df['wind_dir_angle']))

        # ========== ADVANCED ROLLING FEATURES ==========
        df = add_rolling_features(df)
        progress.progress(65, text="Advanced rolling features done")
        df = add_pitch_mix_features(df)
        progress.progress(75, text="Pitch mix features added")

        # ========== PARK-HANDED HR RATE ==========
        df = compute_park_handed_hr_rate(df)
        progress.progress(80, text="Park-handed HR rate done")

        # ========== CLEANUP ==========
        # More binary/categorical cleanup
        batted_ball_flags = ['is_barrel','is_hard_hit','is_sweet_spot','flyball','pull_air']
        for col in batted_ball_flags:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
        if 'stand' in df.columns:
            df['stand_L'] = (df['stand'].astype(str).str.upper() == "L").astype(int)
            df['stand_R'] = (df['stand'].astype(str).str.upper() == "R").astype(int)
        if 'p_throws' in df.columns:
            df['p_throws_L'] = (df['p_throws'].astype(str).str.upper() == "L").astype(int)
            df['p_throws_R'] = (df['p_throws'].astype(str).str.upper() == "R").astype(int)

        # ========== LOGISTIC REGRESSION WEIGHTS ==========
        # Only run if enough HR/non-HR events present
        logit_weights = pd.DataFrame()
        model_features = [c for c in robust_numeric_columns(df) if c not in ['hr_outcome', 'batter', 'pitcher', 'game_date', 'batter_id', 'pitcher_id']]
        if 'hr_outcome' in df.columns and df['hr_outcome'].nunique() == 2:
            model_df = df.dropna(subset=model_features + ['hr_outcome'], how='any')
            X = model_df[model_features]
            y = model_df['hr_outcome'].astype(int)
            if y.nunique() == 2:
                logit = LogisticRegression(max_iter=200, solver='liblinear')
                logit.fit(X, y)
                weights = pd.DataFrame({'feature': model_features, 'weight': logit.coef_[0]})
                logit_weights = weights.sort_values('weight', ascending=False)
                st.markdown("#### Download Logistic Regression Weights")
                st.dataframe(logit_weights)
                st.download_button(
                    "Download Logistic Weights CSV",
                    data=logit_weights.to_csv(index=False),
                    file_name="logistic_weights.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Not enough HR events (at least two classes needed) to compute logistic regression weights.")
        else:
            st.warning("Not enough HR events (at least two classes needed) to compute logistic regression weights.")

        # ========== EVENT-LEVEL CSV DOWNLOAD ==========
        st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
        event_df = df.reset_index(drop=True)
        st.dataframe(event_df.head(20))
        st.download_button(
            "Download Event-Level CSV",
            data=event_df.to_csv(index=False),
            file_name="event_level_features.csv",
            mime="text/csv"
        )
        progress.progress(100, text="Done!")

# ========== TAB 2: UPLOAD & ANALYZE ==========
with tab2:
    st.header("Upload Engineered Data & Model Analysis")
    st.info(
        "Upload all three: **Event-level CSV, Matchups CSV, and Logistic Weights CSV**. "
        "All are required for analysis, scoring, and leaderboard. Only engineered CSVs with `hr_outcome` will work."
    )
    uploaded_event = st.file_uploader("Upload Event-Level CSV (must have 'hr_outcome')", type=['csv'], key="ev2")
    uploaded_matchups = st.file_uploader("Upload Daily Matchups/Lineups CSV", type=['csv'], key="mu2")
    uploaded_logit = st.file_uploader("Upload Logistic Weights CSV", type=['csv'], key="lw2")

    can_run_analysis = (uploaded_event is not None) and (uploaded_matchups is not None) and (uploaded_logit is not None)
    analyze_button = st.button("Run Analysis & Leaderboard", disabled=not can_run_analysis)
    if not can_run_analysis:
        st.warning("All three uploads are required.")

    if can_run_analysis and analyze_button:
        # Progress bar
        analysis_prog = st.progress(0, text="Loading data")
        df = pd.read_csv(uploaded_event)
        matchups = pd.read_csv(uploaded_matchups)
        logit_weights = pd.read_csv(uploaded_logit)
        analysis_prog.progress(10, text="Cleaning columns")

        # Ensure all columns are properly typed
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        matchups.columns = [c.strip().lower().replace(" ", "_") for c in matchups.columns]
        if 'hr_outcome' not in df.columns:
            st.error("No 'hr_outcome' column detected! Upload event-level CSV with 'hr_outcome' included.")
            st.stop()

        # Merge matchup info (e.g., for leaderboard), using mlb id if available
        match_id_col = 'mlb_id' if 'mlb_id' in matchups.columns else 'player_name'
        df['match_key'] = df['batter_id'].astype(str) if 'batter_id' in df.columns else df['batter'].astype(str)
        matchups['match_key'] = matchups[match_id_col].astype(str)
        merged = df.merge(matchups, left_on='match_key', right_on='match_key', suffixes=['','_mu'], how='inner')
        analysis_prog.progress(30, text="Merged matchups")

        # Score features using uploaded logistic weights
        feature_cols = [f for f in logit_weights['feature'] if f in merged.columns]
        X = merged[feature_cols].fillna(0)
        w = logit_weights.set_index('feature')['weight']
        merged['logit_score'] = X.dot(w).values
        analysis_prog.progress(60, text="Logistic scoring done")

        # Fit/finalize XGBoost (if enough data for two classes)
        if merged['hr_outcome'].nunique() == 2:
            Xgb = xgb.XGBClassifier(
                n_estimators=40, max_depth=4, learning_rate=0.18, subsample=0.6,
                use_label_encoder=False, eval_metric='logloss'
            )
            Xgb.fit(X, merged['hr_outcome'])
            merged['xgb_score'] = Xgb.predict_proba(X)[:,1]
            analysis_prog.progress(80, text="XGBoost model fit")
        else:
            merged['xgb_score'] = np.nan
            st.warning("Not enough HR events for XGBoost (at least two classes needed). Only logistic scores will be shown.")

        # Leaderboard output
        leaderboard = merged[['player_name','batting_order','position','team_code','logit_score','xgb_score','hr_outcome']]
        leaderboard = leaderboard.sort_values('logit_score', ascending=False).reset_index(drop=True)
        st.markdown("### HR Prediction Leaderboard")
        st.dataframe(leaderboard.head(30), use_container_width=True)

        # Download leaderboard
        st.download_button(
            "Download Leaderboard CSV",
            data=leaderboard.to_csv(index=False),
            file_name="hr_leaderboard.csv",
            mime="text/csv"
        )
        analysis_prog.progress(100, text="Done!")

st.caption("MLB HR Analyzer v2025 — All Rights Reserved")
