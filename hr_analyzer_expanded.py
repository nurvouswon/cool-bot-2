import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
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

# ============== Utility Functions ==============
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

def ensure_hr_outcome(df):
    if 'hr_outcome' not in df.columns:
        def is_hr(val):
            if pd.isnull(val): return 0
            val = str(val).strip().lower().replace("-", " ").replace("_", " ")
            return int("home run" in val or "homerun" in val)
        if 'events' in df.columns:
            df['hr_outcome'] = df['events'].apply(is_hr)
        elif 'description' in df.columns:
            df['hr_outcome'] = df['description'].apply(is_hr)
        else:
            st.warning("No home run indicator found! Add 'hr_outcome', 'events', or 'description' column to your file.")
    return df

def robust_numeric_columns(df):
    # Return only numeric, non-constant, non-object columns
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1 and not pd.api.types.is_object_dtype(df[c])
    ]

# ========== Streamlit UI ==========
st.title("⚾ All-in-One MLB HR Analyzer & XGBoost/Logit Modeler (2025)")

mode = st.radio("Select Mode", ["Fetch Data (Statcast + Weather)", "Analyze (Upload CSVs for Logit/XGBoost)"], horizontal=True)

if mode == "Fetch Data (Statcast + Weather)":
    from pybaseball import statcast

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", value=datetime.today())
    run_query = st.button("Fetch Statcast Data and Run Analyzer")
    df = None
    if run_query:
        progress = st.progress(0, text="Starting...")
        df = statcast(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        progress.progress(5, text="Statcast downloaded")

        # ==== Feature Engineering ====
        # Standardize column names
        df.columns = [c.lower() for c in df.columns]
        # Add HR outcome
        df = ensure_hr_outcome(df)
        progress.progress(10, text="HR outcome mapped")
        
        # Park/Team fields
        if 'home_team' in df.columns and 'home_team_code' not in df.columns:
            df['home_team_code'] = df['home_team']
        if 'home_team_code' in df.columns:
            df['home_team_code'] = df['home_team_code'].astype(str).str.upper()
        if 'park' not in df.columns and 'home_team_code' in df.columns:
            df['park'] = df['home_team_code'].map(team_code_to_park).fillna(df['home_team'].str.lower().str.replace(' ', '_'))
        df['park_hr_rate'] = df['park'].map(park_hr_rate_map).fillna(1.0)
        df['park_altitude'] = df['park'].map(park_altitude_map).fillna(0)
        df['roof_status'] = df['park'].map(roof_status_map).fillna("open")

        # --- Weather Integration ---
        weather_features = ['temp', 'wind_mph', 'wind_dir', 'humidity', 'condition']
        if 'home_team_code' in df.columns and 'game_date' in df.columns:
            df['weather_key'] = df['home_team_code'] + "_" + pd.to_datetime(df['game_date']).dt.strftime("%Y%m%d")
            unique_keys = df['weather_key'].unique()
            for i, key in enumerate(unique_keys):
                team = key.split('_')[0]
                city = mlb_team_city_map.get(team, "New York")
                date = df[df['weather_key'] == key].iloc[0]['game_date']
                weather = get_weather(city, str(date)[:10])
                for feat in weather_features:
                    df.loc[df['weather_key'] == key, feat] = weather[feat]
                progress.progress(10 + int(60 * (i+1) / len(unique_keys)), text=f"Weather {i+1}/{len(unique_keys)}")
            progress.progress(70, text="Weather merged")
        else:
            for feat in weather_features:
                df[feat] = None

        # --- Rolling and Pitch Mix Features (efficient batch) ---
        # Only for batter_id/pitcher_id/game_date present
        ROLL = [3,5,7,14]
        roll_feats = []
        # Make sure index is clean for concat below
        df = df.reset_index(drop=True)
        # For each type of rolling feat, collect as dicts, then concat once
        for id_col, prefix in [('batter', 'B_'), ('pitcher', 'P_')]:
            if id_col+'_id' in df.columns and 'game_date' in df.columns:
                for stat in ['launch_speed','launch_angle','hit_distance_sc','woba_value']:
                    for w in ROLL:
                        colname = f"{prefix}{stat}_{w}"
                        s = (
                            df.sort_values([id_col+'_id','game_date'])
                            .groupby(id_col+'_id')[stat].apply(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                        )
                        roll_feats.append(pd.DataFrame({colname: s}))
                # Additional EV rolls
                if 'launch_speed' in df.columns:
                    for w in ROLL:
                        roll_feats.append(pd.DataFrame({
                            f"{prefix}max_ev_{w}": (
                                df.sort_values([id_col+'_id','game_date'])
                                .groupby(id_col+'_id')['launch_speed'].apply(lambda x: x.shift(1).rolling(w, min_periods=1).max())
                            ),
                            f"{prefix}median_ev_{w}": (
                                df.sort_values([id_col+'_id','game_date'])
                                .groupby(id_col+'_id')['launch_speed'].apply(lambda x: x.shift(1).rolling(w, min_periods=1).median())
                            )
                        }))
        # Efficient concat and align
        if roll_feats:
            roll_df = pd.concat(roll_feats, axis=1)
            roll_df = roll_df.reset_index(drop=True)
            for col in roll_df.columns:
                df[col] = roll_df[col]

        # Pitch type rolling % (efficient)
        pitch_types = ['SL','SI','FC','FF','ST','CH','CU','FS','FO','SV','KC','EP','FA','KN','CS','SC']
        for id_col, prefix in [('batter','B_'),('pitcher','P_')]:
            if id_col+'_id' in df.columns and 'game_date' in df.columns and 'pitch_type' in df.columns:
                for pt in pitch_types:
                    for w in ROLL:
                        colname = f"{prefix}pitch_pct_{pt}_{w}"
                        s = (
                            df.sort_values([id_col+'_id','game_date'])
                            .groupby(id_col+'_id')['pitch_type'].apply(lambda x: x.shift(1).eq(pt).rolling(w, min_periods=1).mean())
                        )
                        df[colname] = s.reset_index(drop=True)

        # Park-handed HR rate
        df = compute_park_handed_hr_rate(df)

        progress.progress(80, text="Feature engineering complete")

        # --- Save Event-level CSV
        event_cols = list(df.columns)
        st.markdown("#### Download Event-Level CSV (all features):")
        st.dataframe(df.head(20))
        st.download_button("⬇️ Download Event-Level CSV", data=df.to_csv(index=False), file_name="event_level_hr_features.csv")

        # --- Compute logistic regression weights (if sufficient HR outcomes)
        if 'hr_outcome' in df.columns and df['hr_outcome'].nunique() > 1:
            model_features = [c for c in robust_numeric_columns(df) if c != 'hr_outcome']
            model_df = df.dropna(subset=model_features + ['hr_outcome'], how='any')
            X = model_df[model_features]
            y = model_df['hr_outcome'].astype(int)
            logit = LogisticRegression(max_iter=200, solver='liblinear')
            logit.fit(X, y)
            weights = pd.DataFrame({'feature': model_features, 'weight': logit.coef_[0]})
            st.markdown("#### Download Logistic Regression Weights")
            st.dataframe(weights.sort_values('weight', ascending=False).head(30))
            st.download_button("⬇️ Download Logit Weights CSV", data=weights.to_csv(index=False), file_name="logit_weights.csv")
        else:
            st.warning("Not enough home run outcome events to fit logistic regression weights.")
        progress.progress(100, text="All done!")

elif mode == "Analyze (Upload CSVs for Logit/XGBoost)":
    st.subheader("Upload ALL THREE required files for analysis:")
    up_event = st.file_uploader("Upload Event-Level CSV (with features and hr_outcome)", type=['csv'])
    up_matchups = st.file_uploader("Upload Daily Lineups / Matchups CSV", type=['csv'])
    up_logit = st.file_uploader("Upload Logistic Weights CSV", type=['csv'])

    analyze_btn = st.button("Run Full Logit/XGBoost Analysis")

    if analyze_btn:
        if not (up_event and up_matchups and up_logit):
            st.error("Please upload ALL THREE required CSVs: event-level, matchups, and logistic weights.")
        else:
            # Load data
            df = pd.read_csv(up_event)
            df_match = pd.read_csv(up_matchups)
            logit_weights = pd.read_csv(up_logit)
            st.success("All files loaded successfully.")

            # Ensure hr_outcome column (auto-detect if missing)
            df = ensure_hr_outcome(df)
            if 'hr_outcome' not in df.columns or df['hr_outcome'].nunique() <= 1:
                st.error("No or only one unique 'hr_outcome' found in event-level CSV. Please check your data.")
                st.stop()

            # Show some matchup integration logic (merge lineups, park, weather, etc.)
            st.write("Sample of Matchup Data (first 10 rows):")
            st.dataframe(df_match.head(10))

            # Merge example: merge by player or by date/team as appropriate for leaderboard creation
            if 'player name' in df_match.columns:
                player_names = set(df_match['player name'].str.lower().str.strip())
                df['in_starting_lineup'] = df['player_name'].str.lower().str.strip().isin(player_names)
            else:
                df['in_starting_lineup'] = False  # Fallback

            # Prepare features for logistic regression scoring
            model_features = list(logit_weights['feature'])
            X = df[model_features].fillna(0)
            # Score using logistic weights
            logit_score = (X * logit_weights.set_index('feature')['weight']).sum(axis=1)
            df['logit_score'] = logit_score

            # Train XGBoost (if enough HR outcomes)
            st.write("Fitting XGBoost Model...")
            from xgboost import XGBClassifier
            Xgb = XGBClassifier(n_estimators=80, max_depth=5, learning_rate=0.2, use_label_encoder=False, eval_metric="auc")
            # Only train on non-null and numeric features
            model_df = df.dropna(subset=model_features + ['hr_outcome'], how='any')
            X_train = model_df[model_features].astype(float)
            y_train = model_df['hr_outcome'].astype(int)
            Xgb.fit(X_train, y_train)
            st.success("XGBoost trained on uploaded event data.")

            # Show feature importances
            importances = pd.DataFrame({'feature': model_features, 'importance': Xgb.feature_importances_})
            st.markdown("#### XGBoost Feature Importances")
            st.dataframe(importances.sort_values("importance", ascending=False).head(30))

            # Predict & display leaderboard for starting lineup
            df['xgb_score'] = Xgb.predict_proba(X[model_features])[:, 1]
            lb = df[df['in_starting_lineup']].sort_values('xgb_score', ascending=False)
            st.markdown("#### Today's Leaderboard (Starting Lineup Only, sorted by XGBoost Score):")
            st.dataframe(lb[['player_name', 'team', 'logit_score', 'xgb_score', 'hr_outcome'] + model_features].head(20))

            # Download results
            st.download_button("⬇️ Download Full Scored Event CSV", data=df.to_csv(index=False), file_name="event_scored.csv")
            # Optional: Save trained XGB model
            st.download_button("⬇️ Download XGBoost Model (Pickle)", data=pickle.dumps(Xgb), file_name="xgb_model.pkl")

            st.success("All analysis complete!")

# ========== End of Script ==========
st.markdown("---")
st.caption("Built with Statcast, advanced weather, park, pitch, and rolling features. For full customization or MLB support, contact your dev team.")
