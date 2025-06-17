import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybaseball import statcast
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
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

# ========== UTILITY FUNCTIONS ==========

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

# Advanced wind relative angle for pull/oppo
def relative_wind_angle(row):
    try:
        if row['stand'] == 'L':
            return (row['wind_dir_angle'] - 45) % 360  # Approximate pull for L
        else:
            return (row['wind_dir_angle'] - 135) % 360 # Approximate pull for R
    except Exception:
        return np.nan

# ========== APP MAIN UI ==========

st.set_page_config("MLB HR Analyzer", layout="wide")
st.title("⚾ All-in-One MLB HR Analyzer & XGBoost Modeler")
st.caption("Statcast + Physics + Weather + Park + Hand Splits + Pitch Type + Contextual Factors")

tab1, tab2 = st.tabs(["1️⃣ Fetch & Feature Engineer Data", "2️⃣ Upload & Analyze"])

with tab1:
    st.header("Fetch Statcast Data & Generate Features")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", value=datetime.today())

    fetch_btn = st.button("Fetch Statcast, Feature Engineer, and Download", type="primary")
    progress = st.empty()

    # --- Robust rolling pitch type function (fixes MultiIndex error) ---
    def rolling_pitch_type_hr(df, id_col, pitch_col, window):
        df_temp = df.reset_index(drop=False)
        roll_result = (
            df_temp.groupby([id_col, pitch_col])['hr_outcome']
            .apply(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            .reset_index()
            .rename(columns={'hr_outcome': f'rolling_hr_{window}'})
        )
        merged = pd.merge(df_temp, roll_result, how='left', on=['index', id_col, pitch_col])
        merged = merged.sort_values('index')
        return merged[f'rolling_hr_{window}'].values

    if fetch_btn:
        progress.progress(5, "Fetching Statcast data...")
        df = statcast(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        progress.progress(10, "Loaded Statcast")
        st.write(f"Loaded {len(df)} raw Statcast events.")
        if len(df) == 0:
            st.error("No data! Try different dates.")
            st.stop()

        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])

        # --- Filter events (expanded valid events) ---
        valid_events = [
            'single', 'double', 'triple', 'homerun', 'home_run', 'field_out',
            'force_out', 'grounded_into_double_play', 'fielders_choice_out',
            'pop_out', 'lineout', 'flyout', 'sac_fly', 'sac_fly_double_play'
        ]
        if 'events' in df.columns:
            df = df[df['events'].str.lower().str.replace(' ', '').isin(valid_events)].copy()

        # --- HR outcome ---
        if 'hr_outcome' not in df.columns:
            if 'events' in df.columns:
                df['hr_outcome'] = df['events'].astype(str).str.lower().str.replace(' ', '').isin(['homerun', 'home_run']).astype(int)
            else:
                df['hr_outcome'] = np.nan

        # --- Park/team mapping ---
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
        progress.progress(20, "Park/team context merged")

        # --- Weather ---
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

        # --- Advanced Statcast metrics ---
        if 'barrel' in df.columns:
            df['barrel_rate_20'] = df.groupby('batter_id')['barrel'].transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
        if 'launch_speed' in df.columns:
            if 'batter_id' in df.columns:
                df['hard_hit_rate_20'] = df.groupby('batter_id')['launch_speed'].transform(lambda x: (x.shift(1) >= 95).rolling(20, min_periods=5).mean())
        if 'launch_angle' in df.columns:
            if 'batter_id' in df.columns:
                df['sweet_spot_rate_20'] = df.groupby('batter_id')['launch_angle'].transform(lambda x: x.shift(1).between(8, 32).rolling(20, min_periods=5).mean())

        # --- Directional wind context ---
        if 'stand' in df.columns and 'wind_dir_angle' in df.columns:
            df['relative_wind_angle'] = df.apply(relative_wind_angle, axis=1)
            df['relative_wind_sin'] = np.sin(np.deg2rad(df['relative_wind_angle']))
            df['relative_wind_cos'] = np.cos(np.deg2rad(df['relative_wind_angle']))

        # --- Rolling park/handedness HR rates (from your data) ---
        if 'park' in df.columns and 'stand' in df.columns:
            for w in [7, 14, 30]:
                col_name = f'park_hand_HR_{w}'
                df[col_name] = df.groupby(['park', 'stand'])['hr_outcome'].transform(lambda x: x.shift(1).rolling(w, min_periods=5).mean())

        # --- Rolling advanced splits, physics, pitch type, hand splits ---
        roll_windows = [3, 5, 7, 14]
        batter_cols = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value',
                       'release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z']
        pitcher_cols = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value',
                        'release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z']
        if 'batter_id' not in df.columns and 'batter' in df.columns:
            df['batter_id'] = df['batter']
        if 'pitcher_id' not in df.columns and 'pitcher' in df.columns:
            df['pitcher_id'] = df['pitcher']

        batter_feat_dict = {}
        pitcher_feat_dict = {}
        for col in batter_cols:
            if col in df.columns:
                for w in roll_windows:
                    cname = f'B_{col}_{w}'
                    batter_feat_dict[cname] = df.groupby('batter_id')[col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        for col in pitcher_cols:
            if col in df.columns:
                for w in roll_windows:
                    cname = f'P_{col}_{w}'
                    pitcher_feat_dict[cname] = df.groupby('pitcher_id')[col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())

        # Hand matchup splits (batter vs pitcher hand HR rolling mean)
        if 'stand' in df.columns and 'p_throws' in df.columns:
            for w in roll_windows:
                df[f'B_vsP_hand_HR_{w}'] = df.groupby(['batter_id', 'p_throws'])['hr_outcome'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                df[f'P_vsB_hand_HR_{w}'] = df.groupby(['pitcher_id', 'stand'])['hr_outcome'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())

        # --- Rolling pitch type splits (robust fix) ---
        if 'pitch_type' in df.columns:
            for w in roll_windows:
                df[f'B_pitchtype_HR_{w}'] = rolling_pitch_type_hr(df, 'batter_id', 'pitch_type', w)
                df[f'P_pitchtype_HR_{w}'] = rolling_pitch_type_hr(df, 'pitcher_id', 'pitch_type', w)

        # Combine features
        df = pd.concat([df, pd.DataFrame(batter_feat_dict), pd.DataFrame(pitcher_feat_dict)], axis=1)
        df = df.copy()
        progress.progress(80, "Advanced Statcast & context features done")

        # Remove duplicate columns before output
        df = dedup_columns(df)

        # ========== DOWNLOAD ==========
        st.success(f"Feature engineering complete! {len(df)} batted ball events.")
        st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
        st.dataframe(df.head(20))
        st.download_button("⬇️ Download Event-Level CSV", data=df.to_csv(index=False), file_name="event_level_hr_features.csv")
        progress.empty()

        # === LOGISTIC WEIGHTS DOWNLOAD (train logit on this window for quick weights export) ===
        if 'hr_outcome' in df.columns and df['hr_outcome'].nunique() > 1:
            model_features = robust_numeric_columns(df)
            cat_context = [c for c in [
                'park_hr_rate', 'park_altitude', 'temp', 'humidity', 'wind_mph', 
                'wind_dir_angle', 'wind_dir_sin', 'wind_dir_cos', 'relative_wind_angle', 
                'relative_wind_sin', 'relative_wind_cos'
            ] if c in df.columns]
            for c in df.columns:
                if c.startswith("park_hand_HR_"):
                    cat_context.append(c)
            model_features = [c for c in model_features + cat_context if c != 'hr_outcome']

            # Only drop NAs on model features, not all columns!
            feature_na_fracs = {c: df[c].isna().mean() for c in model_features}
            model_features = [c for c in model_features if feature_na_fracs[c] < 0.3]
            model_df = df.dropna(subset=model_features + ['hr_outcome'], how='any')
            if model_df['hr_outcome'].nunique() < 2 or len(model_df) < 30:
                st.warning("Not enough HR/non-HR events for logistic regression weights (check date range or missing features).")
            else:
                X = model_df[model_features].fillna(0)
                y = model_df['hr_outcome'].astype(int)
                # Quick grid search for regularization
                grid = GridSearchCV(LogisticRegression(max_iter=200, solver='liblinear'), param_grid={'C': [0.1, 1, 10]}, cv=3, scoring='roc_auc')
                grid.fit(X, y)
                logit = grid.best_estimator_
                weights = pd.DataFrame({
                    'feature': model_features,
                    'weight': logit.coef_[0],
                    'intercept': logit.intercept_[0]
                })
                st.dataframe(weights.sort_values('weight', ascending=False))
                st.download_button("⬇️ Download Logistic Weights CSV", data=weights.to_csv(index=False), file_name="logit_weights.csv")



# ========== TAB 2 ==========
with tab2:
    st.header("Upload Event, Matchup, and Logistic Weights to Analyze & Score")
    st.markdown("All 3 uploads required! CSVs must match feature sets generated from Tab 1.")

    # Set threshold automatically (0.5 is typical, you can change here)
    threshold = 0.5

    uploaded_events = st.file_uploader("Upload Event-Level Features CSV", type="csv", key="evup")
    uploaded_matchups = st.file_uploader("Upload Matchups CSV", type="csv", key="mup")
    uploaded_logit = st.file_uploader("Upload Logistic Weights CSV", type="csv", key="lup")
    analyze_btn = st.button("Run Analysis (Logit + XGBoost Leaderboard)", type="primary")

    if analyze_btn:
        if not uploaded_events or not uploaded_matchups or not uploaded_logit:
            st.warning("Please upload event-level, matchup, and logistic weights CSVs before running analysis.")
            st.stop()

        event_df = pd.read_csv(uploaded_events)
        matchups = pd.read_csv(uploaded_matchups)
        logit_weights = pd.read_csv(uploaded_logit)

        st.write(f"Loaded {len(event_df)} events, {len(matchups)} matchup rows, {len(logit_weights)} logistic weights.")

        # --- Merge player names and batting order from matchups ---
        batter_id_col = 'batter_id' if 'batter_id' in event_df.columns else 'batter'
        matchup_batter_col = 'mlb id' if 'mlb id' in matchups.columns else 'player id'
        matchup_name_col = 'player name' if 'player name' in matchups.columns else 'name'
        matchup_bo_col = 'batting order' if 'batting order' in matchups.columns else 'batting_order'
        merge_df = event_df.merge(
            matchups[[matchup_batter_col, matchup_name_col, matchup_bo_col, 'position' if 'position' in matchups.columns else matchup_bo_col]],
            left_on=batter_id_col,
            right_on=matchup_batter_col,
            how='left',
            suffixes=('', '_mu')
        )

        # Ensure proper hitter selection: Batting order 1-9 and not a pitcher (filter out 'SP' etc)
        def is_hitter(row):
            try:
                bo = str(row[matchup_bo_col]).strip().upper()
                pos = str(row.get('position', '')).strip().upper()
                return bo.isdigit() and (1 <= int(bo) <= 9) and (pos not in ['SP', 'P', 'RP', 'LHP', 'RHP'])
            except Exception:
                return False

        hitters_df = merge_df[merge_df.apply(is_hitter, axis=1)].copy()
        if hitters_df.empty:
            st.error("No hitter data available for leaderboard! Check your matchups or event CSVs for matching batter IDs and filled batting order columns.")
            st.stop()

        # Assign batter name for display (use matched name, else fallback)
        if matchup_name_col in hitters_df.columns:
            hitters_df['batter_name'] = hitters_df[matchup_name_col]
        elif 'batter' in hitters_df.columns:
            hitters_df['batter_name'] = hitters_df['batter']
        else:
            hitters_df['batter_name'] = hitters_df[batter_id_col].astype(str)

        # --- FILTER EVENTS FOR ANALYSIS ---
        valid_events = [
            'single', 'double', 'triple', 'homerun', 'home_run',
            'field_out', 'force_out', 'grounded_into_double_play',
            'fielders_choice_out', 'pop_out', 'lineout', 'flyout', 'sac_fly', 'sac_fly_double_play'
        ]
        if 'events' in hitters_df.columns:
            hitters_df = hitters_df[hitters_df['events'].astype(str).str.lower().str.replace(' ', '').isin(valid_events)].copy()
        if 'hr_outcome' not in hitters_df.columns:
            if 'events' in hitters_df.columns:
                hitters_df['hr_outcome'] = hitters_df['events'].astype(str).str.lower().str.replace(' ', '').isin(['homerun', 'home_run']).astype(int)
            else:
                st.error("No HR outcome detected, and 'events' column not available for mapping!")
                st.stop()

        hitters_df = hitters_df.loc[:, ~hitters_df.columns.duplicated()]

        # --- LOAD LOGIT FEATURES ---
        all_model_features = [f for f in logit_weights['feature'].values if f in hitters_df.columns and pd.api.types.is_numeric_dtype(hitters_df[f])]
        if not all_model_features or 'hr_outcome' not in hitters_df.columns:
            st.error("Model features or hr_outcome missing from event-level data.")
            st.stop()

        # Prepare X matrix
        X = hitters_df[all_model_features].fillna(0)
        y = hitters_df['hr_outcome'].astype(int)

        # Auto feature selection for logit (RFECV) + grid search for C
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        from sklearn.feature_selection import RFECV
        logit_base = LogisticRegression(max_iter=200, solver='liblinear')
        rfecv = RFECV(estimator=logit_base, step=1, cv=3, scoring='roc_auc', n_jobs=-1)
        rfecv.fit(X_train, y_train)
        features_mask = rfecv.support_
        X_train_sel = X_train.loc[:, features_mask]
        X_test_sel = X_test.loc[:, features_mask]

        grid = GridSearchCV(LogisticRegression(max_iter=200, solver='liblinear'), param_grid={'C': [0.1, 1, 10]}, cv=3, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train_sel, y_train)
        best_logit = grid.best_estimator_

        hitters_df['logit_prob'] = best_logit.predict_proba(hitters_df.loc[:, features_mask])[:, 1]
        hitters_df['logit_hr_pred'] = (hitters_df['logit_prob'] > threshold).astype(int)

        # --- XGBOOST auto-tune ---
        xgb_params = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        xgb_grid = GridSearchCV(
            xgb.XGBClassifier(n_estimators=100, eval_metric='logloss'),
            xgb_params, cv=3, scoring='roc_auc', n_jobs=-1
        )
        xgb_grid.fit(X_train, y_train)
        best_xgb = xgb_grid.best_estimator_
        hitters_df['xgb_prob'] = best_xgb.predict_proba(X)[:, 1]
        hitters_df['xgb_hr_pred'] = (hitters_df['xgb_prob'] > threshold).astype(int)

        # --- LEADERBOARDS ---
        st.markdown("## Side-by-Side HR Probability Leaderboards (Top 15 Hitters)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Logit Leaderboard (Auto-tuned)")
            logit_leaderboard = (
                hitters_df.groupby('batter_name')
                .agg(
                    n_events=('hr_outcome', 'count'),
                    n_predicted_HR=('logit_hr_pred', 'sum'),
                    mean_logit_prob=('logit_prob', 'mean')
                )
                .sort_values(['n_predicted_HR', 'mean_logit_prob'], ascending=False)
                .head(15)
            )
            st.dataframe(logit_leaderboard)
        with col2:
            st.markdown("#### XGBoost Leaderboard (Auto-tuned)")
            xgb_leaderboard = (
                hitters_df.groupby('batter_name')
                .agg(
                    n_events=('hr_outcome', 'count'),
                    n_predicted_HR=('xgb_hr_pred', 'sum'),
                    mean_xgb_prob=('xgb_prob', 'mean')
                )
                .sort_values(['n_predicted_HR', 'mean_xgb_prob'], ascending=False)
                .head(15)
            )
            st.dataframe(xgb_leaderboard)

        st.markdown("#### Download Full Event-Level Data with Model Scores:")
        st.download_button("⬇️ Download Scored Event CSV", data=hitters_df.to_csv(index=False), file_name="event_level_scored.csv")

        # --- LOGIT REPORT ---
        st.markdown("### Logistic Regression Performance (Auto-tuned)")
        try:
            auc = roc_auc_score(y_test, best_logit.predict_proba(X_test_sel)[:, 1])
            st.metric("Logistic Regression ROC-AUC", round(auc, 4))
            st.code(classification_report(y_test, (best_logit.predict_proba(X_test_sel)[:, 1] > threshold).astype(int)), language='text')
        except Exception as e:
            st.warning(f"Logit model report failed: {e}")

        # --- XGBOOST REPORT ---
        st.markdown("### XGBoost Performance (Auto-tuned)")
        try:
            auc = roc_auc_score(y_test, best_xgb.predict_proba(X_test)[:, 1])
            st.metric("XGBoost ROC-AUC", round(auc, 4))
            st.code(classification_report(y_test, (best_xgb.predict_proba(X_test)[:, 1] > threshold).astype(int)), language='text')
        except Exception as e:
            st.warning(f"XGBoost report failed: {e}")
