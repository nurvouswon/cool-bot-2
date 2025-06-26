import streamlit as st
import pandas as pd
import numpy as np
import re
from pybaseball import statcast
from datetime import datetime, timedelta

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

def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

# ----------- WEATHER PARSER for "92 O CF 12-14 30% outdoor" -----------
def parse_custom_weather_string_v2(s):
    """Parse weather string like '92 O CF 12-14 30% outdoor'."""
    if pd.isna(s): return pd.Series([np.nan]*7, index=['temp','wind_vector','wind_field_dir','wind_mph','humidity','condition','wind_dir_string'])
    s = str(s)
    # Temp
    temp_match = re.search(r'(\d{2,3})\s*[OI]\s', s)
    temp = int(temp_match.group(1)) if temp_match else np.nan
    # Wind O/I
    wind_vector_match = re.search(r'\d{2,3}\s*([OI])\s', s)
    wind_vector = wind_vector_match.group(1) if wind_vector_match else np.nan
    # Field direction
    wind_field_dir_match = re.search(r'\s([A-Z]{2})\s*\d', s)
    wind_field_dir = wind_field_dir_match.group(1) if wind_field_dir_match else np.nan
    # Wind mph (midpoint)
    mph = re.search(r'(\d{1,3})\s*-\s*(\d{1,3})', s)
    if mph:
        wind_mph = (int(mph.group(1)) + int(mph.group(2))) / 2
    else:
        mph = re.search(r'([1-9][0-9]?)\s*(?:mph)?', s)
        wind_mph = int(mph.group(1)) if mph else np.nan
    # Humidity
    humidity_match = re.search(r'(\d{1,3})%', s)
    humidity = int(humidity_match.group(1)) if humidity_match else np.nan
    # Condition
    condition = "outdoor" if "outdoor" in s.lower() else ("indoor" if "indoor" in s.lower() else np.nan)
    wind_dir_string = f"{wind_vector} {wind_field_dir}".strip()
    return pd.Series([temp, wind_vector, wind_field_dir, wind_mph, humidity, condition, wind_dir_string],
                     index=['temp','wind_vector','wind_field_dir','wind_mph','humidity','condition','wind_dir_string'])

@st.cache_data(show_spinner=True)
def fast_rolling_stats(df, id_col, date_col, windows, pitch_types=None, prefix=""):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['launch_speed'] = pd.to_numeric(df['launch_speed'], errors='coerce')
    df['launch_angle'] = pd.to_numeric(df['launch_angle'], errors='coerce')
    if 'pitch_type' in df.columns:
        df['pitch_type'] = df['pitch_type'].astype(str).str.lower().str.strip()
    df = df.drop_duplicates(subset=[id_col, date_col], keep='last')
    df = df.sort_values([id_col, date_col])
    feature_frames = []
    grouped = df.groupby(id_col)
    for name, group in grouped:
        out_row = {}
        for w in windows:
            out_row[f"{prefix}avg_exit_velo_{w}"] = group['launch_speed'].rolling(w, min_periods=1).mean().iloc[-1]
            out_row[f"{prefix}hard_hit_rate_{w}"] = (group['launch_speed'].rolling(w, min_periods=1)
                                                     .apply(lambda x: np.mean(x >= 95)).iloc[-1])
            out_row[f"{prefix}barrel_rate_{w}"] = (((group['launch_speed'] >= 98) &
                                                    (group['launch_angle'] >= 26) &
                                                    (group['launch_angle'] <= 30))
                                                    .rolling(w, min_periods=1).mean().iloc[-1])
            out_row[f"{prefix}fb_rate_{w}"] = (group['launch_angle'].rolling(w, min_periods=1)
                                               .apply(lambda x: np.mean(x >= 25)).iloc[-1])
            out_row[f"{prefix}sweet_spot_rate_{w}"] = (group['launch_angle'].rolling(w, min_periods=1)
                                                       .apply(lambda x: np.mean((x >= 8) & (x <= 32))).iloc[-1])
        if pitch_types is not None and "pitch_type" in group.columns:
            for pt in pitch_types:
                pt_group = group[group['pitch_type'] == pt]
                if pt_group.empty:
                    for w in windows:
                        out_row[f"{prefix}avg_exit_velo_{pt}_{w}"] = np.nan
                        out_row[f"{prefix}hard_hit_rate_{pt}_{w}"] = np.nan
                        out_row[f"{prefix}barrel_rate_{pt}_{w}"] = np.nan
                        out_row[f"{prefix}fb_rate_{pt}_{w}"] = np.nan
                        out_row[f"{prefix}sweet_spot_rate_{pt}_{w}"] = np.nan
                else:
                    for w in windows:
                        out_row[f"{prefix}avg_exit_velo_{pt}_{w}"] = pt_group['launch_speed'].rolling(w, min_periods=1).mean().iloc[-1]
                        out_row[f"{prefix}hard_hit_rate_{pt}_{w}"] = (pt_group['launch_speed'].rolling(w, min_periods=1)
                                                                       .apply(lambda x: np.mean(x >= 95)).iloc[-1])
                        out_row[f"{prefix}barrel_rate_{pt}_{w}"] = (((pt_group['launch_speed'] >= 98) &
                                                                     (pt_group['launch_angle'] >= 26) &
                                                                     (pt_group['launch_angle'] <= 30))
                                                                     .rolling(w, min_periods=1).mean().iloc[-1])
                        out_row[f"{prefix}fb_rate_{pt}_{w}"] = (pt_group['launch_angle'].rolling(w, min_periods=1)
                                                                .apply(lambda x: np.mean(x >= 25)).iloc[-1])
                        out_row[f"{prefix}sweet_spot_rate_{pt}_{w}"] = (pt_group['launch_angle'].rolling(w, min_periods=1)
                                                                        .apply(lambda x: np.mean((x >= 8) & (x <= 32))).iloc[-1])
        out_row[id_col] = name
        feature_frames.append(out_row)
    return pd.DataFrame(feature_frames)

st.set_page_config("MLB HR Analyzer", layout="wide")
tab1, tab2 = st.tabs(["1️⃣ Fetch & Feature Engineer Data", "2️⃣ Upload & Analyze"])

with tab1:
    st.header("Fetch Statcast Data & Generate Features")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", value=datetime.today())

    st.markdown("##### Upload Today's Matchups/Lineups CSV (required for TODAY CSV)")
    uploaded_lineups = st.file_uploader("Upload Today's Matchups CSV", type="csv", key="lineupsup")
    fetch_btn = st.button("Fetch Statcast, Feature Engineer, and Download", type="primary")
    progress = st.empty()

    if fetch_btn and uploaded_lineups is not None:
        progress.progress(5, "Fetching Statcast data...")
        df = statcast(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        progress.progress(10, "Loaded Statcast")
        st.write(f"Loaded {len(df)} raw Statcast events.")
        if len(df) == 0:
            st.error("No data! Try different dates.")
            st.stop()
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')

        # -- Lineups --
        try:
            lineup_df = pd.read_csv(uploaded_lineups)
        except Exception as e:
            st.error(f"Could not read lineup CSV: {e}")
            st.stop()
        # Standardize key columns and all string conversions
        for possible_col in ['mlb id', 'mlb_id', 'batter_id', 'batter']:
            if possible_col in lineup_df.columns and 'batter_id' not in lineup_df.columns:
                lineup_df['batter_id'] = lineup_df[possible_col]
        for possible_col in ['player name', 'player_name', 'name']:
            if possible_col in lineup_df.columns and 'player_name' not in lineup_df.columns:
                lineup_df['player_name'] = lineup_df[possible_col]
        if 'game_date' not in lineup_df.columns:
            for date_col in ['game_date', 'game date']:
                if date_col in lineup_df.columns:
                    lineup_df['game_date'] = lineup_df[date_col]
        if 'game_date' in lineup_df.columns:
            lineup_df['game_date'] = pd.to_datetime(lineup_df['game_date'], errors='coerce').dt.strftime("%Y-%m-%d")
        # Ensure team_code and game_number are string
        if 'team_code' in lineup_df.columns:
            lineup_df['team_code'] = lineup_df['team_code'].astype(str)
        if 'game_number' in lineup_df.columns:
            lineup_df['game_number'] = lineup_df['game_number'].astype(str)
        if 'batter_id' in lineup_df.columns:
            lineup_df['batter_id'] = lineup_df['batter_id'].astype(str)
        if 'mlb_id' in lineup_df.columns:
            lineup_df['mlb_id'] = lineup_df['mlb_id'].astype(str)

        # Parse weather string into all columns
        if 'weather' in lineup_df.columns:
            lineup_df[['temp','wind_vector','wind_field_dir','wind_mph','humidity','condition','wind_dir_string']] = \
                lineup_df['weather'].apply(parse_custom_weather_string_v2)
        st.write("Lineup Weather Debug:")
        st.dataframe(lineup_df[['weather','temp','wind_vector','wind_field_dir','wind_mph','humidity','condition','wind_dir_string']].head(10))

        # ---- Assign pitcher_id for each batter from opponent SP logic (if available) ----
        if all(col in lineup_df.columns for col in ['game_number', 'team_code', 'mlb_id', 'batting_order']):
            games = lineup_df[['game_date', 'game_number']].drop_duplicates()
            opp_pitcher_map = {}
            for _, game in games.iterrows():
                game_date, game_number = game['game_date'], game['game_number']
                teams = lineup_df[
                    (lineup_df['game_date'] == game_date) &
                    (lineup_df['game_number'] == game_number)
                ]['team_code'].unique()
                for team in teams:
                    opp_team = [t for t in teams if t != team]
                    if not opp_team:
                        continue
                    opp_team = opp_team[0]
                    opp_sp = lineup_df[
                        (lineup_df['team_code'] == opp_team) &
                        (lineup_df['game_date'] == game_date) &
                        (lineup_df['game_number'] == game_number) &
                        (lineup_df['batting_order'].astype(str).str.upper().str.strip() == "SP")
                    ]
                    if not opp_sp.empty:
                        opp_pitcher_map[(game_date, game_number, team)] = str(opp_sp.iloc[0]['mlb_id'])
            lineup_df['pitcher_id'] = lineup_df.apply(
                lambda row: opp_pitcher_map.get((row['game_date'], row['game_number'], row['team_code']), np.nan), axis=1
            )
        else:
            lineup_df['pitcher_id'] = np.nan

        # ---------- Statcast Event-level Feature Engineering -----------
        valid_events = [
            'single', 'double', 'triple', 'homerun', 'home_run', 'field_out',
            'force_out', 'grounded_into_double_play', 'fielders_choice_out',
            'pop_out', 'lineout', 'flyout', 'sac_fly', 'sac_fly_double_play'
        ]
        if 'events' in df.columns:
            df['events_clean'] = df['events'].astype(str).str.lower().str.replace(' ', '')
            df = df[df['events_clean'].isin(valid_events)].copy()

        # All string fixes for merge columns
        for col in ['game_date', 'team_code', 'batter_id', 'pitcher_id']:
            if col in df.columns:
                df[col] = df[col].astype(str)
        if 'game_date' in lineup_df.columns:
            lineup_df['game_date'] = lineup_df['game_date'].astype(str)
        if 'team_code' in lineup_df.columns:
            lineup_df['team_code'] = lineup_df['team_code'].astype(str)

        # Identify home_team_code for park mapping
        if 'home_team_code' in df.columns:
            df['park'] = df['home_team_code'].map(team_code_to_park)
        elif 'home_team' in df.columns:
            df['park'] = df['home_team'].str.lower().str.replace(' ', '_').map(park_hr_rate_map).index

        df['park_hr_rate'] = df['park'].map(park_hr_rate_map).fillna(1.0)
        df['park_altitude'] = df['park'].map(park_altitude_map).fillna(0)
        df['roof_status'] = df['park'].map(roof_status_map).fillna("open")

        # If city not present, try from park
        if 'city' not in df.columns:
            df['city'] = df['park'].map({k: v for k, v in team_code_to_park.items() if k in mlb_team_city_map}).fillna("")

        # ------- Statcast Rolling and Custom Features --------
        roll_windows = [3, 5, 7, 14, 20]
        main_pitch_types = ["ff", "sl", "cu", "ch", "si", "fc", "fs", "st", "sinker", "splitter", "sweeper"]

        # For demo: assume basic columns
        if 'batter' in df.columns and 'batter_id' not in df.columns:
            df['batter_id'] = df['batter']
        if 'pitcher' in df.columns and 'pitcher_id' not in df.columns:
            df['pitcher_id'] = df['pitcher']

        # If HR outcome not present, generate
        if 'hr_outcome' not in df.columns:
            df['hr_outcome'] = df['events_clean'].isin(['homerun', 'home_run']).astype(int)

        # Rolling stat features (can be extended)
        batter_cols = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value',
                       'release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z']
        pitcher_cols = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value',
                        'release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z']

        for col in batter_cols:
            if col in df.columns:
                for w in roll_windows:
                    df[f'B_{col}_{w}'] = df.groupby('batter_id')[col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        for col in pitcher_cols:
            if col in df.columns:
                for w in roll_windows:
                    df[f'P_{col}_{w}'] = df.groupby('pitcher_id')[col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())

        # Handedness HR rates
        if 'stand' in df.columns and 'p_throws' in df.columns:
            for w in roll_windows:
                df[f'B_vsP_hand_HR_{w}'] = df.groupby(['batter_id', 'p_throws'])['hr_outcome'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                df[f'P_vsB_hand_HR_{w}'] = df.groupby(['pitcher_id', 'stand'])['hr_outcome'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())

        # Park/hand split HR rates
        if 'park' in df.columns and 'stand' in df.columns:
            for w in [7, 14, 30]:
                df[f'park_hand_HR_{w}'] = df.groupby(['park', 'stand'])['hr_outcome'].transform(lambda x: x.shift(1).rolling(w, min_periods=5).mean())

        # =========== Merge weather/context from lineup to event data ==========
        if 'weather' in lineup_df.columns and 'game_date' in df.columns and 'team_code' in df.columns:
            weather_cols = ['game_date','team_code','temp','wind_vector','wind_field_dir','wind_mph','humidity','condition','wind_dir_string']
            for c in weather_cols:
                if c not in lineup_df.columns:
                    lineup_df[c] = np.nan
            weather_merge = lineup_df[weather_cols].drop_duplicates()
            # MAKE SURE string dtypes match before merge
            weather_merge['game_date'] = weather_merge['game_date'].astype(str)
            weather_merge['team_code'] = weather_merge['team_code'].astype(str)
            df['game_date'] = df['game_date'].astype(str)
            df['team_code'] = df['team_code'].astype(str)
            df = pd.merge(df, weather_merge, how='left', on=['game_date','team_code'])

        # -- Output event-level CSV --
        df = dedup_columns(df)
        st.success(f"Feature engineering complete! {len(df)} batted ball events.")
        st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
        st.dataframe(df.head(20))
        st.download_button(
            "⬇️ Download Event-Level CSV",
            data=df.to_csv(index=False),
            file_name="event_level_hr_features.csv",
            key="download_event_level"
        )

        # ==================== TODAY CSV: 1 row per batter ======================
        cols_for_today = [
            "game_date", "batter_id", "player_name", "pitcher_id", "temp", "humidity", "wind_mph", "wind_dir_string", "condition", "hard_hit_rate_20", "sweet_spot_rate_20"
        ]
        today_rows = []
        missing_weather = 0
        for idx, row in lineup_df.iterrows():
            this_batter_id = str(row['batter_id']).split(".")[0]
            filter_df = df[df['batter_id'].astype(str).str.split('.').str[0] == this_batter_id]
            if not filter_df.empty:
                last_row = filter_df.iloc[-1]
                t = last_row.get("temp", np.nan)
                h = last_row.get("humidity", np.nan)
                if pd.isna(t) or pd.isna(h):
                    missing_weather += 1
                today_rows.append({
                    "game_date": row.get('game_date', np.nan),
                    "batter_id": this_batter_id,
                    "player_name": row.get('player_name', np.nan),
                    "pitcher_id": row.get('pitcher_id', np.nan),
                    "temp": t,
                    "humidity": h,
                    "wind_mph": last_row.get("wind_mph", np.nan),
                    "wind_dir_string": last_row.get("wind_dir_string", np.nan),
                    "condition": last_row.get("condition", np.nan),
                    "hard_hit_rate_20": last_row.get("hard_hit_rate_20", np.nan),
                    "sweet_spot_rate_20": last_row.get("sweet_spot_rate_20", np.nan)
                })
            else:
                today_rows.append({
                    "game_date": row.get('game_date', np.nan),
                    "batter_id": this_batter_id,
                    "player_name": row.get('player_name', np.nan),
                    "pitcher_id": row.get('pitcher_id', np.nan),
                    "temp": np.nan,
                    "humidity": np.nan,
                    "wind_mph": np.nan,
                    "wind_dir_string": np.nan,
                    "condition": np.nan,
                    "hard_hit_rate_20": np.nan,
                    "sweet_spot_rate_20": np.nan
                })

        today_df = pd.DataFrame(today_rows, columns=cols_for_today)
        st.markdown("#### Download TODAY CSV (1 row per batter, matchup & weather):")
        st.dataframe(today_df.head(20))
        st.download_button(
            "⬇️ Download TODAY CSV",
            data=today_df.to_csv(index=False),
            file_name="today_hr_features.csv",
            key="download_today_csv"
        )
        if missing_weather > 0:
            st.warning(f"Weather info missing for {missing_weather} of today's batters. Check weather string in lineups.")

    else:
        st.info("Upload a Matchups/Lineups CSV and select a date range to generate the event-level and TODAY CSVs.")
