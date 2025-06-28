import streamlit as st
import pandas as pd
import numpy as np
import re
from pybaseball import statcast
from datetime import datetime, timedelta

# ============ CONTEXT MAPS & RATES ============
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
park_hand_hr_rate = {
    'yankee_stadium': {'L': 1.28, 'R': 1.12},
    'fenway_park': {'L': 0.93, 'R': 1.01},
    'coors_field': {'L': 1.37, 'R': 1.26},
    'rogers_centre': {'L': 1.12, 'R': 1.10},
    'guaranteed_rate_field': {'L': 1.14, 'R': 1.22},
    'camden_yards': {'L': 1.16, 'R': 1.09},
    'dodger_stadium': {'L': 1.14, 'R': 1.08},
    'busch_stadium': {'L': 0.85, 'R': 0.89},
    'wrigley_field': {'L': 1.16, 'R': 1.10},
    'petco_park': {'L': 0.83, 'R': 0.86},
    'minute_maid_park': {'L': 1.08, 'R': 1.04},
    't-mobile_park': {'L': 0.88, 'R': 0.84},
    'oracle_park': {'L': 0.80, 'R': 0.85},
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
    'LAD': 'dodger_stadium', 'OAK': 'sutter_health_park', 'ATH': 'sutter_health_park',
    'CIN': 'great_american_ball_park', 'DET': 'comerica_park', 'HOU': 'minute_maid_park',
    'MIA': 'loandepot_park', 'TB': 'tropicana_field', 'MIL': 'american_family_field',
    'SD': 'petco_park', 'SF': 'oracle_park', 'TOR': 'rogers_centre', 'CLE': 'progressive_field',
    'MIN': 'target_field', 'KC': 'kauffman_stadium', 'CWS': 'guaranteed_rate_field',
    'CHW': 'guaranteed_rate_field', 'LAA': 'angel_stadium', 'SEA': 't-mobile_park',
    'TEX': 'globe_life_field', 'ARI': 'chase_field', 'AZ': 'chase_field', 'COL': 'coors_field', 'PIT': 'pnc_park',
    'STL': 'busch_stadium', 'BAL': 'camden_yards', 'WSH': 'nationals_park', 'WAS': 'nationals_park'
}
mlb_team_city_map = {
    'ANA': 'Anaheim', 'ARI': 'Phoenix', 'AZ': 'Phoenix', 'ATL': 'Atlanta', 'BAL': 'Baltimore', 'BOS': 'Boston',
    'CHC': 'Chicago', 'CIN': 'Cincinnati', 'CLE': 'Cleveland', 'COL': 'Denver', 'CWS': 'Chicago',
    'CHW': 'Chicago', 'DET': 'Detroit', 'HOU': 'Houston', 'KC': 'Kansas City', 'LAA': 'Anaheim',
    'LAD': 'Los Angeles', 'MIA': 'Miami', 'MIL': 'Milwaukee', 'MIN': 'Minneapolis', 'NYM': 'New York',
    'NYY': 'New York', 'OAK': 'Oakland', 'ATH': 'Oakland', 'PHI': 'Philadelphia', 'PIT': 'Pittsburgh',
    'SD': 'San Diego', 'SEA': 'Seattle', 'SF': 'San Francisco', 'STL': 'St. Louis', 'TB': 'St. Petersburg',
    'TEX': 'Arlington', 'TOR': 'Toronto', 'WSH': 'Washington', 'WAS': 'Washington'
}
league_avg_hr_rate = 0.030
pitchtype_hr_rate = {
    'ff': 0.040, 'sl': 0.033, 'cu': 0.024, 'ch': 0.019, 'si': 0.029,
    'fc': 0.027, 'fs': 0.020, 'st': 0.018, 'sinker': 0.029, 'splitter': 0.021, 'sweeper': 0.017
}
pitchtype_hr_rate_by_hand = {
    'ff': {'L': 0.043, 'R': 0.038}, 'sl': {'L': 0.029, 'R': 0.036},
    'cu': {'L': 0.022, 'R': 0.025}, 'ch': {'L': 0.021, 'R': 0.018}, 'si': {'L': 0.031, 'R': 0.028}
}
platoon_hr_rate = {'L_vs_R': 0.035, 'L_vs_L': 0.025, 'R_vs_L': 0.041, 'R_vs_R': 0.027}

def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

def parse_custom_weather_string_v2(s):
    # Returns: temp, wind_vector, wind_field_dir, wind_mph, humidity, condition, wind_dir_string
    if pd.isna(s): return pd.Series([np.nan]*7, index=['temp','wind_vector','wind_field_dir','wind_mph','humidity','condition','wind_dir_string'])
    s = str(s)
    temp_match = re.search(r'(\d{2,3})\s*[OI°]?\s', s)
    temp = int(temp_match.group(1)) if temp_match else np.nan
    wind_vector_match = re.search(r'\d{2,3}\s*([OI])\s', s)
    wind_vector = wind_vector_match.group(1) if wind_vector_match else np.nan
    wind_field_dir_match = re.search(r'\s([A-Z]{2})\s*\d', s)
    wind_field_dir = wind_field_dir_match.group(1) if wind_field_dir_match else np.nan
    mph = re.search(r'(\d{1,3})\s*-\s*(\d{1,3})', s)
    if mph:
        wind_mph = (int(mph.group(1)) + int(mph.group(2))) / 2
    else:
        mph = re.search(r'([1-9][0-9]?)\s*(?:mph)?', s)
        wind_mph = int(mph.group(1)) if mph else np.nan
    humidity_match = re.search(r'(\d{1,3})%', s)
    humidity = int(humidity_match.group(1)) if humidity_match else np.nan
    condition = "outdoor" if "outdoor" in s.lower() else ("indoor" if "indoor" in s.lower() else np.nan)
    wind_dir_string = f"{wind_vector} {wind_field_dir}".strip()
    return pd.Series([temp, wind_vector, wind_field_dir, wind_mph, humidity, condition, wind_dir_string],
                     index=['temp','wind_vector','wind_field_dir','wind_mph','humidity','condition','wind_dir_string'])

@st.cache_data(show_spinner=True)
def fast_rolling_stats(df, id_col, date_col, windows, pitch_types=None, prefix=""):
    df = df.copy()
    if id_col in df.columns and date_col in df.columns:
        df = df.drop_duplicates(subset=[id_col, date_col], keep='last')
        df = df.sort_values([id_col, date_col])
    if 'launch_speed' in df.columns:
        df['launch_speed'] = pd.to_numeric(df['launch_speed'], errors='coerce')
    if 'launch_angle' in df.columns:
        df['launch_angle'] = pd.to_numeric(df['launch_angle'], errors='coerce')
    if 'pitch_type' in df.columns:
        df['pitch_type'] = df['pitch_type'].astype(str).str.lower().str.strip()
    feature_frames = []
    grouped = df.groupby(id_col)
    for name, group in grouped:
        out_row = {}
        for w in windows:
            if 'launch_speed' in group.columns:
                out_row[f"{prefix}avg_exit_velo_{w}"] = group['launch_speed'].rolling(w, min_periods=1).mean().iloc[-1]
                out_row[f"{prefix}hard_hit_rate_{w}"] = (group['launch_speed'].rolling(w, min_periods=1)
                                                         .apply(lambda x: np.mean(x >= 95)).iloc[-1])
            if 'launch_angle' in group.columns:
                out_row[f"{prefix}fb_rate_{w}"] = (group['launch_angle'].rolling(w, min_periods=1)
                                                   .apply(lambda x: np.mean(x >= 25)).iloc[-1])
                out_row[f"{prefix}sweet_spot_rate_{w}"] = (group['launch_angle'].rolling(w, min_periods=1)
                                                           .apply(lambda x: np.mean((x >= 8) & (x <= 32))).iloc[-1])
            if 'launch_speed' in group.columns and 'launch_angle' in group.columns:
                out_row[f"{prefix}barrel_rate_{w}"] = (((group['launch_speed'] >= 98) &
                                                        (group['launch_angle'] >= 26) &
                                                        (group['launch_angle'] <= 30))
                                                        .rolling(w, min_periods=1).mean().iloc[-1])
            for feat in ['hit_distance_sc', 'woba_value', 'release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z']:
                if feat in group.columns:
                    out_row[f"{prefix}{feat}_{w}"] = group[feat].rolling(w, min_periods=1).mean().iloc[-1]
        if pitch_types is not None and "pitch_type" in group.columns:
            for pt in pitch_types:
                pt_group = group[group['pitch_type'] == pt]
                for w in windows:
                    key = f"{prefix}{pt}_"
                    if not pt_group.empty:
                        if 'launch_speed' in pt_group.columns:
                            out_row[f"{key}avg_exit_velo_{w}"] = pt_group['launch_speed'].rolling(w, min_periods=1).mean().iloc[-1]
                            out_row[f"{key}hard_hit_rate_{w}"] = (pt_group['launch_speed'].rolling(w, min_periods=1)
                                                                   .apply(lambda x: np.mean(x >= 95)).iloc[-1])
                        if 'launch_angle' in pt_group.columns:
                            out_row[f"{key}fb_rate_{w}"] = (pt_group['launch_angle'].rolling(w, min_periods=1)
                                                           .apply(lambda x: np.mean(x >= 25)).iloc[-1])
                            out_row[f"{key}sweet_spot_rate_{w}"] = (pt_group['launch_angle'].rolling(w, min_periods=1)
                                                                     .apply(lambda x: np.mean((x >= 8) & (x <= 32))).iloc[-1])
                        if 'launch_speed' in pt_group.columns and 'launch_angle' in pt_group.columns:
                            out_row[f"{key}barrel_rate_{w}"] = (((pt_group['launch_speed'] >= 98) &
                                                                 (pt_group['launch_angle'] >= 26) &
                                                                 (pt_group['launch_angle'] <= 30))
                                                                 .rolling(w, min_periods=1).mean().iloc[-1])
                    else:
                        for feat in ['avg_exit_velo', 'hard_hit_rate', 'barrel_rate', 'fb_rate', 'sweet_spot_rate']:
                            out_row[f"{key}{feat}_{w}"] = np.nan
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
        progress.progress(3, "Fetching Statcast data...")
        df = statcast(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        progress.progress(10, "Loaded Statcast")
        st.write(f"Loaded {len(df)} raw Statcast events.")
        if len(df) == 0:
            st.error("No data! Try different dates.")
            st.stop()
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        progress.progress(12, "Loaded and formatted Statcast columns.")

        # --- Read and clean lineups ---
        try:
            lineup_df = pd.read_csv(uploaded_lineups)
        except Exception as e:
            st.error(f"Could not read lineup CSV: {e}")
            st.stop()
        lineup_df.columns = [str(c).strip().lower().replace(" ", "_") for c in lineup_df.columns]
        for col in ['mlb_id', 'batter_id', 'batter']:
            if col in lineup_df.columns and 'batter_id' not in lineup_df.columns:
                lineup_df['batter_id'] = lineup_df[col]
        for col in ['player_name', 'player name', 'name']:
            if col in lineup_df.columns and 'player_name' not in lineup_df.columns:
                lineup_df['player_name'] = lineup_df[col]
        if 'game_date' not in lineup_df.columns:
            for date_col in ['game_date', 'game date']:
                if date_col in lineup_df.columns:
                    lineup_df['game_date'] = lineup_df[date_col]
        if 'game_date' in lineup_df.columns:
            lineup_df['game_date'] = pd.to_datetime(lineup_df['game_date'], errors='coerce').dt.strftime("%Y-%m-%d")
        if 'batting_order' in lineup_df.columns:
            lineup_df['batting_order'] = lineup_df['batting_order'].astype(str).str.upper().str.strip()
        if 'team_code' in lineup_df.columns:
            lineup_df['team_code'] = lineup_df['team_code'].astype(str).str.strip().str.upper()
        if 'game_number' in lineup_df.columns:
            lineup_df['game_number'] = lineup_df['game_number'].astype(str).str.strip()
        for col in ['batter_id', 'mlb_id']:
            if col in lineup_df.columns:
                lineup_df[col] = lineup_df[col].astype(str).str.replace('.0','',regex=False).str.strip()

        # ==== Parse Weather Fields from Weather String ====
        if 'weather' in lineup_df.columns:
            wx_parsed = lineup_df['weather'].apply(parse_custom_weather_string_v2)
            lineup_df = pd.concat([lineup_df, wx_parsed], axis=1)

        # ==== Assign Opposing SP for Each Batter ====
        progress.progress(14, "Assigning opposing pitcher for each batter in lineup...")
        pitcher_col_assigned = False
        if {'game_date', 'game_number', 'team_code', 'mlb_id', 'batting_order'}.issubset(set(lineup_df.columns)):
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
                        (lineup_df['batting_order'] == "SP")
                    ]
                    if not opp_sp.empty:
                        opp_pitcher_map[(game_date, game_number, team)] = str(opp_sp.iloc[0]['mlb_id'])
            lineup_df['pitcher_id'] = lineup_df.apply(
                lambda row: opp_pitcher_map.get((row['game_date'], row['game_number'], row['team_code']), np.nan), axis=1
            )
            pitcher_col_assigned = True
        else:
            lineup_df['pitcher_id'] = np.nan

        # ==== STATCAST EVENT-LEVEL ENGINEERING ====
        progress.progress(18, "Adding park/city/context and cleaning Statcast event data...")

        for col in ['batter_id', 'mlb_id', 'pitcher_id', 'team_code']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('.0','',regex=False).str.strip()

        # Add park, city, context (UPPER for all codes)
        if 'home_team_code' in df.columns:
            df['team_code'] = df['home_team_code'].str.upper()
            df['park'] = df['home_team_code'].str.lower().str.replace(' ', '_')
        if 'home_team' in df.columns and 'park' not in df.columns:
            df['park'] = df['home_team'].str.lower().str.replace(' ', '_')
        if 'team_code' not in df.columns and 'park' in df.columns:
            park_to_team = {v:k for k,v in team_code_to_park.items()}
            df['team_code'] = df['park'].map(park_to_team).str.upper()
        df['team_code'] = df['team_code'].astype(str).str.upper()
        # Always try to get park/city from matchup csv if available for today csv, but for event-level, get from mapping
        df['park'] = df['team_code'].map(team_code_to_park).str.lower()
        df['park_hr_rate'] = df['park'].map(park_hr_rate_map).fillna(1.0)
        df['park_altitude'] = df['park'].map(park_altitude_map).fillna(0)
        df['roof_status'] = df['park'].map(roof_status_map).fillna("open")
        df['city'] = df['team_code'].map(mlb_team_city_map).fillna("")
        df['park_hand_hr_rate'] = df['park'].map(lambda x: park_hand_hr_rate.get(x, np.nan))
        df['pitchtype_hr_rate'] = str(pitchtype_hr_rate)
        df['pitchtype_hr_rate_hand'] = str(pitchtype_hr_rate_by_hand)
        df['platoon_hr_rate'] = str(platoon_hr_rate)

        # HR outcome flag
        if 'events' in df.columns:
            df['events_clean'] = df['events'].astype(str).str.lower().str.replace(' ', '')
        else:
            df['events_clean'] = ""
        if 'hr_outcome' not in df.columns:
            df['hr_outcome'] = df['events_clean'].isin(['homerun', 'home_run']).astype(int)

        valid_events = [
            'single', 'double', 'triple', 'homerun', 'home_run', 'field_out',
            'force_out', 'grounded_into_double_play', 'fielders_choice_out',
            'pop_out', 'lineout', 'flyout', 'sac_fly', 'sac_fly_double_play'
        ]
        df = df[df['events_clean'].isin(valid_events)].copy()

        # Rolling stat features
        progress.progress(22, "Computing rolling Statcast features (batter & pitcher)...")
        roll_windows = [3, 5, 7, 14, 20]
        main_pitch_types = ["ff", "sl", "cu", "ch", "si", "fc", "fs", "st", "sinker", "splitter", "sweeper"]
        for col in ['batter', 'batter_id']:
            if col in df.columns:
                df['batter_id'] = df[col]
        for col in ['pitcher', 'pitcher_id']:
            if col in df.columns:
                df['pitcher_id'] = df[col]

        batter_event = fast_rolling_stats(df, "batter_id", "game_date", roll_windows, main_pitch_types, prefix="b_")
        if not batter_event.empty:
            batter_event = batter_event.set_index('batter_id')
        df_for_pitchers = df.copy()
        if 'batter_id' in df_for_pitchers.columns:
            df_for_pitchers = df_for_pitchers.drop(columns=['batter_id'])
        df_for_pitchers = df_for_pitchers.rename(columns={"pitcher_id": "batter_id"})
        pitcher_event = fast_rolling_stats(
            df_for_pitchers, "batter_id", "game_date", roll_windows, main_pitch_types, prefix="p_"
        )
        if not pitcher_event.empty:
            pitcher_event = pitcher_event.set_index('batter_id')
        df = pd.merge(df, batter_event.reset_index(), how="left", left_on="batter_id", right_on="batter_id")
        df = pd.merge(df, pitcher_event.reset_index(), how="left", left_on="pitcher_id", right_on="batter_id", suffixes=('', '_pitcherstat'))
        if 'batter_id_pitcherstat' in df.columns:
            df = df.drop(columns=['batter_id_pitcherstat'])
        df = dedup_columns(df)

        progress.progress(80, "Event-level feature engineering/merges complete.")

        # =================== OUTPUTS =======================
        st.success(f"Feature engineering complete! {len(df)} batted ball events.")
        st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
        st.dataframe(df.head(20))
        st.download_button(
            "⬇️ Download Event-Level CSV",
            data=df.to_csv(index=False),
            file_name="event_level_hr_features.csv",
            key="download_event_level"
        )

        # ===== TODAY CSV: 1 row per batter with all rolling/context features =====
        progress.progress(95, "Generating TODAY batter rows and context merges...")
        rolling_feature_cols = [col for col in df.columns if (
            col.startswith('b_') or col.startswith('p_')
        ) and any(str(w) in col for w in roll_windows)]
        # All context/weather columns you want in final
        extra_context_cols = [
            'park', 'park_hr_rate', 'park_altitude', 'roof_status', 'city',
            'park_hand_hr_rate', 'pitchtype_hr_rate', 'pitchtype_hr_rate_hand', 'platoon_hr_rate'
        ]
        today_cols = [
            'game_date', 'batter_id', 'player_name', 'pitcher_id',
            'temp', 'humidity', 'wind_mph', 'wind_dir_string', 'condition'
        ] + extra_context_cols + rolling_feature_cols

        today_rows = []
        for idx, row in lineup_df.iterrows():
            this_batter_id = str(row['batter_id']).split(".")[0]
            filter_df = df[df['batter_id'].astype(str).str.split('.').str[0] == this_batter_id]
            # Get most recent stats/context/weather for this batter
            row_out = {c: np.nan for c in today_cols}
            if not filter_df.empty:
                last_row = filter_df.iloc[-1]
                for c in extra_context_cols + rolling_feature_cols:
                    row_out[c] = last_row.get(c, np.nan)
            # Always pull these from lineup_df or parsed weather string
            row_out['player_name'] = row.get('player_name', np.nan)
            row_out['batter_id'] = this_batter_id
            row_out['pitcher_id'] = row.get('pitcher_id', np.nan)
            row_out['game_date'] = row.get('game_date', np.nan)
            for wx in ['temp', 'humidity', 'wind_mph', 'condition', 'wind_dir_string']:
                if wx in row:
                    row_out[wx] = row[wx]
            today_rows.append(row_out)

        today_df = pd.DataFrame(today_rows, columns=today_cols)
        st.markdown("#### Download TODAY CSV (1 row per batter, matchup, rolling features & weather):")
        st.dataframe(today_df.head(20))
        st.download_button(
            "⬇️ Download TODAY CSV",
            data=today_df.to_csv(index=False),
            file_name="today_hr_features.csv",
            key="download_today_csv"
        )
        st.success("All files and debug outputs ready.")
        progress.progress(100, "All complete.")

    else:
        st.info("Upload a Matchups/Lineups CSV and select a date range to generate the event-level and TODAY CSVs.")
