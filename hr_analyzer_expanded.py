import streamlit as st
import pandas as pd
import numpy as np
import re
from pybaseball import statcast
from datetime import datetime, timedelta

# ===================== CONTEXT MAPS & RATES =====================
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

        # Normalize common alternate names for park/city if needed
        if 'park' not in lineup_df.columns:
            for c in ['ballpark', 'venue', 'stadium']:
                if c in lineup_df.columns:
                    lineup_df['park'] = lineup_df[c]
                    break
        if 'city' not in lineup_df.columns:
            for c in ['team_city', 'location']:
                if c in lineup_df.columns:
                    lineup_df['city'] = lineup_df[c]
                    break

        for col in ['mlb_id', 'batter_id', 'batter']:
            if col in lineup_df.columns:
                lineup_df['batter_id'] = lineup_df[col]
                break
        if 'player_name' not in lineup_df.columns and 'batter_name' in lineup_df.columns:
            lineup_df['player_name'] = lineup_df['batter_name']

        # --- Parse the weather string for TODAY context ---
        weather_cols = ['temp', 'wind_vector', 'wind_field_dir', 'wind_mph', 'humidity', 'condition', 'wind_dir_string']
        if 'weather' in lineup_df.columns:
            weather_parsed = lineup_df['weather'].apply(parse_custom_weather_string_v2)
            weather_parsed.columns = weather_cols
            lineup_df = pd.concat([lineup_df, weather_parsed], axis=1)
        else:
            for col in weather_cols:
                lineup_df[col] = np.nan

        # --- Fill in park/city using park/city/team_code ---
        if 'park' not in lineup_df.columns or lineup_df['park'].isnull().all():
            if 'team_code' in lineup_df.columns:
                lineup_df['park'] = lineup_df['team_code'].map(team_code_to_park)
        if 'city' not in lineup_df.columns or lineup_df['city'].isnull().all():
            if 'team_code' in lineup_df.columns:
                lineup_df['city'] = lineup_df['team_code'].map(mlb_team_city_map)
        lineup_df['park'] = lineup_df['park'].replace({np.nan: '', None: ''})
        lineup_df['city'] = lineup_df['city'].replace({np.nan: '', None: ''})

        # Add park/contextual features
        lineup_df['park_key'] = lineup_df['park'].str.lower().str.replace(" ", "_")
        lineup_df['park_hr_rate'] = lineup_df['park_key'].map(park_hr_rate_map).fillna(1.0)
        lineup_df['park_altitude'] = lineup_df['park_key'].map(park_altitude_map).fillna(0)
        lineup_df['roof_status'] = lineup_df['park_key'].map(roof_status_map).fillna("open")
        # Handed park HR rates (if batter_hand available)
        if 'batter_hand' in lineup_df.columns:
            lineup_df['park_hand_hr_rate'] = [
                park_hand_hr_rate.get(park, {}).get(hand, 1.0)
                for park, hand in zip(lineup_df['park_key'], lineup_df['batter_hand'].fillna('R'))
            ]
        else:
            lineup_df['park_hand_hr_rate'] = 1.0

        # --- Assign opposing pitcher by team (usually via starter_id/pitcher_id columns or left merge logic) ---
        pitcher_id_col = None
        for c in ['pitcher_id', 'starter_id', 'opposing_pitcher_id']:
            if c in lineup_df.columns:
                pitcher_id_col = c
                break

        # --- Prep today context for merging (batter_id, pitcher_id, park, city, weather...) ---
        merge_cols = ['game_date', 'batter_id', 'pitcher_id']
        today_merge_cols = ['game_date', 'batter_id']
        merge_fields = [
            'temp', 'humidity', 'wind_mph', 'wind_dir_string', 'condition', 'park',
            'park_hr_rate', 'park_altitude', 'roof_status', 'city', 'park_hand_hr_rate'
        ]
        today_context = lineup_df.copy()
        if pitcher_id_col is not None:
            today_context['pitcher_id'] = lineup_df[pitcher_id_col]
        elif 'opposing_pitcher_id' in lineup_df.columns:
            today_context['pitcher_id'] = lineup_df['opposing_pitcher_id']
        else:
            today_context['pitcher_id'] = np.nan

        # --- Compute rolling batted ball features for batter (3, 5, 7, 14, 20) ---
        st.write("Generating rolling batted ball features (batters & pitchers, all windows, all types)...")
        df['batter_id'] = df['batter'] if 'batter' in df.columns else df['batter_id']
        batter_features = fast_rolling_stats(df, id_col='batter_id', date_col='game_date', windows=[3, 5, 7, 14, 20],
                                            pitch_types=['ff', 'sl', 'cu', 'ch', 'si', 'fc', 'fs', 'st', 'sinker', 'splitter', 'sweeper'], prefix="b_")
        pitcher_features = fast_rolling_stats(df, id_col='pitcher', date_col='game_date', windows=[3, 5, 7, 14, 20],
                                             pitch_types=['ff', 'sl', 'cu', 'ch', 'si', 'fc', 'fs', 'st', 'sinker', 'splitter', 'sweeper'], prefix="p_")
        batter_features = dedup_columns(batter_features)
        pitcher_features = dedup_columns(pitcher_features)
        # Merge all rolling features onto statcast rows
        progress.progress(30, "Rolling feature merges...")
        df = df.merge(batter_features, left_on='batter_id', right_on='batter_id', how='left')
        df = df.merge(pitcher_features, left_on='pitcher', right_on='pitcher', how='left', suffixes=('', '_pitcher'))
        progress.progress(40, "Merged rolling features.")

        # --- Merge TODAY lineup context onto Statcast batted ball events (for TODAY CSV) ---
        st.write("Generating TODAY batter rows and context merges...")
        today_context_merge = today_context[['game_date', 'batter_id', 'pitcher_id'] + merge_fields].copy()
        # Remove duplicate batter_id if present
        today_context_merge = today_context_merge.drop_duplicates(subset=['game_date', 'batter_id'], keep='first')

        # Create TODAY (batter-level, not event-level) DataFrame
        unique_today = today_context_merge[['game_date', 'batter_id', 'pitcher_id']].drop_duplicates()
        today_df = unique_today.copy()
        for f in merge_fields:
            today_df[f] = today_df.apply(
                lambda row: today_context_merge.loc[
                    (today_context_merge['game_date'] == row['game_date']) &
                    (today_context_merge['batter_id'] == row['batter_id']),
                    f
                ].values[0] if not today_context_merge.loc[
                    (today_context_merge['game_date'] == row['game_date']) &
                    (today_context_merge['batter_id'] == row['batter_id']),
                    f
                ].empty else np.nan, axis=1
            )

        # Merge in rolling batter features to today_df (for each batter/game_date)
        for col in batter_features.columns:
            if col != 'batter_id' and col not in today_df.columns:
                today_df = today_df.merge(batter_features[['batter_id', col]], left_on='batter_id', right_on='batter_id', how='left')
        # Merge in rolling pitcher features to today_df
        for col in pitcher_features.columns:
            if col != 'pitcher' and col not in today_df.columns:
                today_df = today_df.merge(pitcher_features.rename(columns={'pitcher': 'pitcher_id'})[['pitcher_id', col]], left_on='pitcher_id', right_on='pitcher_id', how='left')

        today_df = dedup_columns(today_df)
        today_df = today_df.replace({np.nan: '', None: ''})

        # --- Download buttons for outputs ---
        st.success("Feature engineering complete! {} batted ball events.".format(len(df)))
        st.download_button("Download Event-Level CSV (all features, 1 row per batted ball event):", data=df.to_csv(index=False), file_name="event_level_features.csv", mime="text/csv")
        st.download_button("Download TODAY CSV (1 row per batter, matchup, rolling features & weather):", data=today_df.to_csv(index=False), file_name="today_batter_features.csv", mime="text/csv")

        # Show preview of output
        st.markdown("#### TODAY CSV Weather/Context Audit (first 15 rows):")
        st.dataframe(today_df[[
            'game_date', 'batter_id', 'pitcher_id', 'temp', 'humidity', 'wind_mph',
            'wind_dir_string', 'condition', 'park', 'park_hr_rate', 'park_altitude',
            'roof_status', 'city', 'park_hand_hr_rate'
        ]].head(15))

    elif fetch_btn and uploaded_lineups is None:
        st.warning("Please upload today's matchup/lineup CSV.")
