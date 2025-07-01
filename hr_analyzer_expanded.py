import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import gc
from datetime import datetime, timedelta

# ==========================
#   CONTEXT MAPS & DEEP RESEARCH HR MULTIPLIERS
# ==========================

# --- Park/Team/City/Etc ---
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

# -------------- BATTER SIDE (Deep Research: HR park multipliers by all/RHB/LHB) --------------
batter_park_hr_percent_map_all = {
    'ARI': 0.98, 'ATL': 0.95, 'BAL': 1.11, 'BOS': 0.84, 'CHC': 1.03, 'CHW': 1.25, 'CIN': 1.27, 'CLE': 0.96, 'COL': 1.06, 'DET': 0.96,
    'HOU': 1.10, 'KC': 0.83, 'LAA': 1.01, 'LAD': 1.11, 'MIA': 0.85, 'MIL': 1.14, 'MIN': 0.94, 'NYM': 1.07, 'NYY': 1.20, 'OAK': 0.90,
    'PHI': 1.18, 'PIT': 0.83, 'SD': 1.02, 'SEA': 1.00, 'SF': 0.75, 'STL': 0.86, 'TB': 0.96, 'TEX': 1.07, 'TOR': 1.09, 'WAS': 1.00,
}
batter_park_hr_percent_map_rhb = {
    'ARI': 1.00, 'ATL': 0.93, 'BAL': 1.09, 'BOS': 0.90, 'CHC': 1.09, 'CHW': 1.26, 'CIN': 1.27, 'CLE': 0.91, 'COL': 1.05,
    'DET': 0.96, 'HOU': 1.10, 'KC': 0.83, 'LAA': 1.01, 'LAD': 1.11, 'MIA': 0.84, 'MIL': 1.12, 'MIN': 0.95, 'NYM': 1.11,
    'NYY': 1.15, 'OAK': 0.91, 'PHI': 1.18, 'PIT': 0.80, 'SD': 1.02, 'SEA': 1.03, 'SF': 0.76, 'STL': 0.84, 'TB': 0.94,
    'TEX': 1.06, 'TOR': 1.11, 'WAS': 1.02,
}
batter_park_hr_percent_map_lhb = {
    'ARI': 0.98, 'ATL': 0.99, 'BAL': 1.13, 'BOS': 0.75, 'CHC': 0.93, 'CHW': 1.23, 'CIN': 1.29, 'CLE': 1.01, 'COL': 1.07,
    'DET': 0.96, 'HOU': 1.09, 'KC': 0.81, 'LAA': 1.00, 'LAD': 1.12, 'MIA': 0.87, 'MIL': 1.19, 'MIN': 0.91, 'NYM': 1.06,
    'NYY': 1.28, 'OAK': 0.87, 'PHI': 1.19, 'PIT': 0.90, 'SD': 0.98, 'SEA': 0.96, 'SF': 0.73, 'STL': 0.90, 'TB': 0.99,
    'TEX': 1.11, 'TOR': 1.05, 'WAS': 0.96,
}

# -------------- PITCHER SIDE (Deep Research: HR park multipliers by all/RHB/LHB) --------------
pitcher_park_hr_percent_map_all = {
    'ARI': 0.98, 'ATL': 0.95, 'BAL': 1.11, 'BOS': 0.84, 'CHC': 1.03, 'CHW': 1.25, 'CIN': 1.27, 'CLE': 0.96, 'COL': 1.06, 'DET': 0.96,
    'HOU': 1.10, 'KC': 0.83, 'LAA': 1.01, 'LAD': 1.11, 'MIA': 0.85, 'MIL': 1.14, 'MIN': 0.94, 'NYM': 1.07, 'NYY': 1.20, 'OAK': 0.90,
    'PHI': 1.18, 'PIT': 0.83, 'SD': 1.02, 'SEA': 1.00, 'SF': 0.75, 'STL': 0.86, 'TB': 0.96, 'TEX': 1.07, 'TOR': 1.09, 'WAS': 1.00,
}
pitcher_park_hr_percent_map_rhb = {
    'ARI': 0.97, 'ATL': 1.01, 'BAL': 1.16, 'BOS': 0.84, 'CHC': 1.02, 'CHW': 1.28, 'CIN': 1.27, 'CLE': 0.98, 'COL': 1.06, 'DET': 0.95,
    'HOU': 1.11, 'KC': 0.84, 'LAA': 1.01, 'LAD': 1.11, 'MIA': 0.84, 'MIL': 1.14, 'MIN': 0.96, 'NYM': 1.07, 'NYY': 1.24, 'OAK': 0.90,
    'PHI': 1.19, 'PIT': 0.85, 'SD': 1.02, 'SEA': 1.01, 'SF': 0.73, 'STL': 0.84, 'TB': 0.97, 'TEX': 1.10, 'TOR': 1.11, 'WAS': 1.03,
}
pitcher_park_hr_percent_map_lhb = {
    'ARI': 0.99, 'ATL': 0.79, 'BAL': 0.97, 'BOS': 0.83, 'CHC': 1.03, 'CHW': 1.18, 'CIN': 1.27, 'CLE': 0.89, 'COL': 1.05, 'DET': 0.97,
    'HOU': 1.07, 'KC': 0.79, 'LAA': 1.01, 'LAD': 1.11, 'MIA': 0.90, 'MIL': 1.14, 'MIN': 0.89, 'NYM': 1.10, 'NYY': 1.12, 'OAK': 0.89,
    'PHI': 1.16, 'PIT': 0.78, 'SD': 1.02, 'SEA': 0.97, 'SF': 0.82, 'STL': 0.96, 'TB': 0.94, 'TEX': 1.01, 'TOR': 1.06, 'WAS': 0.90,
}

# ==================== Helper Functions ======================

def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

def parse_custom_weather_string_v2(s):
    if pd.isna(s):
        return pd.Series([np.nan]*7, index=['temp','wind_vector','wind_field_dir','wind_mph','humidity','condition','wind_dir_string'])
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

def downcast_numeric(df):
    for col in df.select_dtypes(include=['float']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def rolling_apply(series, window, func):
    if len(series) < 1:
        return np.nan
    result = series.rolling(window, min_periods=1).apply(func)
    return result.iloc[-1] if not result.empty else np.nan

# =============== ADVANCED ROLLING STATS (INCLUDES SPRAY ANGLE) ===============
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
    # SPRAY ANGLE (statcast field: 'spray_angle')
    if 'hc_x' in df.columns and 'hc_y' in df.columns and 'spray_angle' not in df.columns:
        # Statcast spray angle formula
        df['spray_angle'] = np.degrees(np.arctan2(df['hc_x']-125.42, 198.27-df['hc_y']))
    results = []
    for name, group in df.groupby(id_col, sort=False):
        out_row = {}
        ls = group['launch_speed'] if 'launch_speed' in group.columns else None
        la = group['launch_angle'] if 'launch_angle' in group.columns else None
        hd = group['hit_distance_sc'] if 'hit_distance_sc' in group.columns else None
        spr = group['spray_angle'] if 'spray_angle' in group.columns else None
        for w in windows:
            if ls is not None:
                out_row[f"{prefix}avg_exit_velo_{w}"] = ls.rolling(w, min_periods=1).mean().iloc[-1]
                out_row[f"{prefix}hard_hit_rate_{w}"] = ls.rolling(w, min_periods=1).apply(lambda x: np.mean(x >= 95)).iloc[-1]
            if la is not None:
                out_row[f"{prefix}fb_rate_{w}"] = la.rolling(w, min_periods=1).apply(lambda x: np.mean(x >= 25)).iloc[-1]
                out_row[f"{prefix}sweet_spot_rate_{w}"] = la.rolling(w, min_periods=1).apply(lambda x: np.mean((x >= 8) & (x <= 32))).iloc[-1]
            if ls is not None and la is not None:
                barrels = ((ls >= 98) & (la >= 26) & (la <= 30)).astype(float)
                out_row[f"{prefix}barrel_rate_{w}"] = barrels.rolling(w, min_periods=1).mean().iloc[-1]
            if hd is not None:
                out_row[f"{prefix}hit_dist_avg_{w}"] = hd.rolling(w, min_periods=1).mean().iloc[-1]
            if spr is not None:
                out_row[f"{prefix}spray_angle_avg_{w}"] = spr.rolling(w, min_periods=1).mean().iloc[-1]
                out_row[f"{prefix}pull_rate_{w}"] = spr.rolling(w, min_periods=1).apply(lambda x: np.mean(x > 10)).iloc[-1]
                out_row[f"{prefix}opp_rate_{w}"] = spr.rolling(w, min_periods=1).apply(lambda x: np.mean(x < -10)).iloc[-1]
        out_row[id_col] = name
        results.append(out_row)
    return pd.DataFrame(results)

# ==================== STREAMLIT APP ======================

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
        import pybaseball
        from pybaseball import statcast

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
        if "park" in lineup_df.columns:
            lineup_df["park"] = lineup_df["park"].astype(str).str.lower().str.replace(" ", "_")
        for col in ['mlb_id', 'batter_id']:
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

        # ==== Parse Weather Fields from Weather String (for TODAY CSV only) ====
        if 'weather' in lineup_df.columns:
            wx_parsed = lineup_df['weather'].apply(parse_custom_weather_string_v2)
            lineup_df = pd.concat([lineup_df, wx_parsed], axis=1)

        # ==== Assign Opposing SP for Each Batter ====
        progress.progress(14, "Assigning opposing pitcher for each batter in lineup...")
        lineup_df['pitcher_id'] = np.nan
        grouped = lineup_df.groupby(['game_date', 'park', 'time']) if 'time' in lineup_df.columns else lineup_df.groupby(['game_date', 'park'])
        for group_key, group in grouped:
            if 'team_code' not in group.columns: continue
            teams = group['team_code'].unique()
            if len(teams) < 2: continue
            team_sps = {}
            for team in teams:
                sp_row = group[(group['team_code'] == team) & (group['batting_order'] == "SP")]
                if not sp_row.empty:
                    team_sps[team] = str(sp_row.iloc[0]['batter_id'])
            for team in teams:
                opp_teams = [t for t in teams if t != team]
                if not opp_teams: continue
                opp_sp = team_sps.get(opp_teams[0], np.nan)
                idx = group[group['team_code'] == team].index
                lineup_df.loc[idx, 'pitcher_id'] = opp_sp

        # =================== STATCAST EVENT-LEVEL ENGINEERING ===================
        progress.progress(18, "Adding park/city/context and cleaning Statcast event data...")

        for col in ['batter_id', 'mlb_id', 'pitcher_id', 'team_code']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('.0','',regex=False).str.strip()

        # Add park/city/context from team code
        if 'home_team_code' in df.columns:
            df['team_code'] = df['home_team_code'].str.upper()
            df['park'] = df['home_team_code'].str.lower().str.replace(' ', '_')
        if 'home_team' in df.columns and 'park' not in df.columns:
            df['park'] = df['home_team'].str.lower().str.replace(' ', '_')
        if 'team_code' not in df.columns and 'park' in df.columns:
            park_to_team = {v:k for k,v in team_code_to_park.items()}
            df['team_code'] = df['park'].map(park_to_team).str.upper()
        df['team_code'] = df['team_code'].astype(str).str.upper()
        df['park'] = df['team_code'].map(team_code_to_park).str.lower()
        df['park_hr_rate'] = df['park'].map(park_hr_rate_map).fillna(1.0)
        df['park_altitude'] = df['park'].map(park_altitude_map).fillna(0)
        df['roof_status'] = df['park'].map(roof_status_map).fillna("open")
        df['city'] = df['team_code'].map(mlb_team_city_map).fillna("")

        # ====== Handedness fields ======
        if 'stand' in df.columns:
            df['batter_hand'] = df['stand']
        else:
            df['batter_hand'] = np.nan
        if 'p_throws' in df.columns:
            df['pitcher_hand'] = df['p_throws']
        elif 'p_throw' in df.columns:
            df['pitcher_hand'] = df['p_throw']
        else:
            df['pitcher_hand'] = np.nan

        # ==================== PARK HR PERCENT: DEEP RESEARCH BATTER SIDE ====================
        df['park_hr_pct_all'] = df['team_code'].map(park_hr_percent_map_all).fillna(1.0)
        df['park_hr_pct_rhb'] = df['team_code'].map(park_hr_percent_map_rhb).fillna(1.0)
        df['park_hr_pct_lhb'] = df['team_code'].map(park_hr_percent_map_lhb).fillna(1.0)
        df['park_hr_pct_hand'] = [
            park_hr_percent_map_rhb.get(team, 1.0) if str(stand).upper() == "R"
            else park_hr_percent_map_lhb.get(team, 1.0) if str(stand).upper() == "L"
            else park_hr_percent_map_all.get(team, 1.0)
            for team, stand in zip(df['team_code'], df['batter_hand'])
        ]

        # ==================== PARK HR PERCENT: DEEP RESEARCH PITCHER SIDE ====================
        # Requires pitcher_hand and pitcher_team_code for each event (as best as can be inferred)
        if 'pitcher_team_code' not in df.columns:
            # Try to infer pitcher_team_code as opponent of batting team if possible
            if 'opponent_team_code' in df.columns:
                df['pitcher_team_code'] = df['opponent_team_code']
            else:
                df['pitcher_team_code'] = np.nan
        df['pitcher_team_code'] = df['pitcher_team_code'].fillna(df['team_code'])
        df['pitcher_hand'] = df['pitcher_hand'].fillna("R")

        df['pitcher_park_hr_pct_all'] = df['pitcher_team_code'].map(pitcher_park_hr_percent_map_all).fillna(1.0)
        df['pitcher_park_hr_pct_rhp'] = df['pitcher_team_code'].map(pitcher_park_hr_percent_map_rhp).fillna(1.0)
        df['pitcher_park_hr_pct_lhp'] = df['pitcher_team_code'].map(pitcher_park_hr_percent_map_lhp).fillna(1.0)
        df['pitcher_park_hr_pct_hand'] = [
            pitcher_park_hr_percent_map_rhp.get(team, 1.0) if str(hand).upper() == "R"
            else pitcher_park_hr_percent_map_lhp.get(team, 1.0) if str(hand).upper() == "L"
            else pitcher_park_hr_percent_map_all.get(team, 1.0)
            for team, hand in zip(df['pitcher_team_code'], df['pitcher_hand'])
        ]

        # ========== SLG NUMERIC FOR ROLLING ==========
        slg_map = {'single':1, 'double':2, 'triple':3, 'home_run':4, 'homerun':4}
        if 'events' in df.columns:
            df['events_clean'] = df['events'].astype(str).str.lower().str.replace(' ', '')
        else:
            df['events_clean'] = ""
        df['slg_numeric'] = df['events_clean'].map(slg_map).astype(float)
        df['slg_numeric'] = df['slg_numeric'].fillna(0)
        if 'hr_outcome' not in df.columns:
            df['hr_outcome'] = df['events_clean'].isin(['homerun', 'home_run']).astype(int)

        # ========== FILTER VALID EVENTS ==========
        valid_events = [
            'single', 'double', 'triple', 'homerun', 'home_run', 'field_out',
            'force_out', 'grounded_into_double_play', 'fielders_choice_out',
            'pop_out', 'lineout', 'flyout', 'sac_fly', 'sac_fly_double_play'
        ]
        df = df[df['events_clean'].isin(valid_events)].copy()

        # ========== ADVANCED ROLLING FEATURES (INCL SPRAY ANGLE) ==========
        progress.progress(22, "Computing rolling Statcast features (batter & pitcher, incl spray angle)...")
        roll_windows = [3, 5, 7, 14, 20, 30, 60]
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

        # ====== Downcast numerics for RAM
        df = downcast_numeric(df)
        progress.progress(80, "Event-level feature engineering/merges complete.")

        # =================== OUTPUTS =======================
        st.success(f"Feature engineering complete! {len(df)} batted ball events.")
        st.markdown("#### Download Event-Level CSV / Parquet (all features, 1 row per batted ball event):")
        st.dataframe(df.head(20), use_container_width=True)
        st.download_button(
            "⬇️ Download Event-Level CSV",
            data=df.to_csv(index=False),
            file_name="event_level_hr_features.csv",
            key="download_event_level"
        )
        event_parquet = io.BytesIO()
        df.to_parquet(event_parquet, index=False)
        st.download_button(
            "⬇️ Download Event-Level Parquet",
            data=event_parquet.getvalue(),
            file_name="event_level_hr_features.parquet",
            mime="application/octet-stream",
            key="download_event_level_parquet"
        )

        # ================== TODAY CSV (BOTH BATTER & PITCHER PARK HR) ==================
        progress.progress(95, "Generating TODAY batter rows and context merges...")
        rolling_feature_cols = [col for col in df.columns if (
            col.startswith('b_') or col.startswith('p_')
        ) and any(str(w) in col for w in roll_windows)]
        extra_context_cols = [
            'park', 'park_hr_rate', 'park_hand_hr_rate', 'park_altitude', 'roof_status', 'city',
            'batter_hand', 'pitcher_hand',
            'park_hr_pct_all', 'park_hr_pct_rhb', 'park_hr_pct_lhb', 'park_hr_pct_hand',
            'pitcher_park_hr_pct_all', 'pitcher_park_hr_pct_rhp', 'pitcher_park_hr_pct_lhp', 'pitcher_park_hr_pct_hand'
        ]
        today_cols = [
            'game_date', 'batter_id', 'player_name', 'pitcher_id',
            'temp', 'humidity', 'wind_mph', 'wind_dir_string', 'condition', 'stand'
        ] + extra_context_cols + rolling_feature_cols

        pitcher_hand_map = {}
        pitcher_hand_statcast = df[['pitcher_id', 'pitcher_hand']].drop_duplicates().dropna()
        for _, row_p in pitcher_hand_statcast.iterrows():
            pid = str(row_p['pitcher_id'])
            hand = row_p['pitcher_hand']
            if pid not in pitcher_hand_map and pd.notna(hand):
                pitcher_hand_map[pid] = hand
        for _, row_p in lineup_df.dropna(subset=['pitcher_id']).drop_duplicates(['pitcher_id']).iterrows():
            pid = str(row_p['pitcher_id'])
            hand = row_p.get('p_throws') or row_p.get('stand') or row_p.get('pitcher_hand')
            if pid not in pitcher_hand_map and pd.notna(hand):
                pitcher_hand_map[pid] = hand

        today_rows = []
        for idx, row in lineup_df.iterrows():
            this_batter_id = str(row['batter_id']).split(".")[0]
            park = row.get("park", np.nan)
            city = row.get("city", np.nan)
            team_code = row.get("team_code", np.nan)
            game_date = row.get("game_date", np.nan)
            pitcher_id = str(row.get("pitcher_id", np.nan))
            player_name = row.get("player_name", np.nan)
            stand = row.get("stand", np.nan)
            filter_df = df[df['batter_id'].astype(str).str.split('.').str[0] == this_batter_id]
            if not filter_df.empty:
                last_row = filter_df.iloc[-1]
                row_out = {c: last_row.get(c, np.nan) for c in rolling_feature_cols + [
                    'batter_hand', 'park', 'park_hr_rate', 'park_hand_hr_rate', 'park_altitude', 'roof_status',
                    'city', 'pitcher_hand',
                    'park_hr_pct_all', 'park_hr_pct_rhb', 'park_hr_pct_lhb', 'park_hr_pct_hand',
                    'pitcher_park_hr_pct_all', 'pitcher_park_hr_pct_rhp', 'pitcher_park_hr_pct_lhp', 'pitcher_park_hr_pct_hand'
                ]}
            else:
                row_out = {c: np.nan for c in rolling_feature_cols + [
                    'batter_hand', 'park', 'park_hr_rate', 'park_hand_hr_rate', 'park_altitude', 'roof_status',
                    'city', 'pitcher_hand',
                    'park_hr_pct_all', 'park_hr_pct_rhb', 'park_hr_pct_lhb', 'park_hr_pct_hand',
                    'pitcher_park_hr_pct_all', 'pitcher_park_hr_pct_rhp', 'pitcher_park_hr_pct_lhp', 'pitcher_park_hr_pct_hand'
                ]}
            # Batter hand
            batter_hand = row.get('stand', row_out.get('batter_hand', np.nan))
            # Pitcher hand: always from pitcher_hand_map
            pitcher_hand = pitcher_hand_map.get(pitcher_id, np.nan)
            # Batter side park HR
            park_hand_rate = 1.0
            if not pd.isna(park) and not pd.isna(batter_hand):
                park_hand_rate = park_hand_hr_rate_map.get(str(park).lower(), {}).get(str(batter_hand).upper(), 1.0)
            # Deep research team/hand HR multipliers (BATTER SIDE)
            if not pd.isna(team_code):
                park_hr_pct_all = park_hr_percent_map_all.get(team_code, 1.0)
                park_hr_pct_rhb = park_hr_percent_map_rhb.get(team_code, 1.0)
                park_hr_pct_lhb = park_hr_percent_map_lhb.get(team_code, 1.0)
                if str(batter_hand).upper() == "R":
                    park_hr_pct_hand = park_hr_pct_rhb
                elif str(batter_hand).upper() == "L":
                    park_hr_pct_hand = park_hr_pct_lhb
                else:
                    park_hr_pct_hand = park_hr_pct_all
            else:
                park_hr_pct_all = park_hr_pct_rhb = park_hr_pct_lhb = park_hr_pct_hand = 1.0

            # --- PITCHER SIDE HR PARK FACTORS ---
            pitcher_team_code = row.get("pitcher_team_code", team_code)
            if pd.isna(pitcher_team_code):
                pitcher_team_code = team_code
            if pd.isna(pitcher_hand):
                pitcher_hand = "R"
            pitcher_park_hr_pct_all = pitcher_park_hr_percent_map_all.get(pitcher_team_code, 1.0)
            pitcher_park_hr_pct_rhp = pitcher_park_hr_percent_map_rhp.get(pitcher_team_code, 1.0)
            pitcher_park_hr_pct_lhp = pitcher_park_hr_percent_map_lhp.get(pitcher_team_code, 1.0)
            if str(pitcher_hand).upper() == "R":
                pitcher_park_hr_pct_hand = pitcher_park_hr_pct_rhp
            elif str(pitcher_hand).upper() == "L":
                pitcher_park_hr_pct_hand = pitcher_park_hr_pct_lhp
            else:
                pitcher_park_hr_pct_hand = pitcher_park_hr_pct_all

            row_out.update({
                "game_date": game_date,
                "batter_id": this_batter_id,
                "player_name": player_name,
                "pitcher_id": pitcher_id,
                "park": park,
                "park_hr_rate": park_hr_rate_map.get(str(park).lower(), 1.0) if not pd.isna(park) else 1.0,
                "park_hand_hr_rate": park_hand_rate,
                "park_altitude": park_altitude_map.get(str(park).lower(), 0) if not pd.isna(park) else 0,
                "roof_status": roof_status_map.get(str(park).lower(), "open") if not pd.isna(park) else "open",
                "city": city if not pd.isna(city) else mlb_team_city_map.get(team_code, ""),
                "stand": batter_hand,
                "batter_hand": batter_hand,
                "pitcher_hand": pitcher_hand,
                "park_hr_pct_all": park_hr_pct_all,
                "park_hr_pct_rhb": park_hr_pct_rhb,
                "park_hr_pct_lhb": park_hr_pct_lhb,
                "park_hr_pct_hand": park_hr_pct_hand,
                "pitcher_park_hr_pct_all": pitcher_park_hr_pct_all,
                "pitcher_park_hr_pct_rhp": pitcher_park_hr_pct_rhp,
                "pitcher_park_hr_pct_lhp": pitcher_park_hr_pct_lhp,
                "pitcher_park_hr_pct_hand": pitcher_park_hr_pct_hand
            })
            for c in ['temp', 'humidity', 'wind_mph', 'wind_dir_string', 'condition']:
                row_out[c] = row.get(c, np.nan)
            today_rows.append(row_out)

        today_df = pd.DataFrame(today_rows, columns=today_cols)
        today_df = dedup_columns(today_df)
        today_df = downcast_numeric(today_df)

        st.markdown("#### Download TODAY CSV / Parquet (1 row per batter, matchup, rolling features & weather):")
        st.dataframe(today_df.head(20), use_container_width=True)
        st.download_button(
            "⬇️ Download TODAY CSV",
            data=today_df.to_csv(index=False),
            file_name="today_hr_features.csv",
            key="download_today_csv"
        )
        today_parquet = io.BytesIO()
        today_df.to_parquet(today_parquet, index=False)
        st.download_button(
            "⬇️ Download TODAY Parquet",
            data=today_parquet.getvalue(),
            file_name="today_hr_features.parquet",
            mime="application/octet-stream",
            key="download_today_parquet"
        )
        st.success("All files and debug outputs ready.")
        progress.progress(100, "All complete.")

        # ---- RAM Cleanup ----
        del df, today_df, batter_event, pitcher_event, event_parquet, today_parquet
        gc.collect()

    else:
        st.info("Upload a Matchups/Lineups CSV and select a date range to generate the event-level and TODAY CSVs.")
