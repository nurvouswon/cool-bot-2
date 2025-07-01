import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import gc
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
park_hand_hr_rate_map = {
    'angels_stadium': {'L': 1.09, 'R': 1.02}, 'angel_stadium': {'L': 1.09, 'R': 1.02},
    'minute_maid_park': {'L': 1.13, 'R': 1.06}, 'coors_field': {'L': 1.38, 'R': 1.24},
    'yankee_stadium': {'L': 1.47, 'R': 0.98}, 'fenway_park': {'L': 1.04, 'R': 0.97},
    'rogers_centre': {'L': 1.08, 'R': 1.12}, 'tropicana_field': {'L': 0.84, 'R': 0.89},
    'camden_yards': {'L': 0.98, 'R': 1.27}, 'guaranteed_rate_field': {'L': 1.25, 'R': 1.11},
    'progressive_field': {'L': 0.99, 'R': 1.02}, 'comerica_park': {'L': 1.10, 'R': 0.91},
    'kauffman_stadium': {'L': 0.90, 'R': 1.03}, 'globe_life_field': {'L': 1.01, 'R': 0.98},
    'dodger_stadium': {'L': 1.02, 'R': 1.18}, 'oakland_coliseum': {'L': 0.81, 'R': 0.85},
    't-mobile_park': {'L': 0.81, 'R': 0.92}, 'tmobile_park': {'L': 0.81, 'R': 0.92},
    'oracle_park': {'L': 0.67, 'R': 0.99}, 'wrigley_field': {'L': 1.10, 'R': 1.16},
    'great_american_ball_park': {'L': 1.30, 'R': 1.23}, 'american_family_field': {'L': 1.25, 'R': 1.13},
    'pnc_park': {'L': 0.76, 'R': 0.92}, 'busch_stadium': {'L': 0.78, 'R': 0.91},
    'truist_park': {'L': 1.00, 'R': 1.09}, 'loan_depot_park': {'L': 0.83, 'R': 0.91},
    'loandepot_park': {'L': 0.83, 'R': 0.91}, 'citi_field': {'L': 1.11, 'R': 0.98},
    'nationals_park': {'L': 1.04, 'R': 1.06}, 'petco_park': {'L': 0.90, 'R': 0.88},
    'chase_field': {'L': 1.16, 'R': 1.05}, 'citizens_bank_park': {'L': 1.22, 'R': 1.20},
    'sutter_health_park': {'L': 1.12, 'R': 1.12}, 'target_field': {'L': 1.09, 'R': 1.01}
}

# =================== DEDUPLICATION AND TYPE HELPERS ===================
def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

def downcast_numeric(df):
    for col in df.select_dtypes(include=['float']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

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

# =================== HANDEDNESS AND PLAYER INFO (pybaseball) ===================
@st.cache_data(show_spinner=False)
def fetch_mlb_handness_maps():
    # This can be expanded, but keeps things simple for speed (300 players = fast)
    try:
        from pybaseball import playerid_lookup, batting_stats_bref, pitching_stats_bref
        # Batters (should always have stand info)
        bat_df = batting_stats_bref('2023')
        hand_map = {}
        for i, r in bat_df.iterrows():
            pid = str(r.get('IDfg', ''))
            if pid:
                hand_map[pid] = r.get('Bats', '').upper()
        # Pitchers
        pit_df = pitching_stats_bref('2023')
        p_hand_map = {}
        for i, r in pit_df.iterrows():
            pid = str(r.get('IDfg', ''))
            if pid:
                p_hand_map[pid] = r.get('Throws', '').upper()
        return hand_map, p_hand_map
    except Exception as e:
        st.warning(f"pybaseball handedness fetch failed: {e}")
        return {}, {}

# =================== ADVANCED ROLLING STATS ===================
@st.cache_data(show_spinner=True)
def fast_rolling_stats_deep(df, id_col, date_col, windows, pitch_types=None, prefix="", hand_col=None, pitch_hand_col=None):
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
    # Batted ball direction: pull/center/oppo
    if 'hc_x' in df.columns:
        df['bb_direction'] = np.where(df['hc_x'] < 125, 'pull', np.where(df['hc_x'] > 375, 'oppo', 'center'))
    else:
        df['bb_direction'] = np.nan

    results = []
    for name, group in df.groupby(id_col, sort=False):
        out_row = {}
        ls = group['launch_speed'] if 'launch_speed' in group.columns else None
        la = group['launch_angle'] if 'launch_angle' in group.columns else None
        if hand_col and hand_col in group.columns:
            hand_splits = {h: group[group[hand_col] == h] for h in ['L', 'R', 'S']}
        else:
            hand_splits = {}
        for w in windows:
            # Standard Statcast rolling
            if ls is not None:
                out_row[f"{prefix}avg_exit_velo_{w}"] = ls.rolling(w, min_periods=1).mean().iloc[-1]
                out_row[f"{prefix}hard_hit_rate_{w}"] = ls.rolling(w, min_periods=1).apply(lambda x: np.mean(x >= 95)).iloc[-1]
            if la is not None:
                out_row[f"{prefix}fb_rate_{w}"] = la.rolling(w, min_periods=1).apply(lambda x: np.mean(x >= 25)).iloc[-1]
                out_row[f"{prefix}sweet_spot_rate_{w}"] = la.rolling(w, min_periods=1).apply(lambda x: np.mean((x >= 8) & (x <= 32))).iloc[-1]
            if ls is not None and la is not None:
                out_row[f"{prefix}barrel_rate_{w}"] = (((ls >= 98) & (la >= 26) & (la <= 30)).rolling(w, min_periods=1).mean().iloc[-1])
            # Advanced
            if 'bb_direction' in group.columns:
                pull = (group['bb_direction'] == 'pull').rolling(w, min_periods=1).mean().iloc[-1]
                oppo = (group['bb_direction'] == 'oppo').rolling(w, min_periods=1).mean().iloc[-1]
                out_row[f"{prefix}pull_percent_{w}"] = pull
                out_row[f"{prefix}oppo_percent_{w}"] = oppo
            if 'events_clean' in group.columns:
                hr_rate = (group['events_clean'].isin(['homerun', 'home_run']).rolling(w, min_periods=1).mean().iloc[-1])
                out_row[f"{prefix}hr_rate_{w}"] = hr_rate
                # Slugging/ISO
                tb_map = {'single': 1, 'double': 2, 'triple': 3, 'homerun': 4, 'home_run': 4}
                tb = group['events_clean'].map(tb_map).rolling(w, min_periods=1).sum().iloc[-1]
                ab = len(group['events_clean'].iloc[-w:]) if len(group) >= w else len(group)
                out_row[f"{prefix}slg_{w}"] = tb / ab if ab > 0 else 0
                out_row[f"{prefix}iso_{w}"] = (tb - (group['events_clean'] == 'single').rolling(w, min_periods=1).sum().iloc[-1]) / ab if ab > 0 else 0
            # Pitch type splits
            if pitch_types is not None and "pitch_type" in group.columns:
                for pt in pitch_types:
                    pt_group = group[group['pitch_type'] == pt]
                    for stat, suffix, func in [
                        ('launch_speed', 'avg_exit_velo', lambda x: x.mean()),
                        ('launch_speed', 'hard_hit_rate', lambda x: np.mean(x >= 95)),
                        ('launch_angle', 'fb_rate', lambda x: np.mean(x >= 25)),
                        ('launch_angle', 'sweet_spot_rate', lambda x: np.mean((x >= 8) & (x <= 32))),
                        # For barrels:
                        ('', 'barrel_rate', lambda x: np.nan), # handled below
                    ]:
                        for ww in [w]:
                            colname = f"{prefix}{pt}_{suffix}_{ww}"
                            if stat and stat in pt_group.columns and not pt_group.empty:
                                vals = pt_group[stat].iloc[-ww:]
                                out_row[colname] = func(vals) if len(vals) > 0 else np.nan
                            elif suffix == 'barrel_rate' and not pt_group.empty and \
                                    'launch_speed' in pt_group.columns and 'launch_angle' in pt_group.columns:
                                ls = pt_group['launch_speed'].iloc[-ww:]
                                la = pt_group['launch_angle'].iloc[-ww:]
                                out_row[colname] = np.mean((ls >= 98) & (la >= 26) & (la <= 30)) if len(ls) > 0 and len(la) > 0 else np.nan
                            else:
                                out_row[colname] = np.nan
            # Handedness splits
            if hand_splits:
                for hand, hand_group in hand_splits.items():
                    if hand_group.empty:
                        continue
                    # You can repeat the above for each hand split, using e.g. prefix=f"{prefix}{hand.lower()}_"
                    for stat, suffix, func in [
                        ('launch_speed', 'avg_exit_velo', lambda x: x.mean()),
                        ('launch_speed', 'hard_hit_rate', lambda x: np.mean(x >= 95)),
                        ('launch_angle', 'fb_rate', lambda x: np.mean(x >= 25)),
                        ('launch_angle', 'sweet_spot_rate', lambda x: np.mean((x >= 8) & (x <= 32))),
                        ('', 'barrel_rate', lambda x: np.nan),
                    ]:
                        for ww in [w]:
                            colname = f"{prefix}{hand.lower()}_{suffix}_{ww}"
                            if stat and stat in hand_group.columns and not hand_group.empty:
                                vals = hand_group[stat].iloc[-ww:]
                                out_row[colname] = func(vals) if len(vals) > 0 else np.nan
                            elif suffix == 'barrel_rate' and not hand_group.empty and \
                                    'launch_speed' in hand_group.columns and 'launch_angle' in hand_group.columns:
                                ls = hand_group['launch_speed'].iloc[-ww:]
                                la = hand_group['launch_angle'].iloc[-ww:]
                                out_row[colname] = np.mean((ls >= 98) & (la >= 26) & (la <= 30)) if len(ls) > 0 and len(la) > 0 else np.nan
                            else:
                                out_row[colname] = np.nan
        out_row[id_col] = name
        results.append(out_row)
    return pd.DataFrame(results)

# ======================== STREAMLIT APP START ========================
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
        hand_map, p_hand_map = fetch_mlb_handness_maps()

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
        # Map batter_id and player_name for flexibility
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
        grouped = lineup_df.groupby(['game_date', 'park', 'time'])
        for (gdate, park, time_), group in grouped:
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

        # ==== STATCAST EVENT-LEVEL ENGINEERING ====
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

        # --- Add batter and pitcher handedness ---
        df['batter_hand'] = df['batter_id'].map(hand_map)
        df['pitcher_hand'] = df['pitcher_id'].map(p_hand_map)

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

        # Rolling stat features (batters and pitchers, ALL deep splits)
        progress.progress(22, "Computing rolling Statcast features (batter & pitcher & splits)...")
        roll_windows = [3, 5, 7, 14, 20]
        main_pitch_types = ["ff", "sl", "cu", "ch", "si", "fc", "fs", "st", "sinker", "splitter", "sweeper"]
        for col in ['batter', 'batter_id']:
            if col in df.columns:
                df['batter_id'] = df[col]
        for col in ['pitcher', 'pitcher_id']:
            if col in df.columns:
                df['pitcher_id'] = df[col]
        batter_event = fast_rolling_stats_deep(
            df, "batter_id", "game_date", roll_windows, pitch_types=main_pitch_types,
            prefix="b_", hand_col='batter_hand'
        )
        if not batter_event.empty:
            batter_event = batter_event.set_index('batter_id')
        df_for_pitchers = df.copy()
        if 'batter_id' in df_for_pitchers.columns:
            df_for_pitchers = df_for_pitchers.drop(columns=['batter_id'])
        df_for_pitchers = df_for_pitchers.rename(columns={"pitcher_id": "batter_id"})
        pitcher_event = fast_rolling_stats_deep(
            df_for_pitchers, "batter_id", "game_date", roll_windows, pitch_types=main_pitch_types,
            prefix="p_", hand_col='pitcher_hand'
        )
        if not pitcher_event.empty:
            pitcher_event = pitcher_event.set_index('batter_id')
        df = pd.merge(df, batter_event.reset_index(), how="left", left_on="batter_id", right_on="batter_id")
        df = pd.merge(df, pitcher_event.reset_index(), how="left", left_on="pitcher_id", right_on="batter_id", suffixes=('', '_pitcherstat'))
        if 'batter_id_pitcherstat' in df.columns:
            df = df.drop(columns=['batter_id_pitcherstat'])
        df = dedup_columns(df)

        # ====== Add park_hand_hr_rate to event-level ======
        if 'batter_hand' in df.columns and 'park' in df.columns:
            df['park_hand_hr_rate'] = [
                park_hand_hr_rate_map.get(str(park).lower(), {}).get(str(stand).upper(), 1.0)
                for park, stand in zip(df['park'], df['batter_hand'])
            ]
        else:
            df['park_hand_hr_rate'] = 1.0

        # Downcast numerics for RAM
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

        # ===== TODAY CSV: 1 row per batter with all rolling/context features and WEATHER FROM LINEUP CSV =====
        progress.progress(95, "Generating TODAY batter rows and context merges...")
        rolling_feature_cols = [col for col in df.columns if (
            col.startswith('b_') or col.startswith('p_')
        ) and any(str(w) in col for w in roll_windows)]
        extra_context_cols = [
            'park', 'park_hr_rate', 'park_hand_hr_rate', 'park_altitude', 'roof_status', 'city', 'batter_hand', 'pitcher_hand'
        ]
        today_cols = [
            'game_date', 'batter_id', 'player_name', 'pitcher_id',
            'temp', 'humidity', 'wind_mph', 'wind_dir_string', 'condition', 'batter_hand', 'pitcher_hand'
        ] + extra_context_cols + rolling_feature_cols

        today_rows = []
        for idx, row in lineup_df.iterrows():
            this_batter_id = str(row['batter_id']).split(".")[0]
            park = row.get("park", np.nan)
            city = row.get("city", np.nan)
            team_code = row.get("team_code", np.nan)
            game_date = row.get("game_date", np.nan)
            pitcher_id = row.get("pitcher_id", np.nan)
            player_name = row.get("player_name", np.nan)
            stand = hand_map.get(this_batter_id, np.nan)
            pitch_hand = p_hand_map.get(str(pitcher_id), np.nan)
            filter_df = df[df['batter_id'].astype(str).str.split('.').str[0] == this_batter_id]
            if not filter_df.empty:
                last_row = filter_df.iloc[-1]
                row_out = {c: last_row.get(c, np.nan) for c in rolling_feature_cols}
            else:
                row_out = {c: np.nan for c in rolling_feature_cols}
            row_out.update({
                "game_date": game_date,
                "batter_id": this_batter_id,
                "player_name": player_name,
                "pitcher_id": pitcher_id,
                "park": park,
                "park_hr_rate": park_hr_rate_map.get(str(park).lower(), 1.0) if not pd.isna(park) else 1.0,
                "park_hand_hr_rate": park_hand_hr_rate_map.get(str(park).lower(), {}).get(str(stand).upper(), 1.0) if not pd.isna(park) and not pd.isna(stand) else 1.0,
                "park_altitude": park_altitude_map.get(str(park).lower(), 0) if not pd.isna(park) else 0,
                "roof_status": roof_status_map.get(str(park).lower(), "open") if not pd.isna(park) else "open",
                "city": city if not pd.isna(city) else mlb_team_city_map.get(team_code, ""),
                "batter_hand": stand,
                "pitcher_hand": pitch_hand
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

# --- Tab2 placeholder, unchanged from your app ---
with tab2:
    st.info("Upload the generated files in Tab 2 for downstream analysis.")
