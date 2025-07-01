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

def downcast_numeric(df):
    for col in df.select_dtypes(include=['float']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

# ================== ADVANCED ROLLING STATS WITH ALL FEATURES ================
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
    if 'hc_x' in df.columns and 'hc_y' in df.columns:
        df['pull'] = ((df['hc_x'] > 200) & (df['hc_x'] < 300)).astype(float)
    if 'hit_distance_sc' in df.columns:
        df['hard_hit'] = (df['launch_speed'] >= 95).astype(float)
    # Add FB, HR flags
    if 'launch_angle' in df.columns:
        df['fb'] = (df['launch_angle'] >= 25).astype(float)
        df['sweet_spot'] = ((df['launch_angle'] >= 8) & (df['launch_angle'] <= 32)).astype(float)
    if 'events_clean' in df.columns:
        df['hr'] = df['events_clean'].isin(['home_run', 'homerun']).astype(float)
    results = []
    for name, group in df.groupby(id_col, sort=False):
        out_row = {}
        ls = group['launch_speed'] if 'launch_speed' in group.columns else None
        la = group['launch_angle'] if 'launch_angle' in group.columns else None
        hd = group['hit_distance_sc'] if 'hit_distance_sc' in group.columns else None
        pull = group['pull'] if 'pull' in group.columns else None
        hard = group['hard_hit'] if 'hard_hit' in group.columns else None
        hr = group['hr'] if 'hr' in group.columns else None
        fb = group['fb'] if 'fb' in group.columns else None
        sweet = group['sweet_spot'] if 'sweet_spot' in group.columns else None
        slg_map = {'single':1,'double':2,'triple':3,'home_run':4,'homerun':4}
        for w in windows:
            if ls is not None:
                out_row[f"{prefix}avg_exit_velo_{w}"] = ls.rolling(w, min_periods=1).mean().iloc[-1]
                out_row[f"{prefix}hard_hit_rate_{w}"] = hard.rolling(w, min_periods=1).mean().iloc[-1] if hard is not None else np.nan
            if la is not None:
                out_row[f"{prefix}fb_rate_{w}"] = fb.rolling(w, min_periods=1).mean().iloc[-1] if fb is not None else np.nan
                out_row[f"{prefix}sweet_spot_rate_{w}"] = sweet.rolling(w, min_periods=1).mean().iloc[-1] if sweet is not None else np.nan
            if ls is not None and la is not None:
                barrel_mask = (ls >= 98) & (la >= 26) & (la <= 30)
                out_row[f"{prefix}barrel_rate_{w}"] = barrel_mask.rolling(w, min_periods=1).mean().iloc[-1]
            if hd is not None:
                out_row[f"{prefix}hit_dist_avg_{w}"] = hd.rolling(w, min_periods=1).mean().iloc[-1]
            if pull is not None:
                out_row[f"{prefix}pull_rate_{w}"] = pull.rolling(w, min_periods=1).mean().iloc[-1]
            # HR/FB
            if hr is not None and fb is not None and fb.sum() > 0:
                hr_fb = hr.rolling(w, min_periods=1).sum() / fb.rolling(w, min_periods=1).sum()
                out_row[f"{prefix}hr_fb_{w}"] = hr_fb.iloc[-1] if not hr_fb.empty else np.nan
            # SLG (needs events)
            if 'events_clean' in group.columns:
                evs = group['events_clean']
                slg_events = evs.rolling(w, min_periods=1).apply(
                    lambda evs: np.mean([slg_map.get(e,0) for e in evs]), raw=False)
                out_row[f"{prefix}slg_{w}"] = slg_events.iloc[-1] if not slg_events.empty else np.nan
        out_row[id_col] = name
        results.append(out_row)
    return pd.DataFrame(results)

# =============== STREAMLIT APP LAYOUT START ====================
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

        # ==== Assign Opposing SP for Each Batter (handles game_number bug/fallback) ====
        progress.progress(14, "Assigning opposing pitcher for each batter in lineup...")
        lineup_df['pitcher_id'] = np.nan
        lineup_df['pitcher_hand'] = np.nan
        grouped = lineup_df.groupby(['game_date', 'park', 'time']) if 'time' in lineup_df.columns else lineup_df.groupby(['game_date', 'park'])
        for group_key, group in grouped:
            if 'team_code' not in group.columns: continue
            teams = group['team_code'].unique()
            if len(teams) < 2: continue
            # Identify SP for each team
            team_sps = {}
            team_sp_hand = {}
            for team in teams:
                sp_row = group[(group['team_code'] == team) & (group['batting_order'] == "SP")]
                if not sp_row.empty:
                    team_sps[team] = str(sp_row.iloc[0]['batter_id'])
                    if 'stand' in sp_row.columns:
                        team_sp_hand[team] = sp_row.iloc[0]['stand']
                    elif 'hand' in sp_row.columns:
                        team_sp_hand[team] = sp_row.iloc[0]['hand']
                    else:
                        team_sp_hand[team] = np.nan
            # Assign each batter the opponent's SP and SP's handedness
            for team in teams:
                opp_teams = [t for t in teams if t != team]
                if not opp_teams: continue
                opp_sp = team_sps.get(opp_teams[0], np.nan)
                opp_hand = team_sp_hand.get(opp_teams[0], np.nan)
                idx = group[group['team_code'] == team].index
                lineup_df.loc[idx, 'pitcher_id'] = opp_sp
                lineup_df.loc[idx, 'pitcher_hand'] = opp_hand

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

        # HR outcome flag and events_clean
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
        progress.progress(22, "Computing rolling Statcast features (batter & pitcher, pitch type, deep windows)...")
        roll_windows = [3, 5, 7, 14, 20, 30, 60]
        main_pitch_types = ["ff", "sl", "cu", "ch", "si", "fc", "fs", "st", "sinker", "splitter", "sweeper"]
        for col in ['batter', 'batter_id']:
            if col in df.columns:
                df['batter_id'] = df[col]
        for col in ['pitcher', 'pitcher_id']:
            if col in df.columns:
                df['pitcher_id'] = df[col]

        # ========== ADVANCED ROLLING FEATURES ==========
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

        # ====== Add park_hand_hr_rate to event-level ======
        if 'stand' in df.columns and 'park' in df.columns:
            df['park_hand_hr_rate'] = [
                park_hand_hr_rate_map.get(str(park).lower(), {}).get(str(stand).upper(), 1.0)
                for park, stand in zip(df['park'], df['stand'])
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
        # Also add advanced fields if needed
        extra_context_cols = [
            'park', 'park_hr_rate', 'park_hand_hr_rate', 'park_altitude', 'roof_status', 'city',
            'batter_hand', 'pitcher_hand'
        ]
        today_cols = [
            'game_date', 'batter_id', 'player_name', 'pitcher_id',
            'temp', 'humidity', 'wind_mph', 'wind_dir_string', 'condition', 'stand'
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
            stand = row.get("stand", np.nan)
            batter_hand = row.get("stand", np.nan)
            pitcher_hand = row.get("pitcher_hand", np.nan)
            filter_df = df[df['batter_id'].astype(str).str.split('.').str[0] == this_batter_id]
            if not filter_df.empty:
                last_row = filter_df.iloc[-1]
                row_out = {c: last_row.get(c, np.nan) for c in rolling_feature_cols}
            else:
                row_out = {c: np.nan for c in rolling_feature_cols}
            # Context
            row_out.update({
                "game_date": game_date,
                "batter_id": this_batter_id,
                "player_name": player_name,
                "pitcher_id": pitcher_id,
                "park": park,
                "park_hr_rate": park_hr_rate_map.get(str(park).lower(), 1.0) if not pd.isna(park) else 1.0,
                "park_hand_hr_rate": park_hand_hr_rate_map.get(str(park).lower(), {}).get(str(batter_hand).upper(), 1.0) if not pd.isna(park) and not pd.isna(batter_hand) else 1.0,
                "park_altitude": park_altitude_map.get(str(park).lower(), 0) if not pd.isna(park) else 0,
                "roof_status": roof_status_map.get(str(park).lower(), "open") if not pd.isna(park) else "open",
                "city": city if not pd.isna(city) else mlb_team_city_map.get(team_code, ""),
                "stand": stand,
                "batter_hand": batter_hand,
                "pitcher_hand": pitcher_hand
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
