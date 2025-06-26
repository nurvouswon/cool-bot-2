import streamlit as st
import pandas as pd
import numpy as np
import re
import time
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

# ========== APP HEADER ==========
st.set_page_config("🟦 Generate Today's Event-Level CSV", layout="wide")
st.title("🟦 Generate Today's Event-Level CSV (All Features, Full Debug)")

@st.cache_data(show_spinner=True)
def read_csv(file):
    return pd.read_csv(file)

@st.cache_data(show_spinner=True)
def parse_weather_fields(df):
    if "weather" in df.columns:
        weather_str = df["weather"].astype(str)
        df["temp"] = weather_str.str.extract(r'(\d{2,3})\s*O', expand=False)
        df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
        wind_mph = weather_str.str.extract(r'(\d+)\s*-\s*(\d+)', expand=True)
        df["wind_mph"] = pd.to_numeric(wind_mph[0], errors="coerce")
        df["wind_mph"] = np.where(wind_mph[1].notnull(),
                                  0.5*(pd.to_numeric(wind_mph[0], errors='coerce') +
                                       pd.to_numeric(wind_mph[1], errors='coerce')),
                                  df["wind_mph"])
        df["wind_mph"] = df["wind_mph"].fillna(weather_str.str.extract(r'(\d+)\s*mph', expand=False)).astype(float)
        df["wind_dir"] = weather_str.str.extract(r'(?:mph\s+)?([nswecf]{1,2})', flags=re.I, expand=False)
        df["condition"] = weather_str.str.extract(r'(indoor|outdoor)', flags=re.I, expand=False)
    return df

@st.cache_data(show_spinner=True)
def fast_rolling_stats(df, id_col, date_col, windows, pitch_types=None, prefix=""):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['launch_speed'] = pd.to_numeric(df['launch_speed'], errors='coerce')
    df['launch_angle'] = pd.to_numeric(df['launch_angle'], errors='coerce')
    if 'pitch_type' in df.columns:
        df['pitch_type'] = df['pitch_type'].astype(str).str.lower().str.strip()
    df = df.drop_duplicates(subset=[id_col, date_col], keep='last')  # Dedup
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
        # Per pitch type
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

# --- FEATURE COLUMN LIST ---
all_feature_cols = [
    "team_code","game_date","game_number","mlb_id","player_name","batting_order","position",
    "weather","time","stadium","city","batter_id","p_throws",
    "pitcher_id",
    "hard_hit_rate_20","sweet_spot_rate_20",
    "park_hand_hr_7","park_hand_hr_14","park_hand_hr_30",
    "b_vsp_hand_hr_3","p_vsb_hand_hr_3","b_vsp_hand_hr_5","p_vsb_hand_hr_5",
    "b_vsp_hand_hr_7","p_vsb_hand_hr_7","b_vsp_hand_hr_14","p_vsb_hand_hr_14",
    "b_pitchtype_hr_3","p_pitchtype_hr_3","b_pitchtype_hr_5","p_pitchtype_hr_5",
    "b_pitchtype_hr_7","p_pitchtype_hr_7","b_pitchtype_hr_14","p_pitchtype_hr_14",
    "b_launch_speed_3","b_launch_speed_5","b_launch_speed_7","b_launch_speed_14",
    "b_launch_angle_3","b_launch_angle_5","b_launch_angle_7","b_launch_angle_14",
    "b_hit_distance_sc_3","b_hit_distance_sc_5","b_hit_distance_sc_7","b_hit_distance_sc_14",
    "b_woba_value_3","b_woba_value_5","b_woba_value_7","b_woba_value_14",
    "b_release_speed_3","b_release_speed_5","b_release_speed_7","b_release_speed_14",
    "b_release_spin_rate_3","b_release_spin_rate_5","b_release_spin_rate_7","b_release_spin_rate_14",
    "b_spin_axis_3","b_spin_axis_5","b_spin_axis_7","b_spin_axis_14",
    "b_pfx_x_3","b_pfx_x_5","b_pfx_x_7","b_pfx_x_14",
    "b_pfx_z_3","b_pfx_z_5","b_pfx_z_7","b_pfx_z_14",
    "p_launch_speed_3","p_launch_speed_5","p_launch_speed_7","p_launch_speed_14",
    "p_launch_angle_3","p_launch_angle_5","p_launch_angle_7","p_launch_angle_14",
    "p_hit_distance_sc_3","p_hit_distance_sc_5","p_hit_distance_sc_7","p_hit_distance_sc_14",
    "p_woba_value_3","p_woba_value_5","p_woba_value_7","p_woba_value_14",
    "p_release_speed_3","p_release_speed_5","p_release_speed_7","p_release_speed_14",
    "p_release_spin_rate_3","p_release_spin_rate_5","p_release_spin_rate_7","p_release_spin_rate_14",
    "p_spin_axis_3","p_spin_axis_5","p_spin_axis_7","p_spin_axis_14",
    "p_pfx_x_3","p_pfx_x_5","p_pfx_x_7","p_pfx_x_14",
    "p_pfx_z_3","p_pfx_z_5","p_pfx_z_7","p_pfx_z_14",
    "park","park_hr_rate","park_altitude","roof_status",
    "temp","wind_mph","wind_dir","humidity","condition","hr_prob"
]

# =========================
# ===== MAIN APP LOGIC ====
# =========================

today_file = st.file_uploader("Upload Today's Matchups/Lineups CSV", type=["csv"], key="today_csv")
hist_file = st.file_uploader("Upload Historical Event-Level CSV", type=["csv"], key="hist_csv")

if today_file and hist_file:
    st.info("Loaded today's matchups and historical event data.")
    df_today = read_csv(today_file)
    df_hist = read_csv(hist_file)

    st.write("Today's Data Sample:", df_today.head(2))
    st.write("Historical Data Sample:", df_hist.head(2))

    # Clean column names
    df_today.columns = [str(c).strip().lower().replace(" ", "_") for c in df_today.columns]
    df_hist.columns = [str(c).strip().lower().replace(" ", "_") for c in df_hist.columns]

    for col in ['batter_id', 'mlb_id', 'pitcher_id']:
        if col in df_today.columns:
            df_today[col] = df_today[col].astype(str).str.replace('.0','',regex=False).str.strip()
        if col in df_hist.columns:
            df_hist[col] = df_hist[col].astype(str).str.replace('.0','',regex=False).str.strip()

    # ========== CONTEXT MAP LOGIC (FOR PARK, ALT, ROOF, ETC) ==========
    if 'home_team_code' in df_hist.columns:
        df_hist['home_team_code'] = df_hist['home_team_code'].astype(str).str.upper()
    if 'park' not in df_hist.columns and 'home_team_code' in df_hist.columns:
        df_hist['park'] = df_hist['home_team_code'].map(team_code_to_park)
    if 'park' in df_hist.columns:
        df_hist['park_hr_rate'] = df_hist['park'].map(park_hr_rate_map).fillna(1.0)
        df_hist['park_altitude'] = df_hist['park'].map(park_altitude_map).fillna(0)
        df_hist['roof_status'] = df_hist['park'].map(roof_status_map).fillna("open")
    if 'stadium' in df_hist.columns and 'park' not in df_hist.columns:
        df_hist['park'] = df_hist['stadium'].str.lower().str.replace(' ', '_')

    # ========== WEATHER (if present) ==========
    df_hist = parse_weather_fields(df_hist)
    if 'wind_dir' in df_hist.columns:
        df_hist['wind_dir_angle'] = df_hist['wind_dir'].apply(wind_dir_to_angle)
        df_hist['wind_dir_sin'] = np.sin(np.deg2rad(df_hist['wind_dir_angle']))
        df_hist['wind_dir_cos'] = np.cos(np.deg2rad(df_hist['wind_dir_angle']))
    if 'stand' in df_hist.columns and 'wind_dir_angle' in df_hist.columns:
        def relative_wind_angle(row):
            try:
                if row['stand'] == 'L':
                    return (row['wind_dir_angle'] - 45) % 360
                else:
                    return (row['wind_dir_angle'] - 135) % 360
            except Exception:
                return np.nan
        df_hist['relative_wind_angle'] = df_hist.apply(relative_wind_angle, axis=1)
        df_hist['relative_wind_sin'] = np.sin(np.deg2rad(df_hist['relative_wind_angle']))
        df_hist['relative_wind_cos'] = np.cos(np.deg2rad(df_hist['relative_wind_angle']))

    # ========== ADVANCED ROLLING FEATURE ENGINEERING ==========
    roll_windows = [3, 5, 7, 14, 20]
    main_pitch_types = ["ff", "sl", "cu", "ch", "si", "fc", "fs", "st", "sinker", "splitter", "sweeper"]

    st.write("Running fast_rolling_stats for batters...")
    batter_event = fast_rolling_stats(df_hist, "batter_id", "game_date", roll_windows, main_pitch_types, prefix="")
    st.write("Batter rolling event stats sample:", batter_event.head(3))
    batter_event = batter_event.set_index('batter_id')
    batter_event = batter_event[~batter_event.index.duplicated(keep='last')]

    st.write("Running fast_rolling_stats for pitchers...")
    pitcher_event = pd.DataFrame()
    df_hist_for_pitcher = df_hist.copy()
    if 'batter_id' in df_hist_for_pitcher.columns:
        df_hist_for_pitcher = df_hist_for_pitcher.drop(columns=['batter_id'])
    if 'pitcher_id' in df_hist.columns:
        pitcher_event = fast_rolling_stats(
            df_hist_for_pitcher.rename(columns={"pitcher_id":"batter_id"}),
            "batter_id", "game_date", roll_windows, main_pitch_types, prefix="p_"
        )
    elif 'mlb_id' in df_hist.columns:
        pitcher_event = fast_rolling_stats(
            df_hist.rename(columns={"mlb_id":"batter_id"}),
            "batter_id", "game_date", roll_windows, main_pitch_types, prefix="p_"
        )
    if not pitcher_event.empty:
        pitcher_event = pitcher_event.set_index('batter_id')
        pitcher_event = pitcher_event[~pitcher_event.index.duplicated(keep='last')]
        st.write("Pitcher rolling stats sample:", pitcher_event.head(3))

    # ========== OPPONENT PITCHER ASSIGNMENT ==========
    # Robust: assign opponent SP for each team, each game, to all hitters for that team
    games = df_today[['game_date', 'game_number']].drop_duplicates()
    opp_pitcher_map = {}
    for _, game in games.iterrows():
        game_date, game_number = game['game_date'], game['game_number']
        teams = df_today[
            (df_today['game_date'] == game_date) &
            (df_today['game_number'] == game_number)
        ]['team_code'].unique()
        for team in teams:
            opp_team = [t for t in teams if t != team]
            if not opp_team:
                continue
            opp_team = opp_team[0]
            opp_sp = df_today[
                (df_today['team_code'] == opp_team) &
                (df_today['game_date'] == game_date) &
                (df_today['game_number'] == game_number) &
                (df_today['batting_order'].astype(str).str.upper().str.strip() == "SP")
            ]
            if not opp_sp.empty:
                opp_pitcher_map[(game_date, game_number, team)] = str(opp_sp.iloc[0]['mlb_id'])

    df_today['pitcher_id'] = df_today.apply(
        lambda row: opp_pitcher_map.get((row['game_date'], row['game_number'], row['team_code']), np.nan), axis=1
    )

    st.write("Pitcher_id assigned as opponent SP. Sample:", df_today[['team_code','game_date','game_number','player_name','mlb_id','pitcher_id']].head(10))
    st.write("Pitcher_id null count after assign:", df_today['pitcher_id'].isnull().sum())

    # ========== WEATHER PARSING for TODAY ==========
    df_today = parse_weather_fields(df_today)
    st.write("Weather columns parsed for today. Weather sample:", df_today[['weather','temp','wind_mph','wind_dir','condition']].head(2))

    # ========== MERGE ALL FEATURES ==========
    merged = df_today.copy()
    if 'batter_id' not in merged.columns:
        merged['batter_id'] = merged['mlb_id']
    merged = pd.merge(
        merged, batter_event.reset_index(), how='left', left_on='batter_id', right_on='batter_id'
    )
    if not pitcher_event.empty and 'pitcher_id' in merged.columns:
        merged = pd.merge(
            merged, pitcher_event.reset_index(), how='left',
            left_on='pitcher_id', right_on='batter_id', suffixes=('', '_pitcherstats')
        )
    if 'batter_id_pitcherstats' in merged.columns:
        merged = merged.drop(columns=['batter_id_pitcherstats'])
    merged = merged.loc[:, ~merged.columns.duplicated()]

    # ========== FULL DEBUG OUTPUTS ==========
    st.write("---- FULL MERGED COLUMN LIST ----")
    st.write(list(merged.columns))
    st.write("---- FULL MERGED .head() ----")
    st.write(merged.head(10))
    st.write("---- COLUMN INTERSECTION WITH ALL_FEATURE_COLS ----")
    matching_cols = set(merged.columns).intersection(set(all_feature_cols))
    missing_in_merged = [c for c in all_feature_cols if c not in merged.columns]
    extra_in_merged = [c for c in merged.columns if c not in all_feature_cols]
    st.write(f"Matching columns ({len(matching_cols)}): {matching_cols}")
    st.write(f"Missing in merged: {missing_in_merged}")
    st.write(f"Extra in merged: {extra_in_merged}")
    st.write("---- NULL COUNTS FOR MERGED COLUMNS ----")
    st.write(merged.isnull().sum())
    st.write("---- FIRST 3 ROWS OF MERGED ----")
    st.write(merged.head(3))

    # ========== FINAL OUTPUT ==========
    st.success(f"🟢 Generated file with {merged.shape[0]} rows and {merged.shape[1]} columns.")
    st.dataframe(merged.head(10))
    st.download_button(
        "⬇️ Download Today's Event-Level CSV (Merged Columns)",
        data=merged.to_csv(index=False),
        file_name="event_level_today_full.csv"
    )

    diag_text = f"""Columns: {list(merged.columns)}\nNull counts:\n{merged.isnull().sum().to_string()}"""
    st.download_button("⬇️ Download Diagnostics (.txt)", diag_text, file_name="diagnostics.txt")
else:
    st.info("Please upload BOTH today's matchups/lineups and historical event-level CSV.")
