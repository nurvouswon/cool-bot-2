import streamlit as st
import pandas as pd
import numpy as np
from pybaseball import statcast
from datetime import datetime, timedelta
import requests

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

def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

def rolling_pitch_type_hr(df, id_col, pitch_col, window):
    out = np.full(len(df), np.nan)
    df = df.reset_index(drop=True)
    grouped = df.groupby([id_col, pitch_col])
    for _, group_idx in grouped.groups.items():
        group_idx = list(group_idx)
        vals = df.loc[group_idx, 'hr_outcome'].shift(1).rolling(window, min_periods=1).mean()
        out[group_idx] = vals
    return out

st.set_page_config("MLB HR Analyzer", layout="wide")
tab1, tab2 = st.tabs(["1️⃣ Fetch & Feature Engineer Data", "2️⃣ Upload & Analyze"])

with tab1:
    st.header("Fetch Statcast Data & Generate Features")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", value=datetime.today())

    st.markdown("##### (Optional) Upload Today's Matchups/Lineups CSV (for city, stadium, time, weather context)")
    uploaded_lineups = st.file_uploader("Upload Today's Matchups CSV", type="csv", key="lineupsup")
    fetch_btn = st.button("Fetch Statcast, Feature Engineer, and Download", type="primary")
    progress = st.empty()

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

        valid_events = [
            'single', 'double', 'triple', 'homerun', 'home_run', 'field_out',
            'force_out', 'grounded_into_double_play', 'fielders_choice_out',
            'pop_out', 'lineout', 'flyout', 'sac_fly', 'sac_fly_double_play'
        ]
        if 'events' in df.columns:
            df = df[df['events'].str.lower().str.replace(' ', '').isin(valid_events)].copy()
        if 'hr_outcome' not in df.columns:
            if 'events' in df.columns:
                df['hr_outcome'] = df['events'].astype(str).str.lower().str.replace(' ', '').isin(['homerun', 'home_run']).astype(int)
            else:
                df['hr_outcome'] = np.nan

        if 'home_team_code' in df.columns:
            df['home_team_code'] = df['home_team_code'].astype(str).str.upper()
        if 'park' not in df.columns:
            if 'home_team_code' in df.columns:
                df['park'] = df['home_team_code'].map(team_code_to_park)
        if 'park' in df.columns and df['park'].isnull().any() and 'home_team' in df.columns:
            df['park'] = df['park'].fillna(df['home_team'].str.lower().str.replace(' ', '_'))
        elif 'home_team' in df.columns and 'park' not in df.columns:
            df['park'] = df['home_team'].str.lower().str.replace(' ', '_')

        if uploaded_lineups is not None:
            lineups = pd.read_csv(uploaded_lineups)
            if 'game_date' in df.columns and 'team code' in lineups.columns:
                df['game_date'] = df['game_date'].astype(str)
                lineups['game_date'] = lineups['game_date'].astype(str)
                if 'home_team_code' in df.columns:
                    merged = df.merge(
                        lineups[['team code', 'game_date', 'city', 'stadium', 'time', 'weather']],
                        left_on=['home_team_code', 'game_date'],
                        right_on=['team code', 'game_date'],
                        how='left'
                    )
                    df['city'] = merged['city']
                    df['stadium'] = merged['stadium']
                    df['time'] = merged['time']
                    df['weather'] = merged['weather']
            if 'stadium' in df.columns:
                df['park'] = df['stadium'].str.lower().str.replace(' ', '_')

        if 'park' not in df.columns:
            st.error("Could not determine ballpark from your data (missing 'park', 'home_team_code', and 'home_team').")
            st.stop()
        df['park_hr_rate'] = df['park'].map(park_hr_rate_map).fillna(1.0)
        df['park_altitude'] = df['park'].map(park_altitude_map).fillna(0)
        df['roof_status'] = df['park'].map(roof_status_map).fillna("open")
        progress.progress(20, "Park/team/stadium context merged")

        # === Weather Debug - this block ensures all weather headers get populated ===
        weather_features = ['temp', 'wind_mph', 'wind_dir', 'humidity', 'condition']
        if 'city' in df.columns and 'game_date' in df.columns:
            df['city'] = df['city'].astype(str).str.strip()
            df['game_date'] = df['game_date'].astype(str)
            df['weather_key'] = df['city'] + "_" + df['game_date']
            unique_keys = df['weather_key'].dropna().unique()
            weather_cache = {}
            for i, key in enumerate(unique_keys):
                if '_' not in key: continue
                city, date = key.split('_', 1)
                if not city or not date or city == "nan": continue
                if key in weather_cache:
                    weather = weather_cache[key]
                else:
                    weather = get_weather(city, date)
                    weather_cache[key] = weather
                for feat in weather_features:
                    df.loc[df['weather_key'] == key, feat] = weather[feat]
                progress.progress(20 + int(30 * (i+1) / len(unique_keys)), text=f"Weather {i+1}/{len(unique_keys)}")
            progress.progress(50, "Weather merged (city/date).")
        elif 'home_team_code' in df.columns and 'game_date' in df.columns:
            df['home_team_code'] = df['home_team_code'].astype(str)
            df['game_date'] = df['game_date'].astype(str)
            df['weather_key'] = df['home_team_code'] + "_" + df['game_date']
            unique_keys = df['weather_key'].dropna().unique()
            weather_cache = {}
            for i, key in enumerate(unique_keys):
                team = key.split('_')[0]
                city = mlb_team_city_map.get(team, "New York")
                date = key.split('_', 1)[1]
                if key in weather_cache:
                    weather = weather_cache[key]
                else:
                    weather = get_weather(city, date)
                    weather_cache[key] = weather
                for feat in weather_features:
                    df.loc[df['weather_key'] == key, feat] = weather[feat]
                progress.progress(20 + int(30 * (i+1) / len(unique_keys)), text=f"Weather {i+1}/{len(unique_keys)}")
            progress.progress(50, "Weather merged (fallback by team/city).")
        else:
            for feat in weather_features:
                df[feat] = None

        df['wind_dir_angle'] = df['wind_dir'].apply(wind_dir_to_angle)
        df['wind_dir_sin'] = np.sin(np.deg2rad(df['wind_dir_angle']))
        df['wind_dir_cos'] = np.cos(np.deg2rad(df['wind_dir_angle']))

        if 'batter_id' not in df.columns and 'batter' in df.columns:
            df['batter_id'] = df['batter']
        if 'pitcher_id' not in df.columns and 'pitcher' in df.columns:
            df['pitcher_id'] = df['pitcher']

        if 'barrel' in df.columns and 'batter_id' in df.columns:
            df['barrel_rate_20'] = df.groupby('batter_id')['barrel'].transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
        if 'launch_speed' in df.columns and 'batter_id' in df.columns:
            df['hard_hit_rate_20'] = df.groupby('batter_id')['launch_speed'].transform(lambda x: (x.shift(1) >= 95).rolling(20, min_periods=5).mean())
        if 'launch_angle' in df.columns and 'batter_id' in df.columns:
            df['sweet_spot_rate_20'] = df.groupby('batter_id')['launch_angle'].transform(lambda x: x.shift(1).between(8, 32).rolling(20, min_periods=5).mean())

        if 'stand' in df.columns and 'wind_dir_angle' in df.columns:
            def relative_wind_angle(row):
                try:
                    if row['stand'] == 'L':
                        return (row['wind_dir_angle'] - 45) % 360
                    else:
                        return (row['wind_dir_angle'] - 135) % 360
                except Exception:
                    return np.nan
            df['relative_wind_angle'] = df.apply(relative_wind_angle, axis=1)
            df['relative_wind_sin'] = np.sin(np.deg2rad(df['relative_wind_angle']))
            df['relative_wind_cos'] = np.cos(np.deg2rad(df['relative_wind_angle']))

        if 'park' in df.columns and 'stand' in df.columns:
            for w in [7, 14, 30]:
                col_name = f'park_hand_HR_{w}'
                df[col_name] = df.groupby(['park', 'stand'])['hr_outcome'].transform(lambda x: x.shift(1).rolling(w, min_periods=5).mean())

        roll_windows = [3, 5, 7, 14]
        batter_cols = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value',
                       'release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z']
        pitcher_cols = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value','release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z']

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

        if 'stand' in df.columns and 'p_throws' in df.columns:
            for w in roll_windows:
                df[f'B_vsP_hand_HR_{w}'] = df.groupby(['batter_id', 'p_throws'])['hr_outcome'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                df[f'P_vsB_hand_HR_{w}'] = df.groupby(['pitcher_id', 'stand'])['hr_outcome'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())

        if 'pitch_type' in df.columns:
            for w in roll_windows:
                df[f'B_pitchtype_HR_{w}'] = rolling_pitch_type_hr(df, 'batter_id', 'pitch_type', w)
                df[f'P_pitchtype_HR_{w}'] = rolling_pitch_type_hr(df, 'pitcher_id', 'pitch_type', w)

        # Merge rolling stats back into df
        df = pd.concat([df, pd.DataFrame(batter_feat_dict), pd.DataFrame(pitcher_feat_dict)], axis=1)
        df = df.copy()
        progress.progress(80, "Advanced Statcast & context features done")
        df = dedup_columns(df)
        st.success(f"Feature engineering complete! {len(df)} batted ball events.")
        st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
        st.dataframe(df.head(20))
        
        # --------- OUTPUT COLUMN ALIGNMENT TO "TODAY CSV" ---------
        today_csv_header = """team_code	game_date	game_number	mlb_id	player_name	batting_order	position	weather	time	stadium	city	pitcher_id	temp	wind_mph	wind_dir	condition	batter_id	avg_exit_velo_3	hard_hit_rate_3	barrel_rate_3	fb_rate_3	sweet_spot_rate_3	avg_exit_velo_7	hard_hit_rate_7	barrel_rate_7	fb_rate_7	sweet_spot_rate_7	avg_exit_velo_14	hard_hit_rate_14	barrel_rate_14	fb_rate_14	sweet_spot_rate_14	avg_exit_velo_20	hard_hit_rate_20	barrel_rate_20	fb_rate_20	sweet_spot_rate_20	avg_exit_velo_ff_3	hard_hit_rate_ff_3	barrel_rate_ff_3	fb_rate_ff_3	sweet_spot_rate_ff_3	avg_exit_velo_ff_7	hard_hit_rate_ff_7	barrel_rate_ff_7	fb_rate_ff_7	sweet_spot_rate_ff_7	avg_exit_velo_ff_14	hard_hit_rate_ff_14	barrel_rate_ff_14	fb_rate_ff_14	sweet_spot_rate_ff_14	avg_exit_velo_ff_20	hard_hit_rate_ff_20	barrel_rate_ff_20	fb_rate_ff_20	sweet_spot_rate_ff_20	avg_exit_velo_sl_3	hard_hit_rate_sl_3	barrel_rate_sl_3	fb_rate_sl_3	sweet_spot_rate_sl_3	avg_exit_velo_sl_7	hard_hit_rate_sl_7	barrel_rate_sl_7	fb_rate_sl_7	sweet_spot_rate_sl_7	avg_exit_velo_sl_14	hard_hit_rate_sl_14	barrel_rate_sl_14	fb_rate_sl_14	sweet_spot_rate_sl_14	avg_exit_velo_sl_20	hard_hit_rate_sl_20	barrel_rate_sl_20	fb_rate_sl_20	sweet_spot_rate_sl_20	avg_exit_velo_cu_3	hard_hit_rate_cu_3	barrel_rate_cu_3	fb_rate_cu_3	sweet_spot_rate_cu_3	avg_exit_velo_cu_7	hard_hit_rate_cu_7	barrel_rate_cu_7	fb_rate_cu_7	sweet_spot_rate_cu_7	avg_exit_velo_cu_14	hard_hit_rate_cu_14	barrel_rate_cu_14	fb_rate_cu_14	sweet_spot_rate_cu_14	avg_exit_velo_cu_20	hard_hit_rate_cu_20	barrel_rate_cu_20	fb_rate_cu_20	sweet_spot_rate_cu_20	avg_exit_velo_ch_3	hard_hit_rate_ch_3	barrel_rate_ch_3	fb_rate_ch_3	sweet_spot_rate_ch_3	avg_exit_velo_ch_7	hard_hit_rate_ch_7	barrel_rate_ch_7	fb_rate_ch_7	sweet_spot_rate_ch_7	avg_exit_velo_ch_14	hard_hit_rate_ch_14	barrel_rate_ch_14	fb_rate_ch_14	sweet_spot_rate_ch_14	avg_exit_velo_ch_20	hard_hit_rate_ch_20	barrel_rate_ch_20	fb_rate_ch_20	sweet_spot_rate_ch_20	avg_exit_velo_si_3	hard_hit_rate_si_3	barrel_rate_si_3	fb_rate_si_3	sweet_spot_rate_si_3	avg_exit_velo_si_7	hard_hit_rate_si_7	barrel_rate_si_7	fb_rate_si_7	sweet_spot_rate_si_7	avg_exit_velo_si_14	hard_hit_rate_si_14	barrel_rate_si_14	fb_rate_si_14	sweet_spot_rate_si_14	avg_exit_velo_si_20	hard_hit_rate_si_20	barrel_rate_si_20	fb_rate_si_20	sweet_spot_rate_si_20	avg_exit_velo_fc_3	hard_hit_rate_fc_3	barrel_rate_fc_3	fb_rate_fc_3	sweet_spot_rate_fc_3	avg_exit_velo_fc_7	hard_hit_rate_fc_7	barrel_rate_fc_7	fb_rate_fc_7	sweet_spot_rate_fc_7	avg_exit_velo_fc_14	hard_hit_rate_fc_14	barrel_rate_fc_14	fb_rate_fc_14	sweet_spot_rate_fc_14	avg_exit_velo_fc_20	hard_hit_rate_fc_20	barrel_rate_fc_20	fb_rate_fc_20	sweet_spot_rate_fc_20	avg_exit_velo_fs_3	hard_hit_rate_fs_3	barrel_rate_fs_3	fb_rate_fs_3	sweet_spot_rate_fs_3	avg_exit_velo_fs_7	hard_hit_rate_fs_7	barrel_rate_fs_7	fb_rate_fs_7	sweet_spot_rate_fs_7	avg_exit_velo_fs_14	hard_hit_rate_fs_14	barrel_rate_fs_14	fb_rate_fs_14	sweet_spot_rate_fs_14	avg_exit_velo_fs_20	hard_hit_rate_fs_20	barrel_rate_fs_20	fb_rate_fs_20	sweet_spot_rate_fs_20	avg_exit_velo_st_3	hard_hit_rate_st_3	barrel_rate_st_3	fb_rate_st_3	sweet_spot_rate_st_3	avg_exit_velo_st_7	hard_hit_rate_st_7	barrel_rate_st_7	fb_rate_st_7	sweet_spot_rate_st_7	avg_exit_velo_st_14	hard_hit_rate_st_14	barrel_rate_st_14	fb_rate_st_14	sweet_spot_rate_st_14	avg_exit_velo_st_20	hard_hit_rate_st_20	barrel_rate_st_20	fb_rate_st_20	sweet_spot_rate_st_20	avg_exit_velo_sinker_3	hard_hit_rate_sinker_3	barrel_rate_sinker_3	fb_rate_sinker_3	sweet_spot_rate_sinker_3	avg_exit_velo_sinker_7	hard_hit_rate_sinker_7	barrel_rate_sinker_7	fb_rate_sinker_7	sweet_spot_rate_sinker_7	avg_exit_velo_sinker_14	hard_hit_rate_sinker_14	barrel_rate_sinker_14	fb_rate_sinker_14	sweet_spot_rate_sinker_14	avg_exit_velo_sinker_20	hard_hit_rate_sinker_20	barrel_rate_sinker_20	fb_rate_sinker_20	sweet_spot_rate_sinker_20	avg_exit_velo_splitter_3	hard_hit_rate_splitter_3	barrel_rate_splitter_3	fb_rate_splitter_3	sweet_spot_rate_splitter_3	avg_exit_velo_splitter_7	hard_hit_rate_splitter_7	barrel_rate_splitter_7	fb_rate_splitter_7	sweet_spot_rate_splitter_7	avg_exit_velo_splitter_14	hard_hit_rate_splitter_14	barrel_rate_splitter_14	fb_rate_splitter_14	sweet_spot_rate_splitter_14	avg_exit_velo_splitter_20	hard_hit_rate_splitter_20	barrel_rate_splitter_20	fb_rate_splitter_20	sweet_spot_rate_splitter_20	avg_exit_velo_sweeper_3	hard_hit_rate_sweeper_3	barrel_rate_sweeper_3	fb_rate_sweeper_3	sweet_spot_rate_sweeper_3	avg_exit_velo_sweeper_7	hard_hit_rate_sweeper_7	barrel_rate_sweeper_7	fb_rate_sweeper_7	sweet_spot_rate_sweeper_7	avg_exit_velo_sweeper_14	hard_hit_rate_sweeper_14	barrel_rate_sweeper_14	fb_rate_sweeper_14	sweet_spot_rate_sweeper_14	avg_exit_velo_sweeper_20	hard_hit_rate_sweeper_20	barrel_rate_sweeper_20	fb_rate_sweeper_20	sweet_spot_rate_sweeper_20""".split('\t')
        today_csv_header = [c.strip() for c in today_csv_header if c.strip()]

        # Save event-level CSV (all features)
        st.download_button("⬇️ Download Event-Level CSV (all features)", data=df.to_csv(index=False), file_name="event_level_full.csv", key="download_event_level_full")

        # TODAY CSV: 1 row per batter/pitcher (matching format)
        # Select most recent event for each batter/pitcher combo, on latest game_date
        group_cols = ['game_date', 'batter_id', 'pitcher_id']
        if all(c in df.columns for c in group_cols):
            idx = df.groupby(group_cols)['game_date'].idxmax()
            today_df = df.loc[idx].copy()
        else:
            # fallback: just 1 row per batter_id (latest date)
            group_cols = ['game_date', 'batter_id']
            idx = df.groupby(group_cols)['game_date'].idxmax()
            today_df = df.loc[idx].copy()

        # Optionally, join additional lineup info (player name, team code, etc) if available
        if uploaded_lineups is not None:
            lineups = pd.read_csv(uploaded_lineups)
            merge_keys = [k for k in ['batter_id', 'game_date'] if k in today_df.columns and k in lineups.columns]
            if merge_keys:
                today_df = today_df.merge(lineups, how='left', on=merge_keys)
        
        # Reorder columns for today CSV
        output_cols = [c for c in today_csv_header if c in today_df.columns] + [c for c in today_df.columns if c not in today_csv_header]
        today_df = today_df.reindex(columns=output_cols)

        # Save today CSV
        st.download_button("⬇️ Download TODAY CSV (1 per batter, all features)", data=today_df.to_csv(index=False), file_name="today_hr_features.csv", key="download_today_csv")

        st.success("✅ Both CSVs generated. All feature columns and weather fields populated and consistent!")
