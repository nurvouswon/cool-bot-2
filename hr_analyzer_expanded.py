import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
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

# ------------- WEATHER API FUNCTION (for event-level) ---------------
def get_historical_weather_api(date, city, time="19:00"):
    """ Fetch weather for date/city/time via WeatherAPI. Return dictionary matching our weather columns. """
    key = st.secrets["weatherapi"]["key"]
    endpoint = f"http://api.weatherapi.com/v1/history.json"
    params = {
        "key": key,
        "q": city,
        "dt": date,
    }
    try:
        r = requests.get(endpoint, params=params, timeout=8)
        if r.status_code != 200:
            return {k: np.nan for k in ['temp','wind_mph','humidity','condition','wind_dir_string']}
        data = r.json()
        # Find hour record closest to requested time
        hr = int(time.split(":")[0])
        hourly = data["forecast"]["forecastday"][0]["hour"]
        best_hr = min(hourly, key=lambda x: abs(x["time"].split(" ")[1][:2] and int(x["time"].split(" ")[1][:2]) - hr))
        temp = best_hr["temp_f"]
        wind_mph = best_hr["wind_mph"]
        humidity = best_hr["humidity"]
        condition = best_hr["condition"]["text"].lower()
        wind_dir_string = best_hr["wind_dir"]
        return {
            "temp": temp,
            "wind_mph": wind_mph,
            "humidity": humidity,
            "condition": condition,
            "wind_dir_string": wind_dir_string
        }
    except Exception as e:
        return {k: np.nan for k in ['temp','wind_mph','humidity','condition','wind_dir_string']}

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
        if "park" in lineup_df.columns:
            lineup_df["park"] = lineup_df["park"].astype(str).str.lower().str.replace(" ", "_")
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

        # ==== Parse Weather Fields from Weather String (for TODAY CSV only) ====
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

        # ==== EVENT-LEVEL WEATHER MERGE (WeatherAPI) ====
        progress.progress(30, "Merging WeatherAPI weather to events and running audit...")
        event_weather = []
        for idx, row in df.iterrows():
            city = row.get("city", "")
            date = row.get("game_date", "")
            wx_dict = get_historical_weather_api(date, city)
            out_row = {
                "event_idx": idx,
                "batter_id": row.get("batter_id", ""),
                "player_name": row.get("player_name", ""),
                "team_code": row.get("team_code", ""),
                "game_date": row.get("game_date", ""),
                "park": row.get("park", ""),
                "city": row.get("city", ""),
            }
            out_row.update(wx_dict)
            event_weather.append(out_row)
            # Optional: st.write(f"Fetched weather for event {idx}: {wx_dict}")
        wx_event_df = pd.DataFrame(event_weather)
        # Merge event weather back into event df
        df = pd.concat([df.reset_index(drop=True), wx_event_df[['temp','humidity','wind_mph','condition','wind_dir_string']]], axis=1)
        df = dedup_columns(df)
        
        # === EVENT LEVEL WEATHER AUDIT ===
        weather_audit_rows = []
        for idx, row in df.iterrows():
            missing_cols = []
            for c in ['temp', 'humidity', 'wind_mph', 'condition', 'wind_dir_string']:
                val = row.get(c, np.nan)
                if pd.isna(val) or val == "":
                    missing_cols.append(c)
            weather_audit_rows.append({
                "event_idx": idx,
                "batter_id": row.get("batter_id", ""),
                "player_name": row.get("player_name", ""),
                "team_code": row.get("team_code", ""),
                "game_date": row.get("game_date", ""),
                "park": row.get("park", ""),
                "city": row.get("city", ""),
                "weather_status": "MISSING" if missing_cols else "FOUND",
                "missing_weather_cols": ", ".join(missing_cols)
            })
        weather_audit_df = pd.DataFrame(weather_audit_rows)
        st.markdown("##### Event-Level Weather Audit (first 15 rows):")
        st.dataframe(weather_audit_df.head(15))
        st.download_button(
            "⬇️ Download Full Event-Level Weather Audit CSV",
            data=weather_audit_df.to_csv(index=False),
            file_name="event_level_weather_audit.csv",
            key="download_event_level_weather_audit"
        )

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

        # ===== TODAY CSV: 1 row per batter with all rolling/context features and WEATHER FROM LINEUP CSV =====
        progress.progress(95, "Generating TODAY batter rows and context merges...")
        rolling_feature_cols = [col for col in df.columns if (
            col.startswith('b_') or col.startswith('p_')
        ) and any(str(w) in col for w in roll_windows)]
        extra_context_cols = [
            'park', 'park_hr_rate', 'park_altitude', 'roof_status', 'city'
        ]
        today_cols = [
            'game_date', 'batter_id', 'player_name', 'pitcher_id',
            'temp', 'humidity', 'wind_mph', 'wind_dir_string', 'condition'
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
            # Rolling/context: from last available event
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
                "park_altitude": park_altitude_map.get(str(park).lower(), 0) if not pd.isna(park) else 0,
                "roof_status": roof_status_map.get(str(park).lower(), "open") if not pd.isna(park) else "open",
                "city": city if not pd.isna(city) else mlb_team_city_map.get(team_code, ""),
            })
            # Weather: from parsed weather string in lineup_df
            for c in ['temp', 'humidity', 'wind_mph', 'wind_dir_string', 'condition']:
                row_out[c] = row.get(c, np.nan)
            today_rows.append(row_out)

        today_df = pd.DataFrame(today_rows, columns=today_cols)
        today_df = dedup_columns(today_df)
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
