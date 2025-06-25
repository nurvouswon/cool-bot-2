import streamlit as st
import pandas as pd
import numpy as np
import re
import time

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
    df = df.drop_duplicates([id_col, date_col], keep="last")
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['launch_speed'] = pd.to_numeric(df['launch_speed'], errors='coerce')
    df['launch_angle'] = pd.to_numeric(df['launch_angle'], errors='coerce')
    if 'pitch_type' in df.columns:
        df['pitch_type'] = df['pitch_type'].astype(str).str.lower().str.strip()
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

# ========== MAIN APP ==========
today_file = st.file_uploader("Upload Today's Matchups/Lineups CSV", type=["csv"], key="today_csv")
hist_file = st.file_uploader("Upload Historical Event-Level CSV", type=["csv"], key="hist_csv")

# --- Define your full feature list! ---
all_feature_cols = [
    # Core
    "team_code","game_date","game_number","mlb_id","player_name","batting_order","position",
    "weather","time","stadium","city","batter_id","p_throws","pitcher_id",
    # Rolling Stats
    "hard_hit_rate_20","sweet_spot_rate_20",
    "park_hand_hr_7","park_hand_hr_14","park_hand_hr_30",
    "b_vsp_hand_hr_3","p_vsb_hand_hr_3","b_vsp_hand_hr_5","p_vsb_hand_hr_5",
    "b_vsp_hand_hr_7","p_vsb_hand_hr_7","b_vsp_hand_hr_14","p_vsb_hand_hr_14",
    "b_pitchtype_hr_3","p_pitchtype_hr_3","b_pitchtype_hr_5","p_pitchtype_hr_5",
    "b_pitchtype_hr_7","p_pitchtype_hr_7","b_pitchtype_hr_14","p_pitchtype_hr_14",
    # Batter rolling batted ball
    "b_launch_speed_3","b_launch_speed_5","b_launch_speed_7","b_launch_speed_14",
    "b_launch_angle_3","b_launch_angle_5","b_launch_angle_7","b_launch_angle_14",
    "b_hit_distance_sc_3","b_hit_distance_sc_5","b_hit_distance_sc_7","b_hit_distance_sc_14",
    "b_woba_value_3","b_woba_value_5","b_woba_value_7","b_woba_value_14",
    "b_release_speed_3","b_release_speed_5","b_release_speed_7","b_release_speed_14",
    "b_release_spin_rate_3","b_release_spin_rate_5","b_release_spin_rate_7","b_release_spin_rate_14",
    "b_spin_axis_3","b_spin_axis_5","b_spin_axis_7","b_spin_axis_14",
    "b_pfx_x_3","b_pfx_x_5","b_pfx_x_7","b_pfx_x_14",
    "b_pfx_z_3","b_pfx_z_5","b_pfx_z_7","b_pfx_z_14",
    # Pitcher rolling
    "p_launch_speed_3","p_launch_speed_5","p_launch_speed_7","p_launch_speed_14",
    "p_launch_angle_3","p_launch_angle_5","p_launch_angle_7","p_launch_angle_14",
    "p_hit_distance_sc_3","p_hit_distance_sc_5","p_hit_distance_sc_7","p_hit_distance_sc_14",
    "p_woba_value_3","p_woba_value_5","p_woba_value_7","p_woba_value_14",
    "p_release_speed_3","p_release_speed_5","p_release_speed_7","p_release_speed_14",
    "p_release_spin_rate_3","p_release_spin_rate_5","p_release_spin_rate_7","p_release_spin_rate_14",
    "p_spin_axis_3","p_spin_axis_5","p_spin_axis_7","p_spin_axis_14",
    "p_pfx_x_3","p_pfx_x_5","p_pfx_x_7","p_pfx_x_14",
    "p_pfx_z_3","p_pfx_z_5","p_pfx_z_7","p_pfx_z_14",
    # Environment
    "park","temp","wind_mph","wind_dir","humidity","condition","hr_prob"
]

# ========== LOGIC ==========
if today_file and hist_file:
    st.info("Loaded today's matchups and historical event data.")

    # LOAD DATA
    df_today = read_csv(today_file)
    df_hist = read_csv(hist_file)

    st.write("Today's Data Sample:", df_today.head(2))
    st.write("Historical Data Sample:", df_hist.head(2))

    # -------- STANDARDIZE IDS/STRINGS --------
    df_today.columns = [str(c).strip().lower().replace(" ", "_") for c in df_today.columns]
    df_hist.columns = [str(c).strip().lower().replace(" ", "_") for c in df_hist.columns]

    # Clean IDs
    for col in ['batter_id', 'mlb_id', 'pitcher_id']:
        if col in df_today.columns:
            df_today[col] = df_today[col].astype(str).str.replace('.0','',regex=False).str.strip()
        if col in df_hist.columns:
            df_hist[col] = df_hist[col].astype(str).str.replace('.0','',regex=False).str.strip()

    # ============= PITCHER_ID FILL LOGIC =============
    if 'pitcher_id' not in df_today.columns:
        df_today['pitcher_id'] = np.nan

    sp_rows = df_today[df_today['batting_order'].astype(str).str.upper() == "SP"]
    st.write("Rows where batting_order == 'SP':", sp_rows[['team_code','game_date','game_number','mlb_id','player_name','batting_order']])

    for idx, sp_row in sp_rows.iterrows():
        mlb_id_val = sp_row['mlb_id']
        if pd.isnull(mlb_id_val) or str(mlb_id_val).strip() == "" or str(mlb_id_val).lower() == "nan":
            st.warning(f"Skipping SP row with missing mlb_id: {sp_row.to_dict()}")
            continue
        try:
            pid = str(int(float(mlb_id_val)))
        except Exception as e:
            st.warning(f"Failed to convert mlb_id {mlb_id_val}: {e}")
            continue
        mask = (
            (df_today['team_code'] == sp_row['team_code']) &
            (df_today['game_date'] == sp_row['game_date']) &
            (df_today['game_number'] == sp_row['game_number'])
        )
        st.write(f"Set pitcher_id={pid} for ({sp_row['team_code']}, {sp_row['game_date']}, {sp_row['game_number']})")
        df_today.loc[mask, 'pitcher_id'] = pid

    df_today['pitcher_id'] = df_today['pitcher_id'].astype(str).str.replace('.0','',regex=False).str.strip()
    st.write("Pitcher_id filled. Sample:", df_today[['team_code','game_date','game_number','player_name','batting_order','mlb_id','pitcher_id']].head(10))
    st.write("Pitcher_id null count after fill:", df_today['pitcher_id'].isnull().sum())

    # ---- Weather Parse ----
    df_today = parse_weather_fields(df_today)
    st.write("Weather columns parsed. Weather sample:", df_today[['weather','temp','wind_mph','wind_dir','condition']].head(2))

    # ---- Rolling Stats Setup ----
    event_windows = [3,7,14,20]
    main_pitch_types = ["ff", "sl", "cu", "ch", "si", "fc", "fs", "st", "sinker", "splitter", "sweeper"]

    # ---- Rolling BATTER stats ----
    st.write("Running fast_rolling_stats for batters...")
    batter_event = fast_rolling_stats(df_hist, "batter_id", "game_date", event_windows, main_pitch_types, prefix="")
    st.write("Batter rolling event stats sample:", batter_event.head(3))
    batter_event = batter_event.set_index('batter_id')

    # ---- Rolling PITCHER stats ----
    st.write("Running fast_rolling_stats for pitchers...")
    pitcher_event = pd.DataFrame()
    if 'pitcher_id' in df_hist.columns:
        pitcher_event = fast_rolling_stats(
            df_hist.rename(columns={"pitcher_id":"batter_id"}),
            "batter_id", "game_date", event_windows, main_pitch_types, prefix="p_"
        )
    elif 'mlb_id' in df_hist.columns:
        pitcher_event = fast_rolling_stats(
            df_hist.rename(columns={"mlb_id":"batter_id"}),
            "batter_id", "game_date", event_windows, main_pitch_types, prefix="p_"
        )
    if not pitcher_event.empty:
        st.write("Pitcher rolling stats sample:", pitcher_event.head(3))
    pitcher_event = pitcher_event.set_index('batter_id')
    pitcher_event.index.name = 'pitcher_id'

    # ---- Merge STATS into today ----
    st.write("Merging batter and pitcher rolling stats...")
    if 'batter_id' not in df_today.columns:
        df_today['batter_id'] = df_today['mlb_id']  # fallback if missing
    merged = df_today.set_index('batter_id').join(batter_event, how='left')
    merged = merged.reset_index()
    if not pitcher_event.empty and 'pitcher_id' in merged.columns:
        merged = merged.set_index('pitcher_id').join(pitcher_event, how='left', rsuffix="_pitcher").reset_index()
    st.write("Merged Data Sample:", merged.head(8))

    # ---- Add advanced/statcast/park features ----
    for col in all_feature_cols:
        if col not in merged.columns:
            merged[col] = np.nan  # Fill as NaN for output consistency

    # ---- Final Output Formatting ----
    merged = merged.loc[:,~merged.columns.duplicated()]
    merged = merged.reindex(columns=all_feature_cols)
    st.success(f"🟢 Generated file with {merged.shape[0]} rows and {merged.shape[1]} columns.")
    st.dataframe(merged.head(10))

    # ---- Diagnostic Output (Copy/Paste block) ----
    diag_text = f"""
Data Diagnostics:
- df_today columns: {list(df_today.columns)}
- df_hist columns: {list(df_hist.columns)}
- batter_event shape: {batter_event.shape}
- pitcher_event shape: {pitcher_event.shape}
- merged shape: {merged.shape}
- pitcher_id nulls: {merged['pitcher_id'].isnull().sum()}
- feature columns in output: {all_feature_cols}
"""
    st.download_button("⬇️ Download Diagnostics (.txt)", diag_text, file_name="diagnostics.txt")
    st.download_button("⬇️ Download Today's Event-Level CSV (Exact Format)", data=merged.to_csv(index=False), file_name="event_level_today_full.csv")

else:
    st.info("Please upload BOTH today's matchups/lineups and historical event-level CSV.")
