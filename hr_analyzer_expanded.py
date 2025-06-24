import streamlit as st
import pandas as pd
import numpy as np
import re
import time

st.set_page_config("🟦 Generate Today's Event-Level CSV", layout="wide")
st.title("🟦 Generate Today's Event-Level CSV (FAST, All Rolling Stats, Pitcher+Batter, Pitch Type, All Windows)")

st.markdown("""
**Instructions:**
- Upload Today's Lineups/Matchups CSV (must have `batter_id` or `mlb_id`, `player_name`, etc).
- Upload Historical Event-Level CSV (must have `batter_id`, `pitcher_id`, `pitch_type`, `game_date`, and all stat columns).
- Output: ONE row per batter with ALL rolling/stat features (3/7/14/20, batter & pitcher, per pitch type and overall).
- All numeric columns are auto-corrected for string/decimal mix issues.
""")

today_file = st.file_uploader("Upload Today's Matchups/Lineups CSV", type=["csv"], key="today_csv")
hist_file = st.file_uploader("Upload Historical Event-Level CSV", type=["csv"], key="hist_csv")

event_windows = [3, 7, 14, 20]
main_pitch_types = ["ff", "sl", "cu", "ch", "si", "fc", "fs", "st", "sinker", "splitter", "sweeper"]

# --- your advanced/extra features list here ---
advanced_columns = [
    'hard_hit_rate_20', 'sweet_spot_rate_20', 'park_hand_hr_7', 'park_hand_hr_14', 'park_hand_hr_30',
    'b_vsp_hand_hr_3', 'p_vsb_hand_hr_3', 'b_vsp_hand_hr_5', 'p_vsb_hand_hr_5',
    'b_vsp_hand_hr_7', 'p_vsb_hand_hr_7', 'b_vsp_hand_hr_14', 'p_vsb_hand_hr_14',
    'b_pitchtype_hr_3', 'p_pitchtype_hr_3', 'b_pitchtype_hr_5', 'p_pitchtype_hr_5',
    'b_pitchtype_hr_7', 'p_pitchtype_hr_7', 'b_pitchtype_hr_14', 'p_pitchtype_hr_14',
    'b_launch_speed_3', 'b_launch_speed_5', 'b_launch_speed_7', 'b_launch_speed_14',
    'b_launch_angle_3', 'b_launch_angle_5', 'b_launch_angle_7', 'b_launch_angle_14',
    'b_hit_distance_sc_3', 'b_hit_distance_sc_5', 'b_hit_distance_sc_7', 'b_hit_distance_sc_14',
    'b_woba_value_3', 'b_woba_value_5', 'b_woba_value_7', 'b_woba_value_14',
    'b_release_speed_3', 'b_release_speed_5', 'b_release_speed_7', 'b_release_speed_14',
    'b_release_spin_rate_3', 'b_release_spin_rate_5', 'b_release_spin_rate_7', 'b_release_spin_rate_14',
    'b_spin_axis_3', 'b_spin_axis_5', 'b_spin_axis_7', 'b_spin_axis_14',
    'b_pfx_x_3', 'b_pfx_x_5', 'b_pfx_x_7', 'b_pfx_x_14', 'b_pfx_z_3', 'b_pfx_z_5', 'b_pfx_z_7', 'b_pfx_z_14',
    'p_launch_speed_3', 'p_launch_speed_5', 'p_launch_speed_7', 'p_launch_speed_14',
    'p_launch_angle_3', 'p_launch_angle_5', 'p_launch_angle_7', 'p_launch_angle_14',
    'p_hit_distance_sc_3', 'p_hit_distance_sc_5', 'p_hit_distance_sc_7', 'p_hit_distance_sc_14',
    'p_woba_value_3', 'p_woba_value_5', 'p_woba_value_7', 'p_woba_value_14',
    'p_release_speed_3', 'p_release_speed_5', 'p_release_speed_7', 'p_release_speed_14',
    'p_release_spin_rate_3', 'p_release_spin_rate_5', 'p_release_spin_rate_7', 'p_release_spin_rate_14',
    'p_spin_axis_3', 'p_spin_axis_5', 'p_spin_axis_7', 'p_spin_axis_14',
    'p_pfx_x_3', 'p_pfx_x_5', 'p_pfx_x_7', 'p_pfx_x_14', 'p_pfx_z_3', 'p_pfx_z_5', 'p_pfx_z_7', 'p_pfx_z_14',
    'park', 'temp', 'wind_mph', 'wind_dir', 'humidity', 'condition', 'hr_prob'
]

output_columns = [
    "team_code","game_date","game_number","mlb_id","player_name","batting_order","position",
    "weather","time","stadium","city","batter_id","p_throws", "pitcher_id"
] + advanced_columns

for base in ["avg_exit_velo", "hard_hit_rate", "barrel_rate", "fb_rate", "sweet_spot_rate"]:
    for w in event_windows:
        output_columns.append(f"{base}_{w}")
        output_columns.append(f"p_{base}_{w}")
    for pt in main_pitch_types:
        for w in event_windows:
            output_columns.append(f"{base}_{pt}_{w}")
            output_columns.append(f"p_{base}_{pt}_{w}")

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

@st.cache_data(show_spinner=False)
def cached_fast_rolling_stats(df, id_col, date_col, windows, pitch_types=None, prefix=""):
    return fast_rolling_stats(df, id_col, date_col, windows, pitch_types, prefix)

def fast_rolling_stats(df, id_col, date_col, windows, pitch_types=None, prefix=""):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['launch_speed'] = pd.to_numeric(df.get('launch_speed', np.nan), errors='coerce')
    df['launch_angle'] = pd.to_numeric(df.get('launch_angle', np.nan), errors='coerce')
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

step = [0]
step_total = 8
progress = st.progress(0)
status = st.empty()
def inc(msg=None):
    step[0] += 1
    percent = min(int(100 * step[0] / step_total), 100)
    progress.progress(percent)
    status.markdown(f"{percent}% {'- ' + msg if msg else ''}")
    time.sleep(0.05)

if today_file and hist_file:
    inc("Loading data")
    df_today = pd.read_csv(today_file)
    df_hist = pd.read_csv(hist_file)

    inc("Standardizing columns and IDs")
    df_today.columns = [str(c).strip().lower().replace(" ", "_") for c in df_today.columns]
    df_hist.columns = [str(c).strip().lower().replace(" ", "_") for c in df_hist.columns]

    id_cols_today = ['batter_id', 'mlb_id']
    id_col_today = next((c for c in id_cols_today if c in df_today.columns), None)
    if id_col_today is None:
        st.error("Today's file must have 'batter_id' or 'mlb_id' column.")
        st.stop()
    for col in ['batter_id', 'pitcher_id']:
        if col in df_today.columns:
            df_today[col] = df_today[col].astype(str).str.strip().str.replace('.0', '', regex=False)
        if col in df_hist.columns:
            df_hist[col] = df_hist[col].astype(str).str.strip().str.replace('.0', '', regex=False)

    df_today['batter_id'] = df_today[id_col_today].astype(str).str.strip().str.replace('.0', '', regex=False)
    df_today = df_today.drop_duplicates(subset=["batter_id"])
    df_today = parse_weather_fields(df_today)

    # --- Always create pitcher_id column if missing ---
    if 'pitcher_id' not in df_today.columns:
        df_today['pitcher_id'] = np.nan

    # --- FILL pitcher_id column for all rows for this team/game ---
    if 'position' in df_today.columns and 'mlb_id' in df_today.columns:
        sp_rows = df_today[df_today['position'].astype(str).str.upper().str.strip().isin(['SP', 'P'])]
        for _, sp_row in sp_rows.iterrows():
            mask = (
                (df_today['team_code'] == sp_row['team_code']) &
                (df_today['game_date'] == sp_row['game_date']) &
                (df_today['game_number'] == sp_row['game_number'])
            )
            val = sp_row['mlb_id']
            if not pd.isnull(val):
                try:
                    val = str(int(float(val)))
                except Exception:
                    val = str(val)
            else:
                val = np.nan
            df_today.loc[mask, 'pitcher_id'] = val
    # --- After fill, ENSURE pitcher_id is string and exists! ---
    df_today['pitcher_id'] = df_today['pitcher_id'].astype(str).str.strip().replace('nan','')
    st.write("Pitcher_id column filled. Sample:", df_today[['team_code','game_date','game_number','player_name','position','mlb_id','pitcher_id']].head(8))

    inc("Parsing and validating dates")
    if 'game_date' not in df_hist.columns:
        st.error("Historical file must have a 'game_date' column.")
        st.stop()
    df_hist['game_date'] = pd.to_datetime(df_hist['game_date'], errors='coerce')

    inc("Computing batter rolling stats (cached)...")
    batter_event = cached_fast_rolling_stats(
        df_hist, "batter_id", "game_date", event_windows, main_pitch_types, prefix=""
    )
    batter_event = batter_event.set_index('batter_id')

    inc("Computing pitcher rolling stats (cached)...")
    pitcher_event = cached_fast_rolling_stats(
        df_hist.rename(columns={"pitcher_id": "batter_id", "batter_id": "unused"}),
        "batter_id", "game_date", event_windows, main_pitch_types, prefix="p_"
    )
    pitcher_event = pitcher_event.rename(columns={'batter_id':'pitcher_id'})
    pitcher_event['pitcher_id'] = pitcher_event.index.astype(str)
    pitcher_event = pitcher_event.reset_index(drop=True)

    inc("Merging batter stats into today's data...")
    merged = df_today.set_index('batter_id').join(batter_event, how='left')
    merged = merged.reset_index()

    inc("Merging pitcher stats into today's data...")
    # --- JOIN ON COLUMN NOT INDEX; both as string ---
    merged['pitcher_id'] = merged['pitcher_id'].astype(str).str.strip().replace('nan','')
    pitcher_event['pitcher_id'] = pitcher_event['pitcher_id'].astype(str).str.strip().replace('nan','')
    merged = pd.merge(merged, pitcher_event, how='left', on='pitcher_id', suffixes=('', '_pstat'))

    inc("Final formatting and reindexing")
    missing_cols = [col for col in output_columns if col not in merged.columns]
    if missing_cols:
        nan_df = pd.DataFrame(np.nan, index=merged.index, columns=missing_cols)
        merged = pd.concat([merged, nan_df], axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    merged = merged.reindex(columns=output_columns)

    inc(f"🟢 Generated file: {merged.shape[0]} unique batters, {merged.shape[1]} columns (features).")
    st.dataframe(merged.head(10))

    st.download_button(
        "⬇️ Download Today's Event-Level CSV (Exact Format)",
        data=merged.to_csv(index=False),
        file_name="event_level_today_full.csv"
    )
else:
    st.info("Please upload BOTH today's matchups/lineups and historical event-level CSV.")
