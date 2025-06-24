import streamlit as st
import pandas as pd
import numpy as np
import re
import time

st.set_page_config("🟦 Generate Today's Event-Level CSV", layout="wide")
st.title("🟦 Generate Today's Event-Level CSV (ALL ROLLING + ADVANCED FEATURES)")

st.markdown("""
**Instructions:**
- Upload Today's Lineups/Matchups CSV (must have `batter_id` or `mlb_id`, `player_name`, etc).
- Upload Historical Event-Level CSV (must have `batter_id`, `pitcher_id`, `pitch_type`, `game_date`, and all stat columns).
- Output: ONE row per batter with ALL rolling/stat features (3/7/14/20, batter & pitcher, per pitch type and overall), plus advanced features.
""")

today_file = st.file_uploader("Upload Today's Matchups/Lineups CSV", type=["csv"], key="today_csv")
hist_file = st.file_uploader("Upload Historical Event-Level CSV", type=["csv"], key="hist_csv")

event_windows = [3, 7, 14, 20]
main_pitch_types = ["ff", "sl", "cu", "ch", "si", "fc", "fs", "st", "sinker", "splitter", "sweeper"]

# All advanced features you listed
extra_features = [
    "park_hand_hr_{w}", "b_vsp_hand_hr_{w}", "p_vsb_hand_hr_{w}",
    "b_pitchtype_hr_{w}", "p_pitchtype_hr_{w}",
    "b_launch_speed_{w}", "b_launch_angle_{w}", "b_hit_distance_sc_{w}",
    "b_woba_value_{w}", "b_release_speed_{w}", "b_release_spin_rate_{w}",
    "b_spin_axis_{w}", "b_pfx_x_{w}", "b_pfx_z_{w}",
    "p_launch_speed_{w}", "p_launch_angle_{w}", "p_hit_distance_sc_{w}",
    "p_woba_value_{w}", "p_release_speed_{w}", "p_release_spin_rate_{w}",
    "p_spin_axis_{w}", "p_pfx_x_{w}", "p_pfx_z_{w}"
]

output_columns = [
    "team_code","game_date","game_number","mlb_id","player_name","batting_order","position",
    "weather","temp","wind_mph","wind_dir","condition","stadium","city","batter_id","p_throws"
]
# Add advanced columns for each window
for w in event_windows:
    for col in extra_features:
        output_columns.append(col.format(w=w))

# Add all rolling stat columns (as before)
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

@st.cache_data(show_spinner=True)
def fast_rolling_stats(df, id_col, date_col, windows, pitch_types=None, prefix=""):
    df = df.copy()
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

def advanced_rolling(df_hist, key_col, target_col, w):
    return df_hist.groupby(key_col)[target_col].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)

def hr_rate(df_hist, group_cols, w):
    def _hr(events):
        return np.mean(events.str.contains("home_run", na=False))
    return df_hist.groupby(group_cols)['events'].rolling(w, min_periods=1).apply(_hr).reset_index(level=group_cols, drop=True)

def generic_rolling_stat(df_hist, key_col, col, w, func=np.mean):
    return df_hist.groupby(key_col)[col].rolling(w, min_periods=1).apply(func).reset_index(level=0, drop=True)

def fill_pitcher_id(df_today):
    sp_mask = (df_today["position"].astype(str).str.upper() == "SP")
    sp_df = df_today.loc[sp_mask, ["team_code", "game_date", "mlb_id"]].copy()
    sp_df = sp_df.rename(columns={"mlb_id": "pitcher_id"})
    return pd.merge(
        df_today,
        sp_df[["team_code", "game_date", "pitcher_id"]],
        on=["team_code", "game_date"],
        how="left"
    )

step = [0]
step_total = 9
progress = st.progress(0)
status = st.empty()
def inc(msg=None):
    step[0] += 1
    percent = min(int(100 * step[0] / step_total), 100)
    progress.progress(percent)
    status.markdown(f"{percent}% {'- ' + msg if msg else ''}")
    time.sleep(0.01)

if today_file and hist_file:
    inc("Loading data")
    df_today = pd.read_csv(today_file)
    df_hist = pd.read_csv(hist_file)

    st.write("Loaded Today's matchups and historical event data.")
    st.write(df_today.head(2))
    st.write(df_hist.head(2))

    inc("Standardizing columns and IDs")
    df_today.columns = [str(c).strip().lower().replace(" ", "_") for c in df_today.columns]
    df_hist.columns = [str(c).strip().lower().replace(" ", "_") for c in df_hist.columns]

    if "team_code" in df_today.columns and "game_date" in df_today.columns and "mlb_id" in df_today.columns:
        df_today = fill_pitcher_id(df_today)
        st.write("Pitcher_id column filled. Sample:")
        st.write(df_today[["team_code", "game_date", "position", "mlb_id", "pitcher_id"]].head(8))
    else:
        st.warning("Pitcher_id not filled: Required columns missing.")

    for col in ['batter_id', 'pitcher_id', 'mlb_id']:
        if col in df_today.columns:
            df_today[col] = df_today[col].astype(str).str.replace('.0', '', regex=False).str.strip()
        if col in df_hist.columns:
            df_hist[col] = df_hist[col].astype(str).str.replace('.0', '', regex=False).str.strip()
    if "batter_id" not in df_today.columns:
        df_today['batter_id'] = df_today['mlb_id']

    df_today = parse_weather_fields(df_today)
    inc("Parsed weather data for today's lineup.")

    if 'game_date' not in df_hist.columns:
        st.error("Historical file must have a 'game_date' column.")
        st.stop()
    df_hist['game_date'] = pd.to_datetime(df_hist['game_date'], errors='coerce')

    inc("Computing batter rolling stats...")
    batter_event = fast_rolling_stats(
        df_hist, "batter_id", "game_date", event_windows, main_pitch_types, prefix=""
    )
    batter_event = batter_event.set_index('batter_id')
    st.write("Batter event rolling stats computed.")

    inc("Computing pitcher rolling stats...")
    pitcher_event = fast_rolling_stats(
        df_hist.rename(columns={"pitcher_id": "batter_id", "batter_id": "unused"}),
        "batter_id", "game_date", event_windows, main_pitch_types, prefix="p_"
    )
    pitcher_event = pitcher_event.set_index('batter_id')
    pitcher_event.index.name = 'pitcher_id'
    st.write("Pitcher rolling stats computed.")

    inc("Merging batter stats into today's data...")
    merged = df_today.set_index('batter_id').join(batter_event, how='left').reset_index()

    inc("Merging pitcher stats into today's data...")
    if not pitcher_event.empty and 'pitcher_id' in merged.columns:
        merged = merged.set_index('pitcher_id').join(pitcher_event, how='left').reset_index()
        st.write("Merged pitcher stats into today's data. Sample:")
        st.write(merged.head(4))
    else:
        st.warning("No pitcher stats found.")

    inc("Final formatting and reindexing")
    missing_cols = [col for col in output_columns if col not in merged.columns]
    if missing_cols:
        nan_df = pd.DataFrame(np.nan, index=merged.index, columns=missing_cols)
        merged = pd.concat([merged, nan_df], axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    merged = merged.reindex(columns=output_columns + [c for c in merged.columns if c not in output_columns])

    # --- Add advanced features ---
    st.write("Computing advanced features. This can take a moment...")

    for w in event_windows:
        # Park/hand HR rates (example: group by stadium, handedness, get HR rate over w days)
        if "stadium" in df_hist.columns and "p_throws" in df_hist.columns:
            merged[f"park_hand_hr_{w}"] = merged.apply(
                lambda row: hr_rate(
                    df_hist[df_hist['stadium'] == row['stadium']], 
                    ['p_throws'], w).mean() if pd.notnull(row['stadium']) else np.nan, axis=1)

        # Batter vs Pitcher Hand HR, Pitcher vs Batter Hand HR (stand/p_throws)
        if "stand" in df_hist.columns and "p_throws" in df_hist.columns:
            merged[f"b_vsp_hand_hr_{w}"] = merged.apply(
                lambda row: hr_rate(
                    df_hist[(df_hist['stand'] == row.get('stand', None)) & (df_hist['p_throws'] == row.get('p_throws', None))], 
                    ['stand','p_throws'], w).mean() if pd.notnull(row.get('stand', None)) else np.nan, axis=1)
            merged[f"p_vsb_hand_hr_{w}"] = merged.apply(
                lambda row: hr_rate(
                    df_hist[(df_hist['stand'] == row.get('stand', None)) & (df_hist['p_throws'] == row.get('p_throws', None))], 
                    ['p_throws','stand'], w).mean() if pd.notnull(row.get('p_throws', None)) else np.nan, axis=1)

        # Pitch Type HR Rates
        if "pitch_type" in df_hist.columns:
            merged[f"b_pitchtype_hr_{w}"] = merged.apply(
                lambda row: hr_rate(
                    df_hist[(df_hist['batter_id'] == row['batter_id']) & (df_hist['pitch_type'] == row.get('pitch_type', None))],
                    ['batter_id','pitch_type'], w).mean() if pd.notnull(row['batter_id']) else np.nan, axis=1)
            merged[f"p_pitchtype_hr_{w}"] = merged.apply(
                lambda row: hr_rate(
                    df_hist[(df_hist['pitcher_id'] == row.get('pitcher_id', None)) & (df_hist['pitch_type'] == row.get('pitch_type', None))],
                    ['pitcher_id','pitch_type'], w).mean() if pd.notnull(row.get('pitcher_id', None)) else np.nan, axis=1)

        # Add ALL b_ and p_ rolling stats for every stat column you listed
        for prefix, id_col in [("b_", "batter_id"), ("p_", "pitcher_id")]:
            for feat in ["launch_speed", "launch_angle", "hit_distance_sc", "woba_value",
                         "release_speed", "release_spin_rate", "spin_axis", "pfx_x", "pfx_z"]:
                try:
                    merged[f"{prefix}{feat}_{w}"] = merged.apply(
                        lambda row: generic_rolling_stat(
                            df_hist, id_col, feat, w).get(row[id_col], np.nan) if pd.notnull(row[id_col]) and feat in df_hist.columns else np.nan,
                        axis=1)
                except Exception:
                    merged[f"{prefix}{feat}_{w}"] = np.nan

    st.write("Advanced features added.")

    inc(f"🟢 Generated file: {merged.shape[0]} unique batters, {merged.shape[1]} columns (features).")
    st.dataframe(merged.head(10))

    st.download_button(
        "⬇️ Download Today's Event-Level CSV (Exact Format)",
