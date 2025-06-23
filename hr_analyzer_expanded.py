import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config("🟦 Generate Today's Event-Level CSV", layout="wide")
st.title("🟦 Generate Today's Event-Level CSV (All Rolling Stats, Pitcher+Batter, All Windows)")

st.markdown("""
**Instructions:**
- Upload Today's Lineups/Matchups CSV (must have `batter_id` or `mlb_id`, `player_name`, etc).
- Upload Historical Event-Level CSV (must have `batter_id`, `pitcher_id`, `game_date`, and all stat columns).
- Output: ONE row per batter with ALL rolling/stat features (3/7/14 days and events, batter & pitcher!).
- All numeric columns are auto-corrected for string/decimal mix issues.
""")

today_file = st.file_uploader("Upload Today's Matchups/Lineups CSV", type=["csv"], key="today_csv")
hist_file = st.file_uploader("Upload Historical Event-Level CSV", type=["csv"], key="hist_csv")

# ---- Output columns, update as needed ----
output_columns = [
    # ID/Meta
    "team_code","game_date","game_number","mlb_id","player_name","batting_order","position",
    "weather","temp","wind_mph","wind_dir","condition","stadium","city","batter_id","p_throws",

    # Core Batter Rolling
    "hard_hit_rate_3","hard_hit_rate_7","hard_hit_rate_14","hard_hit_rate_20",
    "sweet_spot_rate_3","sweet_spot_rate_7","sweet_spot_rate_14","sweet_spot_rate_20",
    "barrel_rate_3","barrel_rate_7","barrel_rate_14","barrel_rate_20",
    "fb_rate_3","fb_rate_7","fb_rate_14","fb_rate_20",
    "avg_exit_velo_3","avg_exit_velo_7","avg_exit_velo_14","avg_exit_velo_20",

    # All prior park/pitcher/pitchtype/batted-ball windows (3/5/7/14 for b_*/p_*)
    # [your previous output_columns here]
    # ...truncated for brevity, include all windows & your custom ones...

    # Example:
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
    "park","temp","wind_mph","wind_dir","humidity","condition"
]

# -- Which columns to use for custom rolling calculations --
batter_custom_stats = {
    "hard_hit_rate": "hard_hit",      # needs 1 if launch_speed >= 95, else 0
    "sweet_spot_rate": "sweet_spot",  # needs 1 if 8 <= launch_angle <= 32, else 0
    "barrel_rate": "barrel",          # classic: launch_speed >=98, 26<=angle<=30
    "fb_rate": "fly_ball",            # events that are fly_ball or line_drive
    "avg_exit_velo": "launch_speed"
}
pitcher_custom_stats = {
    "avg_exit_velo": "launch_speed"
}

# -- Windows for rolling (add more if you want) --
event_windows = [3, 7, 14, 20]
day_windows = [3, 7, 14, 20]  # Can be same as event_windows

def compute_event_rolling(df, id_col, date_col, stat_cols, windows):
    df = df.sort_values([id_col, date_col])
    out = []
    for pid, group in df.groupby(id_col):
        group = group.sort_values(date_col)
        for w in windows:
            rolling = group.rolling(w, min_periods=1)
            stats = {}
            for c in stat_cols:
                if c == "avg_exit_velo":
                    stats[f"{c}_{w}"] = rolling["launch_speed"].mean().values[-1] if "launch_speed" in group else np.nan
                elif c == "hard_hit_rate":
                    stats[f"{c}_{w}"] = rolling.apply(lambda x: np.mean(x >= 95), raw=True)["launch_speed"].values[-1] if "launch_speed" in group else np.nan
                elif c == "sweet_spot_rate":
                    stats[f"{c}_{w}"] = rolling.apply(lambda x: np.mean((x >= 8) & (x <= 32)), raw=True)["launch_angle"].values[-1] if "launch_angle" in group else np.nan
                elif c == "barrel_rate":
                    # MLB barrel: launch_speed >=98, launch_angle 26-30
                    stats[f"{c}_{w}"] = rolling.apply(lambda x: np.mean((group["launch_speed"] >= 98) & (group["launch_angle"].between(26, 30))), raw=False).values[-1] if "launch_speed" in group and "launch_angle" in group else np.nan
                elif c == "fb_rate":
                    stats[f"{c}_{w}"] = rolling.apply(lambda x: np.mean(group["bb_type"].isin(["fly_ball", "line_drive"])), raw=False).values[-1] if "bb_type" in group else np.nan
                else:
                    stats[f"{c}_{w}"] = rolling[c].mean().values[-1] if c in group else np.nan
            stats[id_col] = pid
            stats[date_col] = group[date_col].max()
            out.append(stats)
    return pd.DataFrame(out)

# ---- Start main logic ----
if today_file and hist_file:
    df_today = pd.read_csv(today_file)
    df_hist = pd.read_csv(hist_file)

    st.success(f"Today's Matchups file loaded: {df_today.shape[0]} rows, {df_today.shape[1]} columns.")
    st.success(f"Historical Event-Level file loaded: {df_hist.shape[0]} rows, {df_hist.shape[1]} columns.")

    # ---- Clean/standardize column names ----
    df_today.columns = [str(c).strip().lower().replace(" ", "_") for c in df_today.columns]
    df_hist.columns = [str(c).strip().lower().replace(" ", "_") for c in df_hist.columns]

    # ---- Get batter_id ----
    id_cols_today = ['batter_id', 'mlb_id']
    id_col_today = next((c for c in id_cols_today if c in df_today.columns), None)
    if id_col_today is None:
        st.error("Today's file must have 'batter_id' or 'mlb_id' column.")
        st.stop()
    if 'batter_id' not in df_hist.columns:
        st.error("Historical file must have 'batter_id' column.")
        st.stop()

    # --- Fix decimal/string/integer mix for batter_id ---
    df_today['batter_id'] = df_today[id_col_today].astype(str).str.strip().str.replace('.0', '', regex=False)
    df_hist['batter_id'] = df_hist['batter_id'].astype(str).str.strip().str.replace('.0', '', regex=False)
    if "pitcher_id" in df_hist.columns:
        df_hist['pitcher_id'] = df_hist['pitcher_id'].astype(str).str.strip().str.replace('.0', '', regex=False)

    # ---- Deduplicate today's batters (if duplicate batters in lineups) ----
    df_today = df_today.drop_duplicates(subset=["batter_id"])

    # ---- Weather parsing ----
    if "weather" in df_today.columns:
        weather_str = df_today["weather"].fillna("").astype(str)
        df_today["temp"] = weather_str.str.extract(r'(\d{2,3})\s')[0]
        # Attempt to get wind_mph, wind_dir
        df_today["wind_mph"] = weather_str.str.extract(r'(\d+)\s*(?:mph)?')[0]
        df_today["wind_dir"] = weather_str.str.extract(r'(CF|RF|LF|N|S|E|W|NE|NW|SE|SW)', flags=re.I, expand=False)
        df_today["condition"] = weather_str.str.extract(r'(indoor|outdoor)', flags=re.I, expand=False)
        # Fill blanks
        df_today["temp"] = pd.to_numeric(df_today["temp"], errors="coerce")
        df_today["wind_mph"] = pd.to_numeric(df_today["wind_mph"], errors="coerce")
        df_today["wind_dir"] = df_today["wind_dir"].str.upper()
        df_today["condition"] = df_today["condition"].str.lower()
        # Optionally more weather parsing here

    # ---- Rolling stats for BATTERS (event-based windows) ----
    batter_event = compute_event_rolling(df_hist, "batter_id", "game_date", list(batter_custom_stats.keys()), event_windows)
    # Rename for clarity (e.g. hard_hit_rate_3 => hard_hit_rate_3)
    # (Already named well)

    # ---- Rolling stats for PITCHERS (event-based windows) ----
    if "pitcher_id" in df_hist.columns:
        df_pitcher_hist = df_hist.copy()
        if "batter_id" in df_pitcher_hist.columns:
            df_pitcher_hist = df_pitcher_hist.drop(columns=["batter_id"])
        df_pitcher_hist = df_pitcher_hist.rename(columns={"pitcher_id": "batter_id"})
        pitcher_event = compute_event_rolling(df_pitcher_hist, "batter_id", "game_date", list(pitcher_custom_stats.keys()), event_windows)
    else:
        pitcher_event = pd.DataFrame()

    # ---- Merge today's lineups with latest event stats ----
    merged = df_today.merge(batter_event, on="batter_id", how="left", suffixes=('', '_batter_event'))
    if not pitcher_event.empty:
        merged = merged.merge(pitcher_event, on="batter_id", how="left", suffixes=('', '_pitcher_event'))

    # ---- Deduplicate after merge, just in case ----
    merged = merged.drop_duplicates(subset=['batter_id'], keep='first')

    # ---- Force all relevant stats to numeric (auto-fix float/string mix) ----
    for c in merged.columns:
        if c not in ["batter_id", "player_name", "player_name_hist"]:
            merged[c] = pd.to_numeric(merged[c], errors='ignore')

    # ---- Reindex to exact output column order (add missing cols as NaN, keep order) ----
    for col in output_columns:
        if col not in merged.columns:
            merged[col] = np.nan
    merged = merged.reindex(columns=output_columns)

    st.success(f"🟢 Generated file: {merged.shape[0]} unique batters, {merged.shape[1]} columns (features).")

    # ---- Preview output ----
    st.dataframe(merged.head(10))

    # ---- Download button ----
    st.download_button(
        "⬇️ Download Today's Event-Level CSV (Exact Format)",
        data=merged.to_csv(index=False),
        file_name="event_level_today_full.csv"
    )
else:
    st.info("Please upload BOTH today's matchups/lineups and historical event-level CSV.")
