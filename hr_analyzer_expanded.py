import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config("🟦 Generate Today's Event-Level CSV", layout="wide")
st.title("🟦 Generate Today's Event-Level CSV (Batter + Pitcher Rolling Stats, 1 Row Per Matchup)")

st.markdown("""
**Instructions:**
- Upload Today's Lineups/Matchups CSV (must have `batter_id`, `pitcher_id`, `player_name`, etc).
- Upload Historical Event-Level CSV (must have `batter_id`, `pitcher_id`, `game_date`, and all stat columns).
- Output: ONE row per batter-pitcher matchup with ALL rolling/stat features for BOTH sides, parsed weather, and key info.
- Numeric columns are auto-corrected for string/decimal mix issues.
""")

today_file = st.file_uploader("Upload Today's Matchups/Lineups CSV", type=["csv"], key="today_csv")
hist_file = st.file_uploader("Upload Historical Event-Level CSV", type=["csv"], key="hist_csv")

# -- Your target columns (edit as needed) --
output_columns = [
    "team_code","game_date","game_number","mlb_id","player_name","batting_order","position",
    "weather","temp","wind_mph","wind_dir","condition","stadium","city",
    "batter_id","pitcher_id","p_throws",
    # Batting rolling stats
    "hard_hit_rate_20","sweet_spot_rate_20","barrel_rate_20","fb_rate_20","avg_exit_velo_20",
    "park_hand_hr_7","park_hand_hr_14","park_hand_hr_30",
    "b_vsp_hand_hr_3","p_vsb_hand_hr_3","b_vsp_hand_hr_5","p_vsb_hand_hr_5","b_vsp_hand_hr_7","p_vsb_hand_hr_7",
    "b_vsp_hand_hr_14","p_vsb_hand_hr_14","b_pitchtype_hr_3","p_pitchtype_hr_3","b_pitchtype_hr_5","p_pitchtype_hr_5",
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
    # Pitching rolling stats (add ALL p_*)
    "p_hard_hit_rate_20","p_sweet_spot_rate_20","p_barrel_rate_20","p_fb_rate_20","p_avg_exit_velo_20",
    "p_park_hand_hr_7","p_park_hand_hr_14","p_park_hand_hr_30",
    "p_b_vsp_hand_hr_3","p_p_vsb_hand_hr_3","p_b_vsp_hand_hr_5","p_p_vsb_hand_hr_5","p_b_vsp_hand_hr_7","p_p_vsb_hand_hr_7",
    "p_b_vsp_hand_hr_14","p_p_vsb_hand_hr_14","p_b_pitchtype_hr_3","p_p_pitchtype_hr_3","p_b_pitchtype_hr_5","p_p_pitchtype_hr_5",
    "p_b_pitchtype_hr_7","p_p_pitchtype_hr_7","p_b_pitchtype_hr_14","p_p_pitchtype_hr_14",
    "p_b_launch_speed_3","p_b_launch_speed_5","p_b_launch_speed_7","p_b_launch_speed_14",
    "p_b_launch_angle_3","p_b_launch_angle_5","p_b_launch_angle_7","p_b_launch_angle_14",
    "p_b_hit_distance_sc_3","p_b_hit_distance_sc_5","p_b_hit_distance_sc_7","p_b_hit_distance_sc_14",
    "p_b_woba_value_3","p_b_woba_value_5","p_b_woba_value_7","p_b_woba_value_14",
    "p_b_release_speed_3","p_b_release_speed_5","p_b_release_speed_7","p_b_release_speed_14",
    "p_b_release_spin_rate_3","p_b_release_spin_rate_5","p_b_release_spin_rate_7","p_b_release_spin_rate_14",
    "p_b_spin_axis_3","p_b_spin_axis_5","p_b_spin_axis_7","p_b_spin_axis_14",
    "p_b_pfx_x_3","p_b_pfx_x_5","p_b_pfx_x_7","p_b_pfx_x_14",
    "p_b_pfx_z_3","p_b_pfx_z_5","p_b_pfx_z_7","p_b_pfx_z_14",
    # ...add any other pitcher rolling windows as in your event-level CSV...
    "park","humidity" # Add/expand as needed
]

def parse_weather_string(df):
    """Parse weather-related fields from 'weather' string, e.g., '92 O CF 12-14 30% outdoor'."""
    if "weather" not in df.columns:
        return df
    weather_str = df["weather"].astype(str)
    # Temp (numbers at start)
    df["temp"] = weather_str.str.extract(r'(^\d{2,3})').astype(float)
    # Wind mph (digits before mph or N/S/E/W)
    df["wind_mph"] = weather_str.str.extract(r'(\d{1,3})(?=\s*[NSEWCF]{1,2})', flags=re.I, expand=False).astype(float)
    # Wind dir (N/S/E/W/CF)
    df["wind_dir"] = weather_str.str.extract(r'([NSEWCF]{1,2})', flags=re.I, expand=False)
    # Condition
    df["condition"] = weather_str.str.extract(r'(indoor|outdoor)', flags=re.I, expand=False).str.lower()
    return df

if today_file and hist_file:
    df_today = pd.read_csv(today_file)
    df_hist = pd.read_csv(hist_file)

    st.success(f"Today's Matchups file loaded: {df_today.shape[0]} rows, {df_today.shape[1]} columns.")
    st.success(f"Historical Event-Level file loaded: {df_hist.shape[0]} rows, {df_hist.shape[1]} columns.")

    # ---- Standardize column names ----
    df_today.columns = [str(c).strip().lower().replace(" ", "_") for c in df_today.columns]
    df_hist.columns = [str(c).strip().lower().replace(" ", "_") for c in df_hist.columns]

    # ---- Ensure key columns exist ----
    for col in ["batter_id", "pitcher_id"]:
        if col not in df_today.columns:
            st.error(f"Today's file must have '{col}' column.")
            st.stop()
        if col not in df_hist.columns:
            st.error(f"Historical file must have '{col}' column.")
            st.stop()

    # --- Fix decimal/string/integer mix for IDs
    for col in ["batter_id", "pitcher_id"]:
        df_today[col] = df_today[col].astype(str).str.strip().str.replace('.0', '', regex=False)
        df_hist[col] = df_hist[col].astype(str).str.strip().str.replace('.0', '', regex=False)

    # ---- Deduplicate today's batters (if duplicate batters in lineups) ----
    df_today = df_today.drop_duplicates(subset=["batter_id", "pitcher_id"])

    # ---- Get latest NON-NULL rolling stats for each batter ----
    batter_rolling_cols = [c for c in df_hist.columns if c.startswith("b_") or c in [
        "hard_hit_rate_20", "sweet_spot_rate_20", "barrel_rate_20", "fb_rate_20", "avg_exit_velo_20"
    ]]
    latest_batter = (
        df_hist.sort_values('game_date')
        .drop_duplicates(subset=["batter_id"], keep="last")
        [["batter_id"] + batter_rolling_cols]
    )

    # ---- Get latest NON-NULL rolling stats for each pitcher ----
    pitcher_rolling_cols = [c for c in df_hist.columns if c.startswith("p_") or c in [
        "p_hard_hit_rate_20", "p_sweet_spot_rate_20", "p_barrel_rate_20", "p_fb_rate_20", "p_avg_exit_velo_20"
    ]]
    latest_pitcher = (
        df_hist.sort_values('game_date')
        .drop_duplicates(subset=["pitcher_id"], keep="last")
        [["pitcher_id"] + pitcher_rolling_cols]
    )

    # ---- Merge batter and pitcher rolling stats into today's data ----
    merged = df_today.merge(latest_batter, on="batter_id", how="left", suffixes=('', '_bat'))
    merged = merged.merge(latest_pitcher, on="pitcher_id", how="left", suffixes=('', '_pit'))

    # ---- Weather parsing and fixups ----
    merged = parse_weather_string(merged)

    # ---- Deduplicate after merge ----
    merged = merged.drop_duplicates(subset=['batter_id', 'pitcher_id'], keep='first')

    # ---- Reindex to exact output column order (add missing cols as NaN, keep order) ----
    for col in output_columns:
        if col not in merged.columns:
            merged[col] = np.nan
    merged = merged.reindex(columns=output_columns)

    st.success(f"🟢 Generated file: {merged.shape[0]} unique batter-pitcher pairs, {merged.shape[1]} columns (features).")

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
