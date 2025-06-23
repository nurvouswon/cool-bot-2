import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config("🟦 Generate Today's Event-Level CSV", layout="wide")
st.title("🟦 Generate Today's Event-Level CSV (All Rolling Stats, Weather, 1 Row Per Batter)")

st.markdown("""
**Instructions:**
- Upload Today's Lineups/Matchups CSV (must have `batter_id` or `mlb_id`, `player_name`, etc. and `weather`).
- Upload Historical Event-Level CSV (must have `batter_id`, `game_date`, and all stat columns).
- Output: ONE row per batter with all rolling/stat features, calculated from their event history.
- Output columns will match the format needed for prediction, including parsed weather.
- All numeric columns are auto-corrected for string/decimal mix issues.
""")

# -- Paste your target columns here (edit as needed) --
output_columns = [
    "team_code","game_date","game_number","mlb_id","player_name","batting_order","position",
    "weather","temp","wind_mph","wind_dir","condition","stadium","city",
    "batter_id","p_throws",
    "hard_hit_rate_20","sweet_spot_rate_20","barrel_rate_20","fb_rate_20","avg_exit_velo_20",
    "park_hand_hr_7","park_hand_hr_14","park_hand_hr_30",
    "b_vsp_hand_hr_3","p_vsb_hand_hr_3","b_vsp_hand_hr_5","p_vsb_hand_hr_5","b_vsp_hand_hr_7","p_vsb_hand_hr_7",
    "b_vsp_hand_hr_14","p_vsb_hand_hr_14",
    "b_pitchtype_hr_3","p_pitchtype_hr_3","b_pitchtype_hr_5","p_pitchtype_hr_5","b_pitchtype_hr_7","p_pitchtype_hr_7",
    "b_pitchtype_hr_14","p_pitchtype_hr_14",
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
    "p_pfx_z_3","p_pfx_z_5","p_pfx_z_7","p_pfx_z_14"
]

# ---- Weather Parser ----
def parse_weather_string(weather_str):
    # Example: "89 N RL 5-6 40% outdoor"
    if pd.isna(weather_str) or not isinstance(weather_str, str):
        return np.nan, np.nan, np.nan, np.nan
    try:
        parts = weather_str.strip().split()
        temp = int(parts[0]) if parts[0].isdigit() else np.nan
        wind_dir = parts[2] if len(parts) > 2 else np.nan
        wind_range = parts[3] if len(parts) > 3 else ""
        wind_mph = (
            np.mean([float(w) for w in re.split("[-/]", wind_range) if w.replace('.', '', 1).isdigit()])
            if wind_range else np.nan
        )
        condition = parts[-1] if len(parts) > 5 else (parts[-1] if len(parts) > 4 else "")
        return temp, wind_dir, wind_mph, condition
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

# ---- Calculate barrel % etc if not present ----
def calculate_barrel_rate(group, window=20):
    # Barrel = 98+ EV, 26-30° LA. Use rolling window.
    mask = (group['launch_speed'] >= 98) & (group['launch_angle'].between(26, 30))
    rolling = mask.rolling(window, min_periods=1).mean()
    return rolling

def calculate_fb_rate(group, window=20):
    mask = group['launch_angle'] >= 25
    rolling = mask.rolling(window, min_periods=1).mean()
    return rolling

def calculate_avg_ev(group, window=20):
    rolling = group['launch_speed'].rolling(window, min_periods=1).mean()
    return rolling

# ---- Main logic ----
today_file = st.file_uploader("Upload Today's Matchups/Lineups CSV", type=["csv"], key="today_csv")
hist_file = st.file_uploader("Upload Historical Event-Level CSV", type=["csv"], key="hist_csv")

rolling_stat_cols = ["hard_hit_rate_20", "sweet_spot_rate_20", "barrel_rate_20", "fb_rate_20", "avg_exit_velo_20"]

if today_file and hist_file:
    df_today = pd.read_csv(today_file)
    df_hist = pd.read_csv(hist_file)

    st.success(f"Today's Matchups file loaded: {df_today.shape[0]} rows, {df_today.shape[1]} columns.")
    st.success(f"Historical Event-Level file loaded: {df_hist.shape[0]} rows, {df_hist.shape[1]} columns.")

    # Clean/standardize columns
    df_today.columns = [str(c).strip().lower().replace(" ", "_") for c in df_today.columns]
    df_hist.columns = [str(c).strip().lower().replace(" ", "_") for c in df_hist.columns]

    # Get batter_id
    id_cols_today = ['batter_id', 'mlb_id']
    id_col_today = next((c for c in id_cols_today if c in df_today.columns), None)
    if id_col_today is None:
        st.error("Today's file must have 'batter_id' or 'mlb_id' column.")
        st.stop()
    if 'batter_id' not in df_hist.columns:
        st.error("Historical file must have 'batter_id' column.")
        st.stop()

    df_today['batter_id'] = df_today[id_col_today].astype(str).str.strip().str.replace('.0', '', regex=False)
    df_hist['batter_id'] = df_hist['batter_id'].astype(str).str.strip().str.replace('.0', '', regex=False)

    # ---- Weather parsing (add new columns) ----
    weather_data = df_today['weather'].apply(parse_weather_string)
    df_today['temp'] = weather_data.apply(lambda x: x[0])
    df_today['wind_dir'] = weather_data.apply(lambda x: x[1])
    df_today['wind_mph'] = weather_data.apply(lambda x: x[2])
    df_today['condition'] = weather_data.apply(lambda x: x[3])

    # ---- Calculate missing rolling stats ----
    for stat, func in [
        ('barrel_rate_20', calculate_barrel_rate),
        ('fb_rate_20', calculate_fb_rate),
        ('avg_exit_velo_20', calculate_avg_ev),
    ]:
        if stat not in df_hist.columns:
            # Compute if missing
            df_hist[stat] = (
                df_hist.groupby('batter_id', group_keys=False)
                .apply(lambda g: func(g).shift(1))  # shift to not include current event
            )

    # ---- If hard_hit_rate_20 or sweet_spot_rate_20 missing, fill as above (should exist) ----
    for stat in ['hard_hit_rate_20', 'sweet_spot_rate_20']:
        if stat not in df_hist.columns and 'launch_speed' in df_hist.columns:
            if stat == 'hard_hit_rate_20':
                # 95+ EV is hard hit
                df_hist[stat] = (
                    df_hist.groupby('batter_id', group_keys=False)
                    .apply(lambda g: (g['launch_speed'] >= 95).rolling(20, min_periods=1).mean().shift(1))
                )
            elif stat == 'sweet_spot_rate_20':
                # LA between 8-32 is sweet spot
                df_hist[stat] = (
                    df_hist.groupby('batter_id', group_keys=False)
                    .apply(lambda g: (g['launch_angle'].between(8, 32)).rolling(20, min_periods=1).mean().shift(1))
                )

    # ---- Get the latest row per batter with best data ----
    df_hist['game_date'] = pd.to_datetime(df_hist['game_date'], errors='coerce')
    latest_non_null = []
    for batter_id, group in df_hist.sort_values('game_date').groupby('batter_id'):
        non_null_row = group.dropna(subset=rolling_stat_cols)
        if not non_null_row.empty:
            latest_non_null.append(non_null_row.iloc[-1])
        else:
            latest_non_null.append(group.iloc[-1])
    latest_feats = pd.DataFrame(latest_non_null).drop_duplicates(subset=['batter_id'], keep='last')

    # Numeric auto-fix
    for c in latest_feats.columns:
        if c not in ["batter_id", "player_name", "player_name_hist"]:
            latest_feats[c] = pd.to_numeric(latest_feats[c], errors='ignore')

    # Merge lineups and stat features
    merged = df_today.merge(latest_feats, on="batter_id", how="left", suffixes=('', '_hist'))

    # Deduplicate
    merged = merged.drop_duplicates(subset=['batter_id'], keep='first')

    # Reindex to exact output column order (add missing cols as NaN)
    for col in output_columns:
        if col not in merged.columns:
            merged[col] = np.nan
    merged = merged.reindex(columns=output_columns)

    st.success(f"🟢 Generated file: {merged.shape[0]} unique batters, {merged.shape[1]} columns (features).")
    st.dataframe(merged.head(10))

    st.download_button(
        "⬇️ Download Today's Event-Level CSV (Exact Format)",
        data=merged.to_csv(index=False),
        file_name="event_level_today_full.csv"
    )
else:
    st.info("Please upload BOTH today's matchups/lineups and historical event-level CSV.")
