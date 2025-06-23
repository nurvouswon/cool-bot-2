import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config("🟦 Generate Today's Event-Level CSV", layout="wide")
st.title("🟦 Generate Today's Event-Level CSV (All Rolling Stats, 1 Row Per Batter)")

st.markdown("""
**Instructions:**
- Upload Today's Lineups/Matchups CSV (must have `batter_id` or `mlb_id`, `player_name`, etc).
- Upload Historical Event-Level CSV (must have `batter_id`, `game_date`, and all stat columns).
- Output: ONE row per batter with all rolling/stat features, calculated from their event history.
- Weather, barrel%, hard hit%, sweet spot%, FB%, and avg EV are parsed or calculated.
- All numeric columns are auto-corrected for string/decimal mix issues.
""")

today_file = st.file_uploader("Upload Today's Matchups/Lineups CSV", type=["csv"], key="today_csv")
hist_file = st.file_uploader("Upload Historical Event-Level CSV", type=["csv"], key="hist_csv")

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

rolling_stat_cols = ["hard_hit_rate_20", "sweet_spot_rate_20", "barrel_rate_20", "fb_rate_20", "avg_exit_velo_20"]

def compute_rolling_batter_metrics(df, group_col='batter_id', window=20):
    """Adds rolling 20-barrel rate, hard hit rate, sweet spot rate, FB%, and avg EV to event-level df (in-place)."""
    df = df.sort_values(['batter_id','game_date'])
    # Event-level features:
    is_bbe = df['launch_speed'].notnull() & df['launch_angle'].notnull()
    df['is_bbe'] = is_bbe
    df['is_hard_hit'] = df['launch_speed'] >= 95
    df['is_barrel'] = ((df['launch_speed'] >= 98) &
        (df['launch_speed'] <= 130) &
        (df['launch_angle'] >= 26) & (df['launch_angle'] <= 30)
    )
    df['is_sweet_spot'] = (df['launch_angle'] >= 8) & (df['launch_angle'] <= 32)
    df['is_fb'] = (df['launch_angle'] >= 20)
    df['ev'] = df['launch_speed']

    for stat, col in [
        ('is_hard_hit', 'hard_hit_rate_20'),
        ('is_sweet_spot', 'sweet_spot_rate_20'),
        ('is_barrel', 'barrel_rate_20'),
        ('is_fb', 'fb_rate_20')
    ]:
        df[col] = (
            df.groupby(group_col)[stat]
              .transform(lambda x: x.rolling(window, min_periods=5).mean())
        )
    df['avg_exit_velo_20'] = (
        df.groupby(group_col)['ev'].transform(lambda x: x.rolling(window, min_periods=5).mean())
    )
    # Convert event-level to batter-latest
    return df

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

    # ---- Deduplicate today's batters (if duplicate batters in lineups) ----
    df_today = df_today.drop_duplicates(subset=["batter_id"])

    # ---- Parse/clean weather features from string if present ----
    if "weather" in df_today.columns:
        weather_str = df_today["weather"].fillna("").astype(str)
        # Temp
        df_today["temp"] = weather_str.str.extract(r'(\d{2,3})\s')[0]
        df_today["temp"] = pd.to_numeric(df_today["temp"], errors='coerce')
        # Wind mph
        df_today["wind_mph"] = weather_str.str.extract(r'(\d+)\s*mph')[0]
        df_today["wind_mph"] = pd.to_numeric(df_today["wind_mph"], errors='coerce')
        # Wind dir (N, S, E, W, CF, RF, LF etc)
        df_today["wind_dir"] = weather_str.str.extract(r'([nswecf]{1,2})\s', flags=re.I, expand=False)
        # Condition (indoor/outdoor, etc)
        df_today["condition"] = weather_str.str.extract(r'(indoor|outdoor)', flags=re.I, expand=False)
    # Fallback to existing columns if already present

    # ---- Compute rolling batter metrics if not present ----
    missing_rolling = any(col not in df_hist.columns for col in rolling_stat_cols)
    if missing_rolling:
        st.info("Computing rolling stats for barrel%, hard hit%, sweet spot%, FB%, avg EV (20-event window)...")
        df_hist = compute_rolling_batter_metrics(df_hist, group_col='batter_id', window=20)

    # ---- Get the latest event row WITH non-null rolling stats for each batter ----
    if 'game_date' in df_hist.columns:
        df_hist['game_date'] = pd.to_datetime(df_hist['game_date'], errors='coerce')
        latest_non_null = []
        for batter_id, group in df_hist.sort_values('game_date').groupby('batter_id'):
            # Try to get last row where all rolling stats are present (not null)
            non_null_row = group.dropna(subset=rolling_stat_cols)
            if not non_null_row.empty:
                latest_non_null.append(non_null_row.iloc[-1])
            else:
                # If all are null, just take last row (old behavior)
                latest_non_null.append(group.iloc[-1])
        latest_feats = pd.DataFrame(latest_non_null)
    else:
        latest_feats = df_hist.drop_duplicates(subset=['batter_id'], keep='last')

    # ---- Remove duplicated batter_ids in latest_feats (keep last, safest) ----
    latest_feats = latest_feats.drop_duplicates(subset=['batter_id'], keep='last')

    # ---- Force all relevant stats to numeric (auto-fix float/string mix) ----
    for c in latest_feats.columns:
        if c not in ["batter_id", "player_name", "player_name_hist"]:
            latest_feats[c] = pd.to_numeric(latest_feats[c], errors='ignore')

    # ---- Merge today's lineups with latest event stats ----
    merged = df_today.merge(latest_feats, on="batter_id", how="left", suffixes=('', '_hist'))

    # ---- Deduplicate after merge, just in case ----
    merged = merged.drop_duplicates(subset=['batter_id'], keep='first')

    # ---- Reindex to exact output column order (add missing cols as NaN, keep order) ----
    for col in output_columns:
        if col not in merged.columns:
            merged[col] = np.nan
    merged = merged.reindex(columns=output_columns)

    # ---- REMOVE DUPLICATE COLUMNS BY NAME (KEEP FIRST) ----
    _, idx = np.unique(merged.columns, return_index=True)
    merged = merged.iloc[:, np.sort(idx)]

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
