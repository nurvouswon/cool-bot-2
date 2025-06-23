import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config("🟦 Generate Today's Event-Level CSV", layout="wide")
st.title("🟦 Generate Today's Event-Level CSV (All Rolling Stats, 1 Row Per Batter/Pitcher)")

st.markdown("""
**Instructions:**
- Upload Today's Lineups/Matchups CSV (must have `batter_id` or `mlb_id`, `player_name`, etc).
- Upload Historical Event-Level CSV (must have `batter_id`, `game_date`, and all stat columns).
- Output: ONE row per batter with all rolling/stat features (last N events & last N days), calculated from their event history.
- Output columns will match the format needed for prediction, and all weather columns are parsed from the matchup file if present.
""")

today_file = st.file_uploader("Upload Today's Matchups/Lineups CSV", type=["csv"], key="today_csv")
hist_file = st.file_uploader("Upload Historical Event-Level CSV", type=["csv"], key="hist_csv")

# -- Define your stat column roots (batter and pitcher, don't include rolling suffix here!) --
batter_stat_roots = [
    "hard_hit_rate", "sweet_spot_rate", "barrel_rate", "fb_rate", "avg_exit_velo",
    "b_launch_speed", "b_launch_angle", "b_hit_distance_sc", "b_woba_value", "b_release_speed",
    "b_release_spin_rate", "b_spin_axis", "b_pfx_x", "b_pfx_z",
    "park_hand_hr", "b_vsp_hand_hr", "b_pitchtype_hr"
]
pitcher_stat_roots = [
    "p_launch_speed", "p_launch_angle", "p_hit_distance_sc", "p_woba_value", "p_release_speed",
    "p_release_spin_rate", "p_spin_axis", "p_pfx_x", "p_pfx_z", "p_pitchtype_hr", "p_vsb_hand_hr"
]

# -- Define the windows you want --
event_windows = [3, 5, 7, 14, 20]
day_windows = [3, 5, 7, 14]

# -- Build all output rolling columns dynamically --
def build_rolling_cols(roots, windows, suffix=""):
    cols = []
    for root in roots:
        for w in windows:
            cols.append(f"{root}_{w}{suffix}")
    return cols

batter_event_cols = build_rolling_cols(batter_stat_roots, event_windows)
pitcher_event_cols = build_rolling_cols(pitcher_stat_roots, event_windows)
batter_day_cols = build_rolling_cols(batter_stat_roots, day_windows, "d")
pitcher_day_cols = build_rolling_cols(pitcher_stat_roots, day_windows, "d")

# -- Add core info columns --
output_columns = [
    "team_code","game_date","game_number","mlb_id","player_name","batting_order","position",
    "weather","temp","wind_mph","wind_dir","condition","stadium","city",
    "batter_id","p_throws"
] + batter_event_cols + batter_day_cols + pitcher_event_cols + pitcher_day_cols

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
    # Standardize batter_id in both
    df_today['batter_id'] = df_today[id_col_today].astype(str).str.strip().str.replace('.0', '', regex=False)
    df_hist['batter_id'] = df_hist['batter_id'].astype(str).str.strip().str.replace('.0', '', regex=False)

    # ---- Deduplicate today's batters (if duplicate batters in lineups) ----
    df_today = df_today.drop_duplicates(subset=["batter_id"])

    # ---- Parse/standardize weather, temp, wind columns (from `weather` column in today_file) ----
    # Handles e.g. "92 O CF 12-14 30% outdoor" etc
    if "weather" in df_today.columns:
        weather_str = df_today["weather"].astype(str)
        # Try to extract temp (first number)
        df_today["temp"] = weather_str.str.extract(r'(\d+)', expand=False)
        df_today["temp"] = pd.to_numeric(df_today["temp"], errors='coerce')
        # Wind mph (after CF/LF/RF/etc)
        # Try to extract wind range (min-max)
        wind_range = weather_str.str.extract(r'(\d+)\s*-\s*(\d+)', expand=True)
        wind_range = wind_range.apply(pd.to_numeric, errors='coerce')
        df_today["wind_mph"] = wind_range.mean(axis=1)

        # Fallback: if not present, try single wind speed (e.g. "13" in "92 O CF 12-14 30% outdoor")
        no_wind = df_today["wind_mph"].isnull()
        df_today.loc[no_wind, "wind_mph"] = weather_str.str.extract(r'(\d+)', expand=False).astype(float)
        # Wind direction (e.g. "CF", "LF", "RF", "N", "E", etc)
        df_today["wind_dir"] = weather_str.str.extract(r'([A-Z]{1,2})\s*', flags=re.I, expand=False)
        # Condition
        df_today["condition"] = weather_str.str.extract(r'(indoor|outdoor)', flags=re.I, expand=False).str.lower()

    # ---- Make sure date is pd.Timestamp everywhere ----
    if 'game_date' in df_hist.columns:
        df_hist['game_date'] = pd.to_datetime(df_hist['game_date'], errors='coerce')
    if 'game_date' in df_today.columns:
        df_today['game_date'] = pd.to_datetime(df_today['game_date'], errors='coerce')

    # ---- Function to compute event rolling means ----
    def compute_event_rolling(df, id_col, date_col, stat_cols, windows):
        df = df.sort_values([id_col, date_col])
        out = []
        for pid, group in df.groupby(id_col):
            group = group.sort_values(date_col)
            row = {"batter_id": pid}
            for stat in stat_cols:
                col_vals = pd.to_numeric(group[stat], errors='coerce') if stat in group else pd.Series(dtype=float)
                for w in windows:
                    if len(col_vals) >= w:
                        row[f"{stat}_{w}"] = col_vals.iloc[-w:].mean()
            out.append(row)
        return pd.DataFrame(out)

    # ---- Function to compute day rolling means ----
    def compute_day_rolling(df, id_col, date_col, stat_cols, day_windows):
        df = df.sort_values([id_col, date_col])
        out = []
        for pid, group in df.groupby(id_col):
            group = group.sort_values(date_col)
            for w in day_windows:
                cutoff = group[date_col].max() - pd.Timedelta(days=w-1)
                window_group = group[group[date_col] >= cutoff]
                row = {"batter_id": pid}
                for stat in stat_cols:
                    if stat in window_group:
                        vals = pd.to_numeric(window_group[stat], errors='coerce')
                        row[f"{stat}_{w}d"] = vals.mean() if not vals.empty else np.nan
                out.append(row)
        df_all = pd.DataFrame(out).groupby("batter_id").last().reset_index()
        return df_all

    # ---- Prepare stat columns to roll ----
    all_batter_stats = [c for c in df_hist.columns if any(c.startswith(root) for root in batter_stat_roots)]
    all_pitcher_stats = [c for c in df_hist.columns if any(c.startswith(root) for root in pitcher_stat_roots)]

    # ---- Compute batter rolling windows (events) ----
    batter_event = compute_event_rolling(df_hist, "batter_id", "game_date", all_batter_stats, event_windows)
    batter_day = compute_day_rolling(df_hist, "batter_id", "game_date", all_batter_stats, day_windows)
    # ---- Compute pitcher rolling windows (events) ----
    if "pitcher_id" in df_hist.columns:
        df_hist['pitcher_id'] = df_hist['pitcher_id'].astype(str).str.strip().str.replace('.0', '', regex=False)
        pitcher_event = compute_event_rolling(df_hist.rename(columns={"pitcher_id": "batter_id"}), "batter_id", "game_date", all_pitcher_stats, event_windows)
        pitcher_day = compute_day_rolling(df_hist.rename(columns={"pitcher_id": "batter_id"}), "batter_id", "game_date", all_pitcher_stats, day_windows)
    else:
        pitcher_event = pd.DataFrame()
        pitcher_day = pd.DataFrame()

    # ---- Merge all stats into latest_feats (by batter_id) ----
    latest_feats = batter_event.merge(batter_day, on="batter_id", how="outer")
    if not pitcher_event.empty:
        latest_feats = latest_feats.merge(pitcher_event, on="batter_id", how="left")
    if not pitcher_day.empty:
        latest_feats = latest_feats.merge(pitcher_day, on="batter_id", how="left")

    # ---- Merge today's lineups with latest event stats ----
    merged = df_today.merge(latest_feats, on="batter_id", how="left", suffixes=('', '_hist'))
    merged = merged.drop_duplicates(subset=['batter_id'], keep='first')

    # ---- Add missing output columns as NaN, keep order ----
    for col in output_columns:
        if col not in merged.columns:
            merged[col] = np.nan
    merged = merged.reindex(columns=output_columns)

    st.success(f"🟢 Generated file: {merged.shape[0]} unique batters, {merged.shape[1]} columns (features).")
    st.dataframe(merged.head(10))

    # ---- Download button ----
    st.download_button(
        "⬇️ Download Today's Event-Level CSV (Full Stat Windows, All Debug/Weather Fixes)",
        data=merged.to_csv(index=False),
        file_name="event_level_today_full.csv"
    )
else:
    st.info("Please upload BOTH today's matchups/lineups and historical event-level CSV.")
