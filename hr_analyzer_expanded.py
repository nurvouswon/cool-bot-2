import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import defaultdict

st.set_page_config("🟦 Generate Today's Event-Level CSV (Optimized)", layout="wide")
st.title("🟦 Generate Today's Event-Level CSV (FAST, Optimized for Large Data)")

st.markdown("""
**Instructions:**
- Upload Today's Lineups/Matchups CSV (must have `batter_id` or `mlb_id`, `player_name`, etc).
- Upload Historical Event-Level CSV (must have `batter_id`, `pitcher_id`, `pitch_type`, `game_date`, and all stat columns).
- Output: ONE row per batter with ALL rolling/stat features (3/7/14/20, batter & pitcher, per pitch type and overall).
- All numeric columns are auto-corrected for string/decimal mix issues.
""")

today_file = st.file_uploader("Upload Today's Matchups/Lineups CSV", type=["csv"], key="today_csv")
hist_file = st.file_uploader("Upload Historical Event-Level CSV", type=["csv"], key="hist_csv")

output_columns = [
    "team_code","game_date","game_number","mlb_id","player_name","batting_order","position",
    "weather","temp","wind_mph","wind_dir","condition","stadium","city","batter_id","p_throws",
    # Batting
    # Pitching
    # (dynamically expanded below)
]

event_windows = [3, 7, 14, 20]
main_pitch_types = ["ff", "sl", "cu", "ch", "si", "fc", "fs", "st", "sinker", "splitter", "sweeper"]

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

# ---- New Fast Rolling Stat Function ----
def fast_rolling_stats(df, id_col, date_col, windows, pitch_types=None, prefix=""):
    """
    Compute rolling stats for each id_col (batter or pitcher), all windows, overall and per-pitch-type, vectorized.
    Returns: DataFrame with [id_col] as index, one row per entity with all rolling features.
    """
    # Ensure dates and numerics are correct
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['launch_speed'] = pd.to_numeric(df['launch_speed'], errors='coerce')
    df['launch_angle'] = pd.to_numeric(df['launch_angle'], errors='coerce')
    if 'pitch_type' in df.columns:
        df['pitch_type'] = df['pitch_type'].astype(str).str.lower().str.strip()
    results = defaultdict(dict)
    
    # Sort for rolling
    df = df.sort_values([id_col, date_col])
    gb = df.groupby(id_col, group_keys=False)
    for w in windows:
        # Overall
        results[f"{prefix}avg_exit_velo_{w}"] = gb['launch_speed'].apply(lambda x: x.rolling(w, min_periods=1).mean())
        results[f"{prefix}hard_hit_rate_{w}"] = gb['launch_speed'].apply(lambda x: x.rolling(w, min_periods=1).apply(lambda y: np.mean(y >= 95)))
        results[f"{prefix}barrel_rate_{w}"] = gb.apply(lambda group: 
            ((group['launch_speed'] >= 98) & 
             (group['launch_angle'] >= 26) & 
             (group['launch_angle'] <= 30)).rolling(w, min_periods=1).mean().values
        )
        results[f"{prefix}fb_rate_{w}"] = gb['launch_angle'].apply(lambda x: x.rolling(w, min_periods=1).apply(lambda y: np.mean(y >= 25)))
        results[f"{prefix}sweet_spot_rate_{w}"] = gb['launch_angle'].apply(lambda x: x.rolling(w, min_periods=1).apply(lambda y: np.mean((y >= 8) & (y <= 32))))
    # Per-pitch-type
    if pitch_types is not None and "pitch_type" in df.columns:
        for pt in pitch_types:
            pt_mask = (df["pitch_type"] == pt)
            gb_pt = df[pt_mask].groupby(id_col, group_keys=False)
            for w in windows:
                results[f"{prefix}avg_exit_velo_{pt}_{w}"] = gb_pt['launch_speed'].apply(lambda x: x.rolling(w, min_periods=1).mean())
                results[f"{prefix}hard_hit_rate_{pt}_{w}"] = gb_pt['launch_speed'].apply(lambda x: x.rolling(w, min_periods=1).apply(lambda y: np.mean(y >= 95)))
                results[f"{prefix}barrel_rate_{pt}_{w}"] = gb_pt.apply(lambda group: 
                    ((group['launch_speed'] >= 98) & 
                     (group['launch_angle'] >= 26) & 
                     (group['launch_angle'] <= 30)).rolling(w, min_periods=1).mean().values
                )
                results[f"{prefix}fb_rate_{pt}_{w}"] = gb_pt['launch_angle'].apply(lambda x: x.rolling(w, min_periods=1).apply(lambda y: np.mean(y >= 25)))
                results[f"{prefix}sweet_spot_rate_{pt}_{w}"] = gb_pt['launch_angle'].apply(lambda x: x.rolling(w, min_periods=1).apply(lambda y: np.mean((y >= 8) & (y <= 32))))
    # Combine results
    res_df = pd.DataFrame(results)
    res_df[id_col] = df[id_col].values
    res_df[date_col] = df[date_col].values
    # Take latest row for each id_col (most recent game/event)
    res_df = res_df.sort_values([id_col, date_col]).groupby(id_col).tail(1)
    res_df = res_df.set_index(id_col)
    return res_df

def show_diag(title, df):
    st.write(f"#### {title}")
    st.write(df.shape)
    st.dataframe(df.head(8))

if today_file and hist_file:
    df_today = pd.read_csv(today_file)
    df_hist = pd.read_csv(hist_file)

    st.success(f"Today's Matchups file loaded: {df_today.shape[0]} rows, {df_today.shape[1]} columns.")
    st.success(f"Historical Event-Level file loaded: {df_hist.shape[0]} rows, {df_hist.shape[1]} columns.")

    # --- Clean columns ---
    df_today.columns = [str(c).strip().lower().replace(" ", "_") for c in df_today.columns]
    df_hist.columns = [str(c).strip().lower().replace(" ", "_") for c in df_hist.columns]

    id_cols_today = ['batter_id', 'mlb_id']
    id_col_today = next((c for c in id_cols_today if c in df_today.columns), None)
    if id_col_today is None:
        st.error("Today's file must have 'batter_id' or 'mlb_id' column.")
        st.stop()
    if 'batter_id' not in df_hist.columns or 'pitcher_id' not in df_hist.columns:
        st.error("Historical file must have both 'batter_id' and 'pitcher_id' columns.")
        st.stop()
    # Clean ID columns
    for c in ['batter_id', 'pitcher_id']:
        if c in df_hist.columns:
            df_hist[c] = df_hist[c].astype(str).str.strip().str.replace('.0', '', regex=False)
    df_today['batter_id'] = df_today[id_col_today].astype(str).str.strip().str.replace('.0', '', regex=False)
    if 'pitcher_id' in df_today.columns:
        df_today['pitcher_id'] = df_today['pitcher_id'].astype(str).str.strip().str.replace('.0', '', regex=False)
    df_today = df_today.drop_duplicates(subset=["batter_id"])
    df_today = parse_weather_fields(df_today)
    if 'game_date' not in df_hist.columns:
        st.error("Historical file must have a 'game_date' column.")
        st.stop()
    df_hist['game_date'] = pd.to_datetime(df_hist['game_date'], errors='coerce')

    # ---- DIAGNOSTIC: Show input data ----
    show_diag("Today's Input Data", df_today)
    show_diag("Historical Event-Level Data", df_hist)

    # ---- Compute rolling stats: BATTER ----
    st.info("Computing batter rolling stats (vectorized, fast)...")
    batter_event = fast_rolling_stats(
        df_hist, "batter_id", "game_date", event_windows, main_pitch_types, prefix=""
    )
    show_diag("Batter Rolling Stats", batter_event)

    # ---- Compute rolling stats: PITCHER ----
    st.info("Computing pitcher rolling stats (vectorized, fast)...")
    # Rename pitcher_id as batter_id so function is identical
    df_hist_p = df_hist.rename(columns={"pitcher_id": "batter_id"}).copy()
    pitcher_event = fast_rolling_stats(
        df_hist_p, "batter_id", "game_date", event_windows, main_pitch_types, prefix="p_"
    )
    show_diag("Pitcher Rolling Stats", pitcher_event)

    # ---- Build column names ----
    for stat in ["avg_exit_velo", "hard_hit_rate", "barrel_rate", "fb_rate", "sweet_spot_rate"]:
        for w in event_windows:
            output_columns += [
                f"{stat}_{w}", f"p_{stat}_{w}"
            ]
        for pt in main_pitch_types:
            for w in event_windows:
                output_columns += [
                    f"{stat}_{pt}_{w}", f"p_{stat}_{pt}_{w}"
                ]

    # ---- Merge batter/pitcher stats with today's lineups ----
    merged = df_today.set_index("batter_id").join(batter_event, how="left", rsuffix="_batter")
    if 'pitcher_id' in df_today.columns:
        merged = merged.join(
            pitcher_event, on='pitcher_id', how="left", rsuffix="_pitcher"
        )
    else:
        st.warning("No pitcher_id in today's file, skipping pitcher stat merge.")

    # ---- Add missing columns as NaN ----
    missing_cols = [col for col in output_columns if col not in merged.columns]
    if missing_cols:
        nan_df = pd.DataFrame(np.nan, index=merged.index, columns=missing_cols)
        merged = pd.concat([merged, nan_df], axis=1)
    merged = merged.loc[:,~merged.columns.duplicated()]
    merged = merged.reindex(columns=output_columns)

    show_diag("Final Output Data", merged)

    st.success(f"🟢 Generated file: {merged.shape[0]} unique batters, {merged.shape[1]} columns (features).")
    st.dataframe(merged.head(10))

    st.download_button(
        "⬇️ Download Today's Event-Level CSV (Exact Format)",
        data=merged.reset_index(drop=True).to_csv(index=False),
        file_name="event_level_today_full.csv"
    )
else:
    st.info("Please upload BOTH today's matchups/lineups and historical event-level CSV.")
