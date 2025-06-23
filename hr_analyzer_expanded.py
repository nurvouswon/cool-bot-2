import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config("🟦 Generate Today's Event-Level CSV (Debug)", layout="wide")
st.title("🟦 Generate Today's Event-Level CSV (All Rolling Stats, Pitcher+Batter, Pitch Type, All Windows, Debug)")

st.markdown("""
**Instructions:**
- Upload Today's Lineups/Matchups CSV (must have `batter_id` or `mlb_id`, `player_name`, etc).
- Upload Historical Event-Level CSV (must have `batter_id`, `pitcher_id`, `pitch_type`, `game_date`, and all stat columns).
- Output: ONE row per batter with ALL rolling/stat features (3/7/14/20, batter & pitcher, per pitch type and overall).
- All numeric columns are auto-corrected for string/decimal mix issues.
- ⚠️ Includes debug/diagnostic output at bottom.
""")

today_file = st.file_uploader("Upload Today's Matchups/Lineups CSV", type=["csv"], key="today_csv")
hist_file = st.file_uploader("Upload Historical Event-Level CSV", type=["csv"], key="hist_csv")

output_columns = [
    "team_code","game_date","game_number","mlb_id","player_name","batting_order","position",
    "weather","temp","wind_mph","wind_dir","condition","stadium","city","batter_id","p_throws",
    # Overall Batting
    "hard_hit_rate_3","hard_hit_rate_7","hard_hit_rate_14","hard_hit_rate_20",
    "barrel_rate_3","barrel_rate_7","barrel_rate_14","barrel_rate_20",
    "fb_rate_3","fb_rate_7","fb_rate_14","fb_rate_20",
    "avg_exit_velo_3","avg_exit_velo_7","avg_exit_velo_14","avg_exit_velo_20",
    "sweet_spot_rate_3","sweet_spot_rate_7","sweet_spot_rate_14","sweet_spot_rate_20",
    # Overall Pitching
    "p_hard_hit_rate_3","p_hard_hit_rate_7","p_hard_hit_rate_14","p_hard_hit_rate_20",
    "p_barrel_rate_3","p_barrel_rate_7","p_barrel_rate_14","p_barrel_rate_20",
    "p_fb_rate_3","p_fb_rate_7","p_fb_rate_14","p_fb_rate_20",
    "p_avg_exit_velo_3","p_avg_exit_velo_7","p_avg_exit_velo_14","p_avg_exit_velo_20",
    "p_sweet_spot_rate_3","p_sweet_spot_rate_7","p_sweet_spot_rate_14","p_sweet_spot_rate_20",
]

event_windows = [3, 7, 14, 20]
main_pitch_types = ["ff", "sl", "cu", "ch", "si", "fc", "fs", "st", "sinker", "splitter", "sweeper"]

batter_custom_stats = {
    "hard_hit_rate": lambda df: np.mean((pd.to_numeric(df['launch_speed'], errors='coerce') >= 95)),
    "barrel_rate": lambda df: np.mean((pd.to_numeric(df['launch_speed'], errors='coerce') >= 98) &
                                      (pd.to_numeric(df['launch_angle'], errors='coerce') >= 26) &
                                      (pd.to_numeric(df['launch_angle'], errors='coerce') <= 30)),
    "fb_rate": lambda df: np.mean((pd.to_numeric(df['launch_angle'], errors='coerce') >= 25)),
    "avg_exit_velo": lambda df: np.nanmean(pd.to_numeric(df['launch_speed'], errors='coerce')),
    "sweet_spot_rate": lambda df: np.mean((pd.to_numeric(df['launch_angle'], errors='coerce') >= 8) &
                                          (pd.to_numeric(df['launch_angle'], errors='coerce') <= 32)),
}
pitcher_custom_stats = {f"p_{k}": v for k, v in batter_custom_stats.items()}

def compute_event_rolling(df, id_col, date_col, stat_fns, windows, pitch_types=None, prefix=""):
    out = []
    df = df.sort_values([id_col, date_col])
    for pid, group in df.groupby(id_col):
        group = group.sort_values(date_col)
        for idx in range(len(group)):
            row = group.iloc[idx]
            stats = {id_col: row[id_col], date_col: row[date_col]}
            # Overall
            for stat, func in stat_fns.items():
                for w in windows:
                    wgroup = group.iloc[max(0, idx-w+1):idx+1]
                    try:
                        stats[f"{prefix}{stat}_{w}"] = func(wgroup)
                    except Exception:
                        stats[f"{prefix}{stat}_{w}"] = np.nan
            # Per pitch type
            if pitch_types is not None and "pitch_type" in group:
                for pt in pitch_types:
                    pt_mask = (group["pitch_type"].str.lower() == pt)
                    pt_group = group[pt_mask]
                    for stat, func in stat_fns.items():
                        for w in windows:
                            wgroup = pt_group.iloc[max(0, idx-w+1):idx+1]
                            try:
                                stats[f"{prefix}{stat}_{pt}_{w}"] = func(wgroup)
                            except Exception:
                                stats[f"{prefix}{stat}_{pt}_{w}"] = np.nan
            out.append(stats)
    return pd.DataFrame(out)

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

# ---------------- MAIN APP LOGIC -----------------
if today_file and hist_file:
    df_today = pd.read_csv(today_file)
    df_hist = pd.read_csv(hist_file)

    st.success(f"Today's Matchups file loaded: {df_today.shape[0]} rows, {df_today.shape[1]} columns.")
    st.success(f"Historical Event-Level file loaded: {df_hist.shape[0]} rows, {df_hist.shape[1]} columns.")

    # -------- CLEANING ID COLUMNS FOR BOTH BATTERS AND PITCHERS ---------
    df_today.columns = [str(c).strip().lower().replace(" ", "_") for c in df_today.columns]
    df_hist.columns = [str(c).strip().lower().replace(" ", "_") for c in df_hist.columns]

    # Find batter_id column in today's file
    id_cols_today = ['batter_id', 'mlb_id']
    id_col_today = next((c for c in id_cols_today if c in df_today.columns), None)
    if id_col_today is None:
        st.error("Today's file must have 'batter_id' or 'mlb_id' column.")
        st.stop()
    if 'batter_id' not in df_hist.columns or 'pitcher_id' not in df_hist.columns:
        st.error("Historical file must have both 'batter_id' and 'pitcher_id' columns.")
        st.stop()
    if 'pitcher_id' not in df_today.columns:
        st.warning("Today's file has no 'pitcher_id' column -- pitcher stats will be empty.")

    # Clean batter_id
    df_today['batter_id'] = df_today[id_col_today].astype(str).str.strip().str.replace('.0', '', regex=False)
    df_hist['batter_id'] = df_hist['batter_id'].astype(str).str.strip().str.replace('.0', '', regex=False)
    # Clean pitcher_id
    if 'pitcher_id' in df_today.columns:
        df_today['pitcher_id'] = df_today['pitcher_id'].astype(str).str.strip().str.replace('.0', '', regex=False)
    if 'pitcher_id' in df_hist.columns:
        df_hist['pitcher_id'] = df_hist['pitcher_id'].astype(str).str.strip().str.replace('.0', '', regex=False)

    df_today = df_today.drop_duplicates(subset=["batter_id"])
    df_today = parse_weather_fields(df_today)
    if 'game_date' not in df_hist.columns:
        st.error("Historical file must have a 'game_date' column.")
        st.stop()
    df_hist['game_date'] = pd.to_datetime(df_hist['game_date'], errors='coerce')

    # ---- Batter event rolling, per pitch type and overall ----
    st.info("Computing batter rolling stats...")
    batter_event = compute_event_rolling(
        df_hist, "batter_id", "game_date", batter_custom_stats, event_windows, main_pitch_types, prefix=""
    ).sort_values(['batter_id', 'game_date']).drop_duplicates(['batter_id'], keep='last')
    st.write(f"Batter rolling stats shape: {batter_event.shape}")

    # ---- Pitcher event rolling, per pitch type and overall ----
    st.info("Computing pitcher rolling stats...")
    pitcher_event = compute_event_rolling(
        df_hist.rename(columns={"pitcher_id": "batter_id"}), "batter_id", "game_date",
        pitcher_custom_stats, event_windows, main_pitch_types, prefix="p_"
    ).sort_values(['batter_id', 'game_date']).drop_duplicates(['batter_id'], keep='last')
    pitcher_event = pitcher_event.rename(columns={'batter_id': 'pitcher_id', 'game_date': 'game_date'})
    st.write(f"Pitcher rolling stats shape: {pitcher_event.shape}")

    # Dynamically add per-pitch-type stat names to output_columns (batter, pitcher)
    for stat in ["hard_hit_rate", "barrel_rate", "fb_rate", "avg_exit_velo", "sweet_spot_rate"]:
        for pt in main_pitch_types:
            for w in event_windows:
                output_columns.append(f"{stat}_{pt}_{w}")
                output_columns.append(f"p_{stat}_{pt}_{w}")

    # ---- Merge batter and pitcher stats into today's lineups ----
    merged = df_today.merge(batter_event, on='batter_id', how='left', suffixes=('', '_batter'))
    if 'pitcher_id' in df_today.columns:
        merged = merged.merge(pitcher_event, left_on='pitcher_id', right_on='pitcher_id', how='left', suffixes=('', '_pitcher'))

    # ---- Add any missing columns all at once to avoid fragmentation ----
    missing_cols = [col for col in output_columns if col not in merged.columns]
    if missing_cols:
        nan_df = pd.DataFrame(np.nan, index=merged.index, columns=missing_cols)
        merged = pd.concat([merged, nan_df], axis=1)
    merged = merged.loc[:,~merged.columns.duplicated()]
    merged = merged.reindex(columns=output_columns)

    st.success(f"🟢 Generated file: {merged.shape[0]} unique batters, {merged.shape[1]} columns (features).")
    st.dataframe(merged.head(10))

    st.download_button(
        "⬇️ Download Today's Event-Level CSV (Exact Format)",
        data=merged.to_csv(index=False),
        file_name="event_level_today_full.csv"
    )

    # ---------- DIAGNOSTICS & DEBUG PANEL -----------
    with st.expander("🩺 Debug/Diagnostics: Show ID Merge Info and Problems", expanded=False):
        st.write("**Unique pitcher_id values in today's file:**", df_today['pitcher_id'].unique()[:30])
        st.write("**Unique pitcher_id values in event stats:**", pitcher_event['pitcher_id'].unique()[:30])
        pitcher_id_today = set(df_today['pitcher_id'].dropna().unique()) if 'pitcher_id' in df_today.columns else set()
        pitcher_id_stats = set(pitcher_event['pitcher_id'].dropna().unique())
        unmatched_in_stats = pitcher_id_today - pitcher_id_stats
        st.write(f"⚠️ pitcher_id values in today's file but NOT in event-level stats ({len(unmatched_in_stats)} shown):", list(unmatched_in_stats)[:30])
        if 'pitcher_id' in df_today.columns:
            st.write("**First 5 rows after merge (pitcher stats):**")
            st.dataframe(merged.head(5)[[c for c in merged.columns if c.startswith("p_") or c in ['pitcher_id', 'batter_id']]])

else:
    st.info("Please upload BOTH today's matchups/lineups and historical event-level CSV.")
