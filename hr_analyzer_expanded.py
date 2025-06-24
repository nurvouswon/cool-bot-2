import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config("🟦 Generate Today's Event-Level CSV", layout="wide")
st.title("🟦 Generate Today's Event-Level CSV (FAST, One Row Per Batter, All Features)")

st.markdown("""
**Instructions:**
- Upload Today's Lineups/Matchups CSV (must have `batter_id` or `mlb_id`, `player_name`, etc).
- Upload Historical Event-Level CSV (must have ALL rolling/stat columns for both batter & pitcher, per event).
- Output: ONE row per batter with all rolling/stat features (no in-app slow rolling).
""")

today_file = st.file_uploader("Upload Today's Matchups/Lineups CSV", type=["csv"], key="today_csv")
hist_file = st.file_uploader("Upload Historical Event-Level CSV", type=["csv"], key="hist_csv")

output_columns = [...]  # <-- paste your desired columns here!

# Batter and pitcher stat columns you want to be always non-null if possible
batter_rolling_stat_cols = ["hard_hit_rate_20", "sweet_spot_rate_20"]  # etc, adjust as needed
pitcher_rolling_stat_cols = ["p_hard_hit_rate_20", "p_sweet_spot_rate_20"]  # etc, adjust as needed

if today_file and hist_file:
    df_today = pd.read_csv(today_file)
    df_hist = pd.read_csv(hist_file)

    st.success(f"Today's Matchups file loaded: {df_today.shape[0]} rows, {df_today.shape[1]} columns.")
    st.success(f"Historical Event-Level file loaded: {df_hist.shape[0]} rows, {df_hist.shape[1]} columns.")

    df_today.columns = [str(c).strip().lower().replace(" ", "_") for c in df_today.columns]
    df_hist.columns = [str(c).strip().lower().replace(" ", "_") for c in df_hist.columns]

    # ---- Fix IDs ----
    for col in ['batter_id', 'pitcher_id']:
        if col in df_today.columns:
            df_today[col] = df_today[col].astype(str).str.strip().str.replace('.0', '', regex=False)
        if col in df_hist.columns:
            df_hist[col] = df_hist[col].astype(str).str.strip().str.replace('.0', '', regex=False)

    # ---- Deduplicate today's batters ----
    df_today = df_today.drop_duplicates(subset=["batter_id"])

    # ---- Get latest event row WITH non-null batter rolling stats ----
    df_hist['game_date'] = pd.to_datetime(df_hist['game_date'], errors='coerce')
    latest_batter = []
    for batter_id, group in df_hist.sort_values('game_date').groupby('batter_id'):
        non_null_row = group.dropna(subset=batter_rolling_stat_cols)
        if not non_null_row.empty:
            latest_batter.append(non_null_row.iloc[-1])
        else:
            latest_batter.append(group.iloc[-1])
    batter_feats = pd.DataFrame(latest_batter).drop_duplicates(subset=['batter_id'], keep='last')

    # ---- Get latest event row WITH non-null pitcher rolling stats ----
    latest_pitcher = []
    for pitcher_id, group in df_hist.sort_values('game_date').groupby('pitcher_id'):
        non_null_row = group.dropna(subset=pitcher_rolling_stat_cols)
        if not non_null_row.empty:
            latest_pitcher.append(non_null_row.iloc[-1])
        else:
            latest_pitcher.append(group.iloc[-1])
    pitcher_feats = pd.DataFrame(latest_pitcher).drop_duplicates(subset=['pitcher_id'], keep='last')

    # ---- Merge today's lineups with batter and pitcher event stats ----
    merged = df_today.merge(batter_feats, on="batter_id", how="left", suffixes=('', '_batter'))
    if 'pitcher_id' in df_today.columns and 'pitcher_id' in pitcher_feats.columns:
        merged = merged.merge(pitcher_feats, on="pitcher_id", how="left", suffixes=('', '_pitcher'))

    # ---- Force all relevant stats to numeric (auto-fix float/string mix) ----
    for c in merged.columns:
        if c not in ["batter_id", "pitcher_id", "player_name", "player_name_hist"]:
            merged[c] = pd.to_numeric(merged[c], errors='ignore')

    # ---- Reindex to exact output column order ----
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
