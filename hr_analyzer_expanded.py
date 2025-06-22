import streamlit as st
import pandas as pd
import numpy as np

st.title("🟦 Generate Today's Event-Level CSV (with All Historical Features, One Row Per Batter)")

st.markdown("""
**Instructions:**
1. Upload your *Today’s Matchups/Lineups* CSV (must have `batter_id` or `mlb_id`, `player_name`, etc).
2. Upload your *Historical Event-Level* CSV (must have `batter_id`, `game_date`, and all features you want to carry forward).
3. App will match each batter to their latest stats. Output is ONE row per batter (no duplicates).
""")

# --- Uploaders
today_file = st.file_uploader("Upload Today's Matchups/Lineups CSV", type=["csv"], key="today_csv")
hist_file = st.file_uploader("Upload Historical Event-Level CSV", type=["csv"], key="hist_csv")

if today_file and hist_file:
    # Read CSVs
    df_today = pd.read_csv(today_file)
    df_hist = pd.read_csv(hist_file)

    st.success(f"Today's Matchups file loaded: {df_today.shape[0]} rows, {df_today.shape[1]} columns.")
    st.success(f"Historical Event-Level file loaded: {df_hist.shape[0]} rows, {df_hist.shape[1]} columns.")

    # Standardize column names
    df_today.columns = [c.strip().lower().replace(" ", "_") for c in df_today.columns]
    df_hist.columns = [c.strip().lower().replace(" ", "_") for c in df_hist.columns]

    # --- Normalize batter IDs (fix decimals, type mismatches)
    def norm_id(x):
        try:
            return str(int(float(x)))
        except:
            if pd.isnull(x):
                return ""
            s = str(x)
            if "." in s:
                return s.split(".")[0]
            return s

    # Use mlb_id or batter_id
    if 'mlb_id' in df_today.columns:
        df_today['batter_id'] = df_today['mlb_id'].apply(norm_id)
    elif 'batter_id' in df_today.columns:
        df_today['batter_id'] = df_today['batter_id'].apply(norm_id)
    else:
        st.error("Today's CSV must have a 'batter_id' or 'mlb_id' column!")
        st.stop()
    df_hist['batter_id'] = df_hist['batter_id'].apply(norm_id)

    # --- Make sure game_date is datetime
    if 'game_date' not in df_hist.columns:
        st.error("Historical file must have a 'game_date' column.")
        st.stop()
    df_hist['game_date'] = pd.to_datetime(df_hist['game_date'], errors='coerce')

    # --- For each batter, get ONLY their latest event (row) by game_date
    hist_latest = df_hist.sort_values('game_date').groupby('batter_id', as_index=False).tail(1)

    # --- Merge: today's lineup info + latest hist features
    merged = df_today.merge(hist_latest, on='batter_id', how='left', suffixes=('', '_hist'))

    # --- Deduplicate just in case
    merged = merged.drop_duplicates(subset=['batter_id'], keep='first')

    st.info(f"🟢 Generated file: {merged.shape[0]} unique batters, {merged.shape[1]} columns (features).")
    st.write("Preview of merged result (first 10 rows):")
    st.dataframe(merged.head(10))

    # --- Download button
    st.download_button(
        "⬇️ Download Today's Event-Level CSV (1 row per batter, all features)",
        data=merged.to_csv(index=False),
        file_name="event_level_today_full_merged.csv"
    )

    # --- Diagnostics
    st.markdown("### Null Report (rolling/stat features only):")
    stat_cols = [c for c in hist_latest.columns if c not in df_today.columns and c != 'batter_id']
    null_report = merged[stat_cols].isnull().sum().sort_values(ascending=False)
    st.text(null_report.to_string())

else:
    st.info("Please upload both today's matchups/lineups and historical event-level CSVs to generate the file.")
