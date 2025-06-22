import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config("MLB HR Today Event-Level Merger", layout="wide")
st.header("🟦 Generate Today's Event-Level CSV (w/ all historical features)")

st.markdown("""
**Instructions:**
1. Upload your "Today's Matchups/Lineups" CSV (must contain `batter_id` or `mlb_id`, and `player_name`).
2. Upload your *full* historical event-level CSV (must have `batter_id`, `game_date`, and all features you want to carry forward).
3. This tool will match all batters by `batter_id` (or `mlb_id`), bring in the *latest available features* for each, and let you download a ready-to-predict file.
""")

today_file = st.file_uploader("Upload Today's Matchups/Lineups CSV", type="csv", key="today_csv")
hist_file = st.file_uploader("Upload Historical Event-Level CSV", type="csv", key="hist_csv")

if today_file and hist_file:
    df_today = pd.read_csv(today_file)
    df_hist = pd.read_csv(hist_file)

    # --- DEBUG: show sample columns and shapes
    st.markdown("#### 🔎 Column Check & Shapes")
    st.write("Today's columns:", list(df_today.columns))
    st.write("Historical columns:", list(df_hist.columns))
    st.write(f"Today's CSV: {df_today.shape[0]} rows")
    st.write(f"Historical CSV: {df_hist.shape[0]} rows")

    # --- Normalize column names (for easy matching)
    df_today.columns = [c.strip().lower().replace(" ", "_") for c in df_today.columns]
    df_hist.columns = [c.strip().lower().replace(" ", "_") for c in df_hist.columns]

    # --- Find join key
    id_col = None
    for candidate in ['batter_id', 'mlb_id']:
        if candidate in df_today.columns:
            id_col = candidate
            break
    if not id_col or 'batter_id' not in df_hist.columns:
        st.error("ERROR: Both files must have `batter_id` or `mlb_id` for join. (And `batter_id` must exist in historical CSV.)")
        st.stop()

    # --- Convert IDs to string to ensure match
    df_today[id_col] = df_today[id_col].astype(str).str.strip()
    df_hist['batter_id'] = df_hist['batter_id'].astype(str).str.strip()

    # --- Show sample IDs for debugging
    st.markdown("#### 🆔 Sample IDs for Merge Debugging")
    st.write("Today's IDs:", df_today[id_col].unique()[:10])
    st.write("Historical IDs:", df_hist['batter_id'].unique()[:10])

    # --- Get latest features per batter
    if 'game_date' in df_hist.columns:
        df_hist['game_date'] = pd.to_datetime(df_hist['game_date'], errors='coerce')
        # Only most recent row per batter
        last_feats = df_hist.sort_values('game_date').drop_duplicates('batter_id', keep='last')
    else:
        last_feats = df_hist.drop_duplicates('batter_id', keep='last')

    # --- Exclude duplicate columns from merge
    today_cols = set(df_today.columns)
    hist_cols = [c for c in df_hist.columns if c not in today_cols or c == 'batter_id']

    # --- Merge
    merged = df_today.merge(last_feats[['batter_id'] + [c for c in hist_cols if c != 'batter_id']], 
                            left_on=id_col, right_on='batter_id', how='left', suffixes=('', '_hist'))

    st.markdown("#### 🧪 Merge Result Diagnostics")
    st.write("Merged shape:", merged.shape)
    null_counts = merged.isnull().sum().sort_values(ascending=False)
    st.write("Null count for each column (top 30):")
    st.dataframe(null_counts.head(30))

    unmatched = merged[merged.filter(regex="^b_", axis=1).isnull().all(axis=1)]
    st.write(f"Batters in today's CSV *without* any matching historical features: {len(unmatched)}")
    st.dataframe(unmatched[[id_col, 'player_name']] if 'player_name' in unmatched.columns else unmatched.head(5))

    st.write("Sample merged data:")
    st.dataframe(merged.head(20))

    # --- Download button
    st.success(f"Today's event-level file created! {merged.shape[0]} rows, {merged.shape[1]} columns.")
    st.download_button(
        "⬇️ Download Today's Event-Level CSV (with all historical features)",
        data=merged.to_csv(index=False),
        file_name="event_level_today_full.csv"
    )

else:
    st.info("Please upload BOTH 'today' and 'historical' files to continue.")
