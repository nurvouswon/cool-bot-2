import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config("🟦 Generate Today's Event-Level CSV", layout="wide")
st.header("🟦 Generate Today's Event-Level CSV (w/ all historical features)")

st.markdown("""
**Instructions:**  
- Upload today's Matchups/Lineups CSV (must have `batter_id` or `mlb_id`, `player_name`, etc).
- Upload a historical Event-Level CSV (must have `batter_id`, `game_date`, and all features you want to carry forward).
- This app will merge the latest (most recent game_date) historical features onto each batter in today's file.
""")

today_csv = st.file_uploader(
    "Upload Today's Matchups/Lineups CSV (must have batter_id or mlb_id, player_name, etc)", type="csv", key="lineups"
)
hist_csv = st.file_uploader(
    "Upload Historical Event-Level CSV (must have batter_id, game_date, and all features you want to carry forward)", type="csv", key="hist"
)

def clean_id_col(col):
    """Converts to string, strips whitespace, removes .0 and leading zeros."""
    return (
        col.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"^0+(\d+)$", r"\1", regex=True)
    )

if today_csv and hist_csv:
    df_today = pd.read_csv(today_csv)
    df_hist = pd.read_csv(hist_csv)

    # --- Normalize column names ---
    df_today.columns = [c.strip().lower().replace(" ", "_") for c in df_today.columns]
    df_hist.columns = [c.strip().lower().replace(" ", "_") for c in df_hist.columns]

    # --- Find and clean join columns ---
    # Try batter_id, fallback to mlb_id for today, always use batter_id for history
    id_col_today = None
    for c in ['batter_id', 'mlb_id']:
        if c in df_today.columns:
            id_col_today = c
            break

    if not id_col_today or 'batter_id' not in df_hist.columns:
        st.error("Missing 'batter_id' or 'mlb_id' in today's file, or 'batter_id' in history file.")
        st.stop()

    # Clean and cast to string
    df_today[id_col_today] = clean_id_col(df_today[id_col_today])
    df_hist['batter_id'] = clean_id_col(df_hist['batter_id'])

    # --- Get latest historical row per batter ---
    if 'game_date' not in df_hist.columns:
        st.error("Historical file missing 'game_date' column.")
        st.stop()
    df_hist['game_date'] = pd.to_datetime(df_hist['game_date'], errors='coerce')
    last_feats = df_hist.sort_values('game_date').groupby('batter_id').tail(1)

    # --- Columns to merge: everything except duplicates of ID (handled by merge), no unnamed ---
    hist_cols = [c for c in df_hist.columns if not c.startswith("unnamed")]

    # --- Merge and debug diagnostics ---
    # Remove duplicate columns from last_feats for merge (besides batter_id)
    last_feats = last_feats.loc[:, ~last_feats.columns.duplicated()]
    try:
        merged = df_today.merge(last_feats[["batter_id"] + [c for c in hist_cols if c != "batter_id"]],
                                left_on=id_col_today, right_on="batter_id", how="left", suffixes=('', '_hist'))
    except Exception as e:
        st.error(f"Merge failed: {e}")
        st.write("Check if your IDs are unique and all columns are clean.")
        st.stop()

    # --- Diagnostics ---
    st.write(f"### Merged output: {len(merged)} batters")
    st.write("Sample merged data (first 25 rows):")
    st.dataframe(merged.head(25))

    # Any missing merges?
    nulls = merged["batter_id"].isnull().sum()
    if nulls > 0:
        st.warning(f"{nulls} rows did not merge to history (no matching batter_id/mlb_id).")
        st.write("Check below for unmatched IDs in today's file:")
        unmatched = df_today.loc[~df_today[id_col_today].isin(last_feats['batter_id'])]
        st.dataframe(unmatched[[id_col_today, *(col for col in df_today.columns if col != id_col_today)][:5]])

    # Null diagnostics for historical columns (excluding today's columns)
    hist_only = [c for c in hist_cols if c not in df_today.columns and c != "batter_id"]
    st.write("#### Nulls in historical columns after merge (top 20):")
    st.write(merged[hist_only].isnull().sum().sort_values(ascending=False).head(20))

    # Output: CSV with all features
    merged_cols = [c for c in merged.columns if not c.startswith("unnamed")]
    st.success(f"Created today's event-level file: {len(merged)} batters. All columns from history are included.")
    st.download_button(
        "⬇️ Download Today's Event-Level CSV (with all features)",
        data=merged[merged_cols].to_csv(index=False),
        file_name=f"event_level_today_full_{pd.Timestamp.now().strftime('%Y_%m_%d')}.csv"
    )

else:
    st.info("Upload both files to generate your today event-level file.")
