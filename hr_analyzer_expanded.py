import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config("🟦 Generate Today's Event-Level CSV", layout="wide")
st.header("🟦 Generate Today's Event-Level CSV (w/ all historical features)")

# --- File Uploaders
today_csv = st.file_uploader(
    "Upload Today's Matchups/Lineups CSV (must have batter_id or mlb_id, player_name, etc)",
    type=["csv"], key="today_csv"
)
hist_csv = st.file_uploader(
    "Upload Historical Event-Level CSV (must have batter_id, game_date, and all features you want to carry forward)",
    type=["csv"], key="hist_csv"
)

if today_csv and hist_csv:
    # === Load & clean data
    df_today = pd.read_csv(today_csv)
    df_hist = pd.read_csv(hist_csv)
    st.success(f"Loaded today's matchups: {df_today.shape[0]} rows, {df_today.shape[1]} columns.")
    st.success(f"Loaded historical event-level: {df_hist.shape[0]} rows, {df_hist.shape[1]} columns.")

    # --- Standardize column names
    df_today.columns = [c.strip().lower().replace(" ", "_") for c in df_today.columns]
    df_hist.columns = [c.strip().lower().replace(" ", "_") for c in df_hist.columns]

    # --- Robust batter_id handling (fixes decimals, missing cols)
    # Try to find the batter_id field in today's and hist files
    id_col_today = None
    for c in ['batter_id', 'mlb_id']:
        if c in df_today.columns:
            id_col_today = c
            break
    if not id_col_today:
        st.error("Today's lineup file must have 'batter_id' or 'mlb_id'.")
        st.stop()

    if 'batter_id' not in df_hist.columns:
        st.error("Historical file must have 'batter_id'.")
        st.stop()

    # Convert IDs to string for join, stripping decimals (e.g. 681297.0 → '681297')
    def norm_id(x):
        try:
            return str(int(float(x)))
        except:
            return str(x).split(".")[0]

    df_today[id_col_today] = df_today[id_col_today].apply(norm_id)
    df_hist['batter_id'] = df_hist['batter_id'].apply(norm_id)

    # --- Take latest row per batter from history (by game_date)
    if 'game_date' in df_hist.columns:
        df_hist['game_date'] = pd.to_datetime(df_hist['game_date'], errors='coerce')
        last_feats = df_hist.sort_values('game_date').groupby('batter_id').tail(1)
    else:
        # fallback: just use last row per batter
        last_feats = df_hist.groupby('batter_id').tail(1)

    # Remove duplicate batter_ids just in case (keep latest)
    last_feats = last_feats.drop_duplicates(subset=['batter_id'], keep='last')

    # Merge ALL columns from history
    hist_cols = [c for c in last_feats.columns if c != 'batter_id']  # batter_id will be used for join
    merged = df_today.merge(last_feats[['batter_id'] + hist_cols], left_on=id_col_today, right_on='batter_id', how='left', suffixes=('', '_hist'))

    # Drop extra batter_id col from right merge if duplicated
    if id_col_today != 'batter_id':
        merged = merged.drop(columns=['batter_id'])

    # --- De-duplicate: Keep ONLY 1 row per batter (keep most complete / latest row)
    dedup_method = st.radio(
        "If a batter appears multiple times in today's file, how should we deduplicate?",
        ("Keep first row", "Keep latest game_date from history (if available)", "Keep row with most non-null features"), index=1
    )
    # Make sure we operate on ID as string
    merged['dedup_id'] = merged[id_col_today].apply(str)
    if dedup_method == "Keep first row":
        merged = merged.drop_duplicates(subset=['dedup_id'], keep='first')
    elif dedup_method == "Keep latest game_date from history (if available)":
        if 'game_date' in merged.columns:
            merged = merged.sort_values('game_date').drop_duplicates(subset=['dedup_id'], keep='last')
        else:
            merged = merged.drop_duplicates(subset=['dedup_id'], keep='last')
    else:  # most non-null
        merged['notnull_count'] = merged.notnull().sum(axis=1)
        merged = merged.sort_values('notnull_count', ascending=False).drop_duplicates(subset=['dedup_id'], keep='first')
        merged = merged.drop(columns=['notnull_count'])

    merged = merged.drop(columns=['dedup_id'])

    # --- Final diagnostics
    st.success(f"Created output with {merged.shape[0]} unique batters. No duplicates remain.")
    st.dataframe(merged.head(25))

    # Nulls diagnostic
    null_report = merged.isnull().sum().sort_values(ascending=False)
    st.markdown("#### Null feature count per column:")
    st.text(null_report[null_report > 0].to_string())

    # --- Download
    st.download_button(
        "⬇️ Download Today's Event-Level CSV (fully merged, de-duplicated)",
        data=merged.to_csv(index=False),
        file_name="event_level_today_full_dedup.csv"
    )
else:
    st.info("Please upload BOTH today's lineups/matchups AND historical event-level CSVs to start.")
