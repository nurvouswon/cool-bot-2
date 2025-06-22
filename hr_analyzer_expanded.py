import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config("Today's Event-Level CSV Generator", layout="wide")
st.header("🟦 Generate Today's Event-Level CSV (w/ all historical features)")

uploaded_today = st.file_uploader(
    "Upload Today's Matchups/Lineups CSV (must have batter_id or mlb_id, player_name, etc)",
    type="csv", key="today_csv"
)
uploaded_hist = st.file_uploader(
    "Upload Historical Event-Level CSV (must have batter_id, game_date, and all features you want to carry forward)",
    type="csv", key="hist_csv"
)

if uploaded_today and uploaded_hist:
    df_today = pd.read_csv(uploaded_today)
    df_hist = pd.read_csv(uploaded_hist)

    # --- Normalize columns ---
    df_today.columns = [c.strip().lower().replace(" ", "_") for c in df_today.columns]
    df_hist.columns = [c.strip().lower().replace(" ", "_") for c in df_hist.columns]

    # Use 'batter_id' everywhere if possible
    if 'mlb_id' in df_today.columns and 'batter_id' not in df_today.columns:
        df_today['batter_id'] = df_today['mlb_id'].astype(str)
    elif 'batter_id' in df_today.columns:
        df_today['batter_id'] = df_today['batter_id'].astype(str)
    if 'batter_id' in df_hist.columns:
        df_hist['batter_id'] = df_hist['batter_id'].astype(str)

    # Ensure date column is datetime
    if 'game_date' in df_hist.columns:
        df_hist['game_date'] = pd.to_datetime(df_hist['game_date'], errors='coerce')

    # --- For each batter in today's file, grab latest (by game_date) historical row ---
    last_feats = (
        df_hist.sort_values('game_date')
        .groupby('batter_id')
        .tail(1)
        .reset_index(drop=True)
    )

    # --- Merge: today's batters + ALL latest historical features for each batter ---
    # If today file has columns overlapping hist, only keep "today" version of things like player name, team, position, etc.
    merge_on = 'batter_id'
    today_cols = set(df_today.columns)
    # Don't duplicate these columns from history; they are in today's matchups already
    skip_cols = ['player_name', 'team_code', 'game_date', 'position', 'batting_order', 'team', 'mlb_id']
    hist_cols = [c for c in last_feats.columns if c not in today_cols or c not in skip_cols]
    merged = df_today.merge(last_feats[["batter_id"] + hist_cols], on="batter_id", how="left")

    st.subheader("Preview of Today's Event-Level CSV (with *all* historical features):")
    st.dataframe(merged.head(20))

    st.success(f"Created merged file for {len(merged)} batters. All columns from historical event-level data included.")

    st.download_button(
        "⬇️ Download Today's Event-Level CSV",
        data=merged.to_csv(index=False),
        file_name=f"event_level_today_full_{datetime.now().strftime('%Y_%m_%d')}.csv"
    )

    st.markdown("#### Null count for features pulled from historical:")
    hist_feature_cols = [c for c in merged.columns if c in df_hist.columns and c != 'batter_id']
    st.dataframe(merged[hist_feature_cols].isnull().sum().sort_values(ascending=False).to_frame("null_count"))
else:
    st.info("Upload both historical event-level and today's matchups/lineups CSVs to begin.")
