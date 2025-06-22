import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config("🟦 Generate Today's Event-Level CSV", layout="wide")
st.title("🟦 Generate Today's Event-Level CSV (All Rolling Stats, 1 Row Per Batter)")

st.markdown("""
**Instructions:**
- Upload Today's Lineups/Matchups CSV (must have `batter_id` or `mlb_id`, `player_name`, etc).
- Upload Historical Event-Level CSV (must have `batter_id`, `game_date`, and all stat columns).
- Output: ONE row per batter with all rolling/stat features, calculated from their event history.
- All numeric columns are auto-corrected for string/decimal mix issues.
""")

today_file = st.file_uploader("Upload Today's Matchups/Lineups CSV", type=["csv"], key="today_csv")
hist_file = st.file_uploader("Upload Historical Event-Level CSV", type=["csv"], key="hist_csv")

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

    # ---- Get the *latest* event row for each batter from history ----
    if 'game_date' in df_hist.columns:
        # Coerce date
        df_hist['game_date'] = pd.to_datetime(df_hist['game_date'], errors='coerce')
        latest_feats = (
            df_hist.sort_values('game_date')
            .groupby('batter_id', as_index=False)
            .tail(1)
        )
    else:
        latest_feats = df_hist.drop_duplicates(subset=['batter_id'], keep='last')

    # ---- Remove duplicated batter_ids in latest_feats (keep last, safest) ----
    latest_feats = latest_feats.drop_duplicates(subset=['batter_id'], keep='last')

    # ---- Handle any column dtype (auto-correct numerics for all rolling/stat features) ----
    rolling_stat_cols = [
        c for c in latest_feats.columns
        if c.startswith('b_') or c.startswith('p_') or c.endswith('_rate_20') or 'rolling' in c or
           c.startswith('park_hand_hr_') or c.startswith('b_vsp_hand_hr_') or c.startswith('p_vsb_hand_hr_') or
           c.startswith('b_pitchtype_hr_') or c.startswith('p_pitchtype_hr_')
    ]
    # Try to convert rolling stat columns to numeric if possible
    for c in rolling_stat_cols:
        if c in latest_feats.columns:
            latest_feats[c] = pd.to_numeric(latest_feats[c], errors='coerce')

    # ---- Merge (left join) today's lineups with latest event stats ----
    merged = df_today.merge(latest_feats, on="batter_id", how="left", suffixes=('', '_hist'))

    # ---- Deduplicate after merge, just in case ----
    merged = merged.drop_duplicates(subset=['batter_id'], keep='first')

    st.success(f"🟢 Generated file: {merged.shape[0]} unique batters, {merged.shape[1]} columns (features).")

    # ---- Preview output ----
    st.dataframe(merged.head(10))

    # ---- Null report on rolling/stat features ----
    st.markdown("#### Null Report (rolling/stat features only):")
    null_report = merged[rolling_stat_cols].isnull().sum().sort_values(ascending=False)
    st.code(null_report.to_string())

    # ---- Download button ----
    st.download_button(
        "⬇️ Download Today's Event-Level CSV",
        data=merged.to_csv(index=False),
        file_name="event_level_today_full.csv"
    )
else:
    st.info("Please upload BOTH today's matchups/lineups and historical event-level CSV.")
