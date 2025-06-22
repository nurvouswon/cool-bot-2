import streamlit as st
import pandas as pd

st.set_page_config("MLB HR Feature Merger", layout="centered")

st.title("🟦 Generate Today's Event-Level CSV (w/ all historical features)")

# 1. Uploaders
today_file = st.file_uploader(
    "Upload Today's Matchups/Lineups CSV (must have batter_id or mlb_id, player_name, etc)",
    type=["csv"], key="today"
)
hist_file = st.file_uploader(
    "Upload Historical Event-Level CSV (must have batter_id, game_date, and all features you want to carry forward)",
    type=["csv"], key="hist"
)

if today_file and hist_file:
    df_today = pd.read_csv(today_file)
    df_hist = pd.read_csv(hist_file)

    # Normalize column names for easy matching
    df_today.columns = [c.strip().lower().replace(" ", "_") for c in df_today.columns]
    df_hist.columns = [c.strip().lower().replace(" ", "_") for c in df_hist.columns]

    # Find ID columns
    id_col = None
    for col in ["batter_id", "mlb_id"]:
        if col in df_today.columns:
            id_col = col
            break
    if not id_col or "batter_id" not in df_hist.columns:
        st.error("Both files must have a batter_id or mlb_id column for merge (and batter_id in historical data).")
        st.stop()

    # Ensure IDs are string for merge safety
    df_today[id_col] = df_today[id_col].astype(str)
    df_hist["batter_id"] = df_hist["batter_id"].astype(str)

    # Use all columns from event-level CSV
    hist_cols = [c for c in df_hist.columns if not c.startswith("unnamed")]

    # Get latest row per batter_id (by game_date if available)
    if "game_date" in df_hist.columns:
        df_hist["game_date"] = pd.to_datetime(df_hist["game_date"], errors="coerce")
        last_feats = (
            df_hist.sort_values("game_date")
                   .drop_duplicates("batter_id", keep="last")
        )
    else:
        last_feats = df_hist.drop_duplicates("batter_id", keep="last")

    # Safety check
    assert last_feats["batter_id"].is_unique, "Internal error: duplicate batter_ids in last_feats!"

    # Merge! Use all available columns
    merged = df_today.merge(last_feats[["batter_id"] + [c for c in hist_cols if c != "batter_id"]],
                            left_on=id_col, right_on="batter_id", how="left")

    # Diagnostics
    st.success(f"Created merged file: {len(merged)} rows, {merged.shape[1]} columns. All historical features carried forward.")
    st.dataframe(merged.head(20))

    # Null report for historical columns
    rolling_stat_cols = [c for c in hist_cols if c not in df_today.columns and c != "batter_id"]
    st.markdown("#### Null count for historical features in merged output:")
    st.write(merged[rolling_stat_cols].isnull().sum().sort_values(ascending=False))

    # Download
    st.download_button(
        "⬇️ Download Today's Event-Level CSV (merged)",
        data=merged.to_csv(index=False),
        file_name="event_level_today_full.csv"
    )
else:
    st.info("Please upload both Today's Matchups/Lineups CSV and Historical Event-Level CSV.")
