import streamlit as st
import pandas as pd
import numpy as np

st.title("🟦 Generate Today's Event-Level CSV (All Rolling Stats, 1 Row Per Batter)")

st.markdown("""
**Instructions:**
- Upload *Today's Lineups/Matchups* CSV (must have `batter_id` or `mlb_id`, `player_name`, etc).
- Upload *Historical Event-Level* CSV (must have `batter_id`, `game_date`, and all stat columns).
- Output: **ONE row per batter** with all rolling/stat features, calculated from their event history.  
- All numeric columns are auto-corrected for string/decimal mix issues.
""")

today_file = st.file_uploader("Upload Today's Matchups/Lineups CSV", type=["csv"], key="today_csv")
hist_file = st.file_uploader("Upload Historical Event-Level CSV", type=["csv"], key="hist_csv")

ROLLING_WINDOWS = [3, 5, 7, 14]
ROLL_COLS = [
    'launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value',
    'release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z'
]

def to_numeric_df(df):
    """Attempt to fix decimal-string issue by converting columns with numeric content."""
    for col in df.columns:
        # Try only on object columns
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except Exception:
                pass
    return df

def rolling_stats(hist_df, id_col, date_col):
    df = hist_df.copy()
    results = []
    for batter_id, group in df.groupby(id_col):
        group = group.sort_values(date_col)
        for w in ROLLING_WINDOWS:
            for col in ROLL_COLS:
                cname = f'b_{col}_{w}'
                if col in group.columns:
                    group[cname] = pd.to_numeric(group[col], errors='coerce').shift(1).rolling(w, min_periods=1).mean()
        results.append(group)
    return pd.concat(results, ignore_index=True)

if today_file and hist_file:
    df_today = pd.read_csv(today_file)
    df_hist = pd.read_csv(hist_file)

    # Normalize columns for join
    df_today.columns = [c.strip().lower().replace(" ", "_") for c in df_today.columns]
    df_hist.columns = [c.strip().lower().replace(" ", "_") for c in df_hist.columns]
    if "mlb_id" in df_today.columns:
        df_today["batter_id"] = df_today["mlb_id"].astype(str)
    elif "batter_id" in df_today.columns:
        df_today["batter_id"] = df_today["batter_id"].astype(str)
    else:
        st.error("Today's file must have mlb_id or batter_id column.")
        st.stop()
    df_hist["batter_id"] = df_hist["batter_id"].astype(str)
    if "game_date" in df_hist.columns:
        df_hist["game_date"] = pd.to_datetime(df_hist["game_date"])
    else:
        st.error("Historical file must have game_date column.")
        st.stop()

    # --- Fix decimal/string issues for all columns in both files ---
    df_today = to_numeric_df(df_today)
    df_hist = to_numeric_df(df_hist)

    # --- Compute rolling stats ---
    progress = st.empty()
    progress.info("Calculating rolling stats...")
    df_hist_rolled = rolling_stats(df_hist, "batter_id", "game_date")
    progress.success("Rolling stats calculated.")

    # --- Get latest event row per batter, with rolling stats ---
    idx = df_hist_rolled.groupby("batter_id")["game_date"].idxmax()
    latest_feats = df_hist_rolled.loc[idx].copy()
    latest_feats = latest_feats.drop_duplicates(subset=["batter_id"])

    # --- Merge to today's file, only one row per batter ---
    merged = df_today.merge(latest_feats, on="batter_id", how="left", suffixes=('', '_hist'))
    merged = merged.drop_duplicates(subset=["batter_id"], keep="first")

    st.success(f"🟢 Generated file: {len(merged)} unique batters, {merged.shape[1]} columns.")

    st.markdown("**Preview of merged result (first 10 rows):**")
    st.dataframe(merged.head(10))

    # Show null report for rolling/stat features only
    roll_stat_cols = [c for c in merged.columns if c.startswith("b_")]
    null_report = merged[roll_stat_cols].isnull().sum().sort_values(ascending=False)
    st.markdown("**Null Report (rolling/stat features only):**")
    st.text(null_report.to_string())

    st.download_button(
        "⬇️ Download Today's Event-Level CSV (with all rolling features)",
        data=merged.to_csv(index=False),
        file_name="event_level_today_full.csv"
    )
