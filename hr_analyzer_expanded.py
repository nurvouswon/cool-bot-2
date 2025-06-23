import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config("🟦 Generate Today's Event-Level CSV (Batters & Pitchers, All Rolling/Custom Stats)", layout="wide")
st.title("🟦 Generate Today's Event-Level CSV (All Rolling & Custom Stats, 1 Row Per Player)")

st.markdown("""
**Instructions:**
- Upload Today's Lineups/Probables CSV (`batter_id` or `pitcher_id`, plus core columns).
- Upload Historical Event-Level CSV (must have IDs, `game_date`, and stat columns).
- Output: 1 row per player with all rolling, advanced, and custom features.
""")

today_file = st.file_uploader("Upload Today's Lineups/Probables CSV", type=["csv"], key="today_csv")
hist_file = st.file_uploader("Upload Historical Event-Level CSV", type=["csv"], key="hist_csv")

ROLLING_WINDOWS = [3, 5, 7, 14, 20]

BATTER_BASE_STATS = ["hard_hit", "sweet_spot", "barrel", "fb", "avg_exit_velo", "home_run", "fly_ball", "strikeout", "walk"]
PITCHER_BASE_STATS = ["hard_hit_allowed", "barrel_allowed", "fb_allowed", "avg_exit_velo_allowed",
                      "hr_allowed", "innings_pitched", "strikeout_allowed", "walk_allowed"]

def is_batter_mode(df_today):
    batter_cols = {"batter_id", "mlb_id"}
    pitcher_cols = {"pitcher_id", "mlb_id"}
    batter_found = bool(batter_cols & set(df_today.columns))
    pitcher_found = bool(pitcher_cols & set(df_today.columns))
    return batter_found and not pitcher_found

def find_id_column(df_today, mode="batter"):
    if mode == "batter":
        for c in ["batter_id", "mlb_id"]:
            if c in df_today.columns:
                return c
    else:
        for c in ["pitcher_id", "mlb_id"]:
            if c in df_today.columns:
                return c
    return None

def rolling_mean(df_hist, id_col, event_col, window):
    df_hist = df_hist.sort_values(['game_date'])
    return (
        df_hist.groupby(id_col, group_keys=False)
        .apply(lambda grp: grp[event_col].rolling(window, min_periods=1).mean().shift(1))
        .reset_index(level=0, drop=True)
    )

def rolling_sum(df_hist, id_col, event_col, window):
    df_hist = df_hist.sort_values(['game_date'])
    return (
        df_hist.groupby(id_col, group_keys=False)
        .apply(lambda grp: grp[event_col].rolling(window, min_periods=1).sum().shift(1))
        .reset_index(level=0, drop=True)
    )

def rolling_ratio(df_hist, id_col, num_col, den_col, window):
    df_hist = df_hist.sort_values(['game_date'])
    return (
        df_hist.groupby(id_col, group_keys=False)
        .apply(lambda grp:
            (grp[num_col].rolling(window, min_periods=1).sum().shift(1) /
             grp[den_col].rolling(window, min_periods=1).sum().shift(1)).replace([np.inf, -np.inf], np.nan)
        ).reset_index(level=0, drop=True)
    )

def rolling_hr_per_fb(df_hist, id_col, window, hr_col="home_run", fb_col="fly_ball"):
    # HR/FB = HR / fly balls
    return rolling_ratio(df_hist, id_col, hr_col, fb_col, window)

def rolling_k_pct(df_hist, id_col, window, k_col="strikeout", pa_col="plate_appearances"):
    return rolling_ratio(df_hist, id_col, k_col, pa_col, window)

def rolling_bb_pct(df_hist, id_col, window, bb_col="walk", pa_col="plate_appearances"):
    return rolling_ratio(df_hist, id_col, bb_col, pa_col, window)

def rolling_k9(df_hist, id_col, window, k_col="strikeout_allowed", ip_col="innings_pitched"):
    return rolling_ratio(df_hist, id_col, k_col, ip_col, window) * 9

def rolling_bb9(df_hist, id_col, window, bb_col="walk_allowed", ip_col="innings_pitched"):
    return rolling_ratio(df_hist, id_col, bb_col, ip_col, window) * 9

def rolling_hr9(df_hist, id_col, window, hr_col="hr_allowed", ip_col="innings_pitched"):
    return rolling_ratio(df_hist, id_col, hr_col, ip_col, window) * 9

def generate_features(df_hist, id_col, is_batter=True):
    df_hist = df_hist.copy()
    df_hist['game_date'] = pd.to_datetime(df_hist['game_date'], errors='coerce')
    rolling_features = pd.DataFrame(index=df_hist.index)
    if is_batter:
        for stat in BATTER_BASE_STATS:
            for window in ROLLING_WINDOWS:
                if stat in df_hist.columns:
                    rolling_features[f"{stat}_{window}"] = rolling_mean(df_hist, id_col, stat, window)
        # Advanced stats: HR/FB%, K%, BB%
        for window in ROLLING_WINDOWS:
            if all(col in df_hist.columns for col in ["home_run", "fly_ball"]):
                rolling_features[f"hr_per_fb_{window}"] = rolling_hr_per_fb(df_hist, id_col, window)
            if all(col in df_hist.columns for col in ["strikeout", "plate_appearances"]):
                rolling_features[f"k_pct_{window}"] = rolling_k_pct(df_hist, id_col, window)
            if all(col in df_hist.columns for col in ["walk", "plate_appearances"]):
                rolling_features[f"bb_pct_{window}"] = rolling_bb_pct(df_hist, id_col, window)
    else:
        for stat in PITCHER_BASE_STATS:
            for window in ROLLING_WINDOWS:
                if stat in df_hist.columns:
                    rolling_features[f"{stat}_{window}"] = rolling_mean(df_hist, id_col, stat, window)
        # Pitcher custom stats: HR/9, K/9, BB/9
        for window in ROLLING_WINDOWS:
            if all(col in df_hist.columns for col in ["hr_allowed", "innings_pitched"]):
                rolling_features[f"hr_per_9_{window}"] = rolling_hr9(df_hist, id_col, window)
            if all(col in df_hist.columns for col in ["strikeout_allowed", "innings_pitched"]):
                rolling_features[f"k_per_9_{window}"] = rolling_k9(df_hist, id_col, window)
            if all(col in df_hist.columns for col in ["walk_allowed", "innings_pitched"]):
                rolling_features[f"bb_per_9_{window}"] = rolling_bb9(df_hist, id_col, window)
    return pd.concat([df_hist[[id_col, "game_date"]], rolling_features], axis=1)

if today_file and hist_file:
    df_today = pd.read_csv(today_file)
    df_hist = pd.read_csv(hist_file)

    df_today.columns = [str(c).strip().lower().replace(" ", "_") for c in df_today.columns]
    df_hist.columns = [str(c).strip().lower().replace(" ", "_") for c in df_hist.columns]

    batter_mode = is_batter_mode(df_today)
    mode = "batter" if batter_mode else "pitcher"
    id_col = find_id_column(df_today, mode)
    if id_col is None:
        st.error(f"Could not find id column for {mode}s.")
        st.stop()

    df_today[id_col] = df_today[id_col].astype(str).str.strip().str.replace('.0', '', regex=False)
    df_hist[id_col] = df_hist[id_col].astype(str).str.strip().str.replace('.0', '', regex=False)
    df_today = df_today.drop_duplicates(subset=[id_col])

    st.info("Generating rolling/stat features (may take a minute on large files)...")
    rolling_feats = generate_features(df_hist, id_col, batter_mode)
    latest_feats = (
        rolling_feats.sort_values("game_date")
        .groupby(id_col, as_index=False)
        .tail(1)
        .drop_duplicates(subset=[id_col], keep='last')
    )
    merged = df_today.merge(latest_feats, on=id_col, how="left", suffixes=('', '_hist'))
    merged = merged.drop_duplicates(subset=[id_col], keep='first')

    st.success(f"🟢 Generated file: {merged.shape[0]} unique players, {merged.shape[1]} columns (features).")
    st.dataframe(merged.head(10))

    st.download_button(
        "⬇️ Download Today's Event-Level CSV (w/ All Rolling & Custom Windows)",
        data=merged.to_csv(index=False),
        file_name=f"event_level_today_full_{mode}.csv"
    )

    if st.button("Audit Feature Nulls and Ranges"):
        audit = pd.DataFrame({
            "Null %": merged.isnull().mean().sort_values(ascending=False),
            "Min": merged.min(numeric_only=True),
            "Max": merged.max(numeric_only=True),
        })
        st.write(audit)
else:
    st.info("Please upload BOTH today's lineups/probables and historical event-level CSV.")
