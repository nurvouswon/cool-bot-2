import streamlit as st
import pandas as pd
import numpy as np
import re
import time

st.set_page_config("🟦 Generate Today's Event-Level CSV", layout="wide")
st.title("🟦 Generate Today's Event-Level CSV (FAST, All Rolling Stats, Pitcher+Batter, Pitch Type, All Windows)")

st.markdown("""
**Instructions:**
- Upload Today's Lineups/Matchups CSV (must have `batter_id` or `mlb_id`, `player_name`, etc).
- Upload Historical Event-Level CSV (must have `batter_id`, `pitcher_id`, `pitch_type`, `game_date`, and all stat columns).
- Output: ONE row per batter with ALL rolling/stat features (3/7/14/20, batter & pitcher, per pitch type and overall).
- All numeric columns are auto-corrected for string/decimal mix issues.
""")

today_file = st.file_uploader("Upload Today's Matchups/Lineups CSV", type=["csv"], key="today_csv")
hist_file = st.file_uploader("Upload Historical Event-Level CSV", type=["csv"], key="hist_csv")

# --- Config ---
event_windows = [3, 7, 14, 20]
main_pitch_types = ["ff", "sl", "cu", "ch", "si", "fc", "fs", "st", "sinker", "splitter", "sweeper"]

output_columns = [
    "team_code","game_date","game_number","mlb_id","player_name","batting_order","position",
    "weather","temp","wind_mph","wind_dir","condition","stadium","city","batter_id","p_throws"
]
for base in ["avg_exit_velo", "hard_hit_rate", "barrel_rate", "fb_rate", "sweet_spot_rate"]:
    for w in event_windows:
        output_columns.append(f"{base}_{w}")
        output_columns.append(f"p_{base}_{w}")
    for pt in main_pitch_types:
        for w in event_windows:
            output_columns.append(f"{base}_{pt}_{w}")
            output_columns.append(f"p_{base}_{pt}_{w}")

# --- Utility for Weather Parsing ---
def parse_weather_fields(df):
    if "weather" in df.columns:
        weather_str = df["weather"].astype(str)
        df["temp"] = weather_str.str.extract(r'(\d{2,3})\s*O', expand=False)
        df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
        wind_mph = weather_str.str.extract(r'(\d+)\s*-\s*(\d+)', expand=True)
        df["wind_mph"] = pd.to_numeric(wind_mph[0], errors="coerce")
        df["wind_mph"] = np.where(wind_mph[1].notnull(),
                                  0.5*(pd.to_numeric(wind_mph[0], errors='coerce') +
                                       pd.to_numeric(wind_mph[1], errors='coerce')),
                                  df["wind_mph"])
        df["wind_mph"] = df["wind_mph"].fillna(weather_str.str.extract(r'(\d+)\s*mph', expand=False)).astype(float)
        df["wind_dir"] = weather_str.str.extract(r'(?:mph\s+)?([nswecf]{1,2})', flags=re.I, expand=False)
        df["condition"] = weather_str.str.extract(r'(indoor|outdoor)', flags=re.I, expand=False)
    return df

# --- Fast Rolling Stats, One Row Per Entity ---
def fast_rolling_stats(df, id_col, date_col, windows, pitch_types=None, prefix=""):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['launch_speed'] = pd.to_numeric(df['launch_speed'], errors='coerce')
    df['launch_angle'] = pd.to_numeric(df['launch_angle'], errors='coerce')
    if 'pitch_type' in df.columns:
        df['pitch_type'] = df['pitch_type'].astype(str).str.lower().str.strip()
    df = df.sort_values([id_col, date_col])

    feature_frames = []
    grouped = df.groupby(id_col)

    for name, group in grouped:
        out_row = {}
        for w in windows:
            out_row[f"{prefix}avg_exit_velo_{w}"] = group['launch_speed'].rolling(w, min_periods=1).mean().iloc[-1]
            out_row[f"{prefix}hard_hit_rate_{w}"] = (group['launch_speed'].rolling(w, min_periods=1)
                                                     .apply(lambda x: np.mean(x >= 95)).iloc[-1])
            out_row[f"{prefix}barrel_rate_{w}"] = (((group['launch_speed'] >= 98) &
                                                    (group['launch_angle'] >= 26) &
                                                    (group['launch_angle'] <= 30))
                                                   .rolling(w, min_periods=1).mean().iloc[-1])
            out_row[f"{prefix}fb_rate_{w}"] = (group['launch_angle'].rolling(w, min_periods=1)
                                               .apply(lambda x: np.mean(x >= 25)).iloc[-1])
            out_row[f"{prefix}sweet_spot_rate_{w}"] = (group['launch_angle'].rolling(w, min_periods=1)
                                                       .apply(lambda x: np.mean((x >= 8) & (x <= 32))).iloc[-1])
        # Per pitch type
        if pitch_types is not None and "pitch_type" in group.columns:
            for pt in pitch_types:
                pt_group = group[group['pitch_type'] == pt]
                if pt_group.empty:
                    for w in windows:
                        out_row[f"{prefix}avg_exit_velo_{pt}_{w}"] = np.nan
                        out_row[f"{prefix}hard_hit_rate_{pt}_{w}"] = np.nan
                        out_row[f"{prefix}barrel_rate_{pt}_{w}"] = np.nan
                        out_row[f"{prefix}fb_rate_{pt}_{w}"] = np.nan
                        out_row[f"{prefix}sweet_spot_rate_{pt}_{w}"] = np.nan
                else:
                    for w in windows:
                        out_row[f"{prefix}avg_exit_velo_{pt}_{w}"] = pt_group['launch_speed'].rolling(w, min_periods=1).mean().iloc[-1]
                        out_row[f"{prefix}hard_hit_rate_{pt}_{w}"] = (pt_group['launch_speed'].rolling(w, min_periods=1)
                                                                       .apply(lambda x: np.mean(x >= 95)).iloc[-1])
                        out_row[f"{prefix}barrel_rate_{pt}_{w}"] = (((pt_group['launch_speed'] >= 98) &
                                                                     (pt_group['launch_angle'] >= 26) &
                                                                     (pt_group['launch_angle'] <= 30))
                                                                    .rolling(w, min_periods=1).mean().iloc[-1])
                        out_row[f"{prefix}fb_rate_{pt}_{w}"] = (pt_group['launch_angle'].rolling(w, min_periods=1)
                                                                .apply(lambda x: np.mean(x >= 25)).iloc[-1])
                        out_row[f"{prefix}sweet_spot_rate_{pt}_{w}"] = (pt_group['launch_angle'].rolling(w, min_periods=1)
                                                                        .apply(lambda x: np.mean((x >= 8) & (x <= 32))).iloc[-1])
        out_row[id_col] = name
        feature_frames.append(out_row)
    return pd.DataFrame(feature_frames)

# --- Diagnostics: Print & Streamlit ---
def diag(msg, df=None):
    st.info(msg)
    if df is not None:
        st.write(f"Shape: {df.shape}")
        st.dataframe(df.head(3))

def pct(x):
    return f"{x}%"

# --- Progress bar utilities ---
step = [0]
step_total = 8  # Number of steps below!
progress = st.progress(0)
status = st.empty()
def inc(msg=None):
    step[0] += 1
    percent = min(int(100 * step[0] / step_total), 100)  # cap at 100%
    progress.progress(percent)
    status.markdown(f"{pct(percent)} {'- ' + msg if msg else ''}")
    time.sleep(0.05)

# --- MAIN APP LOGIC ---
debug_out = []
if today_file and hist_file:
    inc("Loading data")
    df_today = pd.read_csv(today_file)
    df_hist = pd.read_csv(hist_file)

    debug_out.append("Loaded Today's matchups and historical event data.")
    debug_out.append("Today's Data Sample:\n" + str(df_today.head(2)))
    debug_out.append("Historical Data Sample:\n" + str(df_hist.head(2)))

    inc("Standardizing columns and IDs")
    df_today.columns = [str(c).strip().lower().replace(" ", "_") for c in df_today.columns]
    df_hist.columns = [str(c).strip().lower().replace(" ", "_") for c in df_hist.columns]
    debug_out.append("Standardized columns.")

    # === Pitcher Assignment Fix ===
    if "pitcher_id" not in df_today.columns:
        df_today["pitcher_id"] = np.nan
        # For each team/game/game_number, assign the SP's mlb_id as pitcher_id for all
        for key, group in df_today.groupby(["team_code", "game_date", "game_number"]):
            sp_row = group[group["position"].str.upper() == "SP"]
            if len(sp_row) == 1:
                pitcher_id_val = sp_row.iloc[0]["mlb_id"]
                mask = (
                    (df_today["team_code"] == key[0]) &
                    (df_today["game_date"] == key[1]) &
                    (df_today["game_number"] == key[2])
                )
                df_today.loc[mask, "pitcher_id"] = pitcher_id_val
            else:
                continue
    # Clean pitcher_id (decimal to string, like batter_id)
    df_today["pitcher_id"] = df_today["pitcher_id"].astype(str).str.strip().str.replace('.0', '', regex=False)

    debug_out.append("Pitcher_id column filled. Sample:\n" + str(df_today[["team_code","game_date","game_number","player_name","position","mlb_id","pitcher_id"]].head(8)))

    # --- ID Fixes for both files ---
    id_cols_today = ['batter_id', 'mlb_id']
    id_col_today = next((c for c in id_cols_today if c in df_today.columns), None)
    if id_col_today is None:
        st.error("Today's file must have 'batter_id' or 'mlb_id' column.")
        st.stop()
    for col in ['batter_id', 'pitcher_id']:
        if col in df_today.columns:
            df_today[col] = df_today[col].astype(str).str.strip().str.replace('.0', '', regex=False)
        if col in df_hist.columns:
            df_hist[col] = df_hist[col].astype(str).str.strip().str.replace('.0', '', regex=False)

    df_today['batter_id'] = df_today[id_col_today].astype(str).str.strip().str.replace('.0', '', regex=False)
    df_today = df_today.drop_duplicates(subset=["batter_id"])
    df_today = parse_weather_fields(df_today)

    debug_out.append("Parsed weather data for today's lineup.")

    inc("Parsing and validating dates")
    if 'game_date' not in df_hist.columns:
        st.error("Historical file must have a 'game_date' column.")
        st.stop()
    df_hist['game_date'] = pd.to_datetime(df_hist['game_date'], errors='coerce')

    inc("Computing batter rolling stats (vectorized, fast)...")
    batter_event = fast_rolling_stats(
        df_hist, "batter_id", "game_date", event_windows, main_pitch_types, prefix=""
    )
    batter_event = batter_event.set_index('batter_id')
    debug_out.append("Batter event rolling stats computed.")

    inc("Computing pitcher rolling stats (vectorized, fast)...")
    if 'pitcher_id' in df_today.columns and 'pitcher_id' in df_hist.columns:
        pitcher_event = fast_rolling_stats(
            df_hist.rename(columns={"pitcher_id": "batter_id", "batter_id": "unused"}),
            "batter_id", "game_date", event_windows, main_pitch_types, prefix="p_"
        )
        pitcher_event = pitcher_event.set_index('batter_id')
        pitcher_event.index.name = 'pitcher_id'
        debug_out.append("Pitcher rolling stats computed.\nSample:\n" + str(pitcher_event.head(4)))
    else:
        pitcher_event = pd.DataFrame()
        debug_out.append("No pitcher stats found.")

    inc("Merging batter stats into today's data...")
    merged = df_today.set_index('batter_id').join(batter_event, how='left')

    inc("Merging pitcher stats into today's data...")
    if not pitcher_event.empty and 'pitcher_id' in df_today.columns:
        merged = merged.reset_index().set_index('pitcher_id').join(pitcher_event, how='left').reset_index()
        merged.rename(columns={'index': 'batter_id'}, inplace=True)
        debug_out.append("Merged pitcher stats into today's data. Sample:\n" + str(merged.head(4)))
    else:
        merged = merged.reset_index()
        debug_out.append("Skipped pitcher merge (empty).")

    inc("Final formatting and reindexing")
    missing_cols = [col for col in output_columns if col not in merged.columns]
    if missing_cols:
        nan_df = pd.DataFrame(np.nan, index=merged.index, columns=missing_cols)
        merged = pd.concat([merged, nan_df], axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    merged = merged.reindex(columns=output_columns)

    debug_out.append(f"🟢 Generated file with {merged.shape[0]} rows and {merged.shape[1]} columns.")

    inc(f"🟢 Generated file: {merged.shape[0]} unique batters, {merged.shape[1]} columns (features).")
    st.dataframe(merged.head(10))

    st.download_button(
        "⬇️ Download Today's Event-Level CSV (Exact Format)",
        data=merged.to_csv(index=False),
        file_name="event_level_today_full.csv"
    )

    # ---- Diagnostic text output for copy-paste ----
    st.text_area(
        "DEBUG OUTPUT (copy and paste this to share diagnostics):",
        value="\n\n".join(debug_out),
        height=500
    )

else:
    st.info("Please upload BOTH today's matchups/lineups and historical event-level CSV.")
