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
            out_row[f"{prefix}barrel_rate_{w}"] = (( (group['launch_speed'] >= 98) &  
                                                     (group['launch_angle'] >= 26) &  
                                                     (group['launch_angle'] <= 30) )  
                                                   .rolling(w, min_periods=1).mean().iloc[-1])  
            out_row[f"{prefix}fb_rate_{w}"] = (group['launch_angle'].rolling(w, min_periods=1)  
                                               .apply(lambda x: np.mean(x >= 25)).iloc[-1])  
            out_row[f"{prefix}sweet_spot_rate_{w}"] = (group['launch_angle'].rolling(w, min_periods=1)  
                                                       .apply(lambda x: np.mean((x >= 8) & (x <= 32))).iloc[-1])  
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
                        out_row[f"{prefix}barrel_rate_{pt}_{w}"] = ( ( (pt_group['launch_speed'] >= 98) &  
                                                                        (pt_group['launch_angle'] >= 26) &  
                                                                        (pt_group['launch_angle'] <= 30) )  
                                                                      .rolling(w, min_periods=1).mean().iloc[-1])  
                        out_row[f"{prefix}fb_rate_{pt}_{w}"] = (pt_group['launch_angle'].rolling(w, min_periods=1)  
                                                                .apply(lambda x: np.mean(x >= 25)).iloc[-1])  
                        out_row[f"{prefix}sweet_spot_rate_{pt}_{w}"] = (pt_group['launch_angle'].rolling(w, min_periods=1)  
                                                                        .apply(lambda x: np.mean((x >= 8) & (x <= 32))).iloc[-1])  
        out_row[id_col] = name  
        feature_frames.append(out_row)  
    return pd.DataFrame(feature_frames)  
  
def pct(x):  
    return f"{x}%"  
  
# --- Progress bar utilities ---  
step = [0]  
step_total = 8  
progress = st.progress(0)  
status = st.empty()  
def inc(msg=None):  
    step[0] += 1  
    percent = min(int(100 * step[0] / step_total), 100)  
    progress.progress(percent)  
    status.markdown(f"{pct(percent)} {'- ' + msg if msg else ''}")  
    time.sleep(0.05)  
  
# Diagnostic log setup
diagnostic_log = []  # All diagnostic information will go here

# Add a helper to log diagnostics
def log_diagnostic(msg):
    diagnostic_log.append(msg)
    st.info(msg)

# --- MAIN LOGIC ---  
if today_file and hist_file:  
    inc("Loading data")  
    df_today = pd.read_csv(today_file)  
    df_hist = pd.read_csv(hist_file)  
    log_diagnostic("Loaded Today's matchups and historical event data.")  
    log_diagnostic(f"Today's Data Sample: {df_today.head(2)}")  
    log_diagnostic(f"Historical Data Sample: {df_hist.head(2)}")  
  
    inc("Standardizing columns and IDs")  
    df_today.columns = [str(c).strip().lower().replace(" ", "_") for c in df_today.columns]  
    df_hist.columns = [str(c).strip().lower().replace(" ", "_") for c in df_hist.columns]  
    log_diagnostic("Standardized columns.")  
  
    id_cols_today = ['batter_id', 'mlb_id']  
    id_col_today = next((c for c in id_cols_today if c in df_today.columns), None)  
    if id_col_today is None:  
        st.error("Today's file must have 'batter_id' or 'mlb_id' column.")  
        st.stop()  

    df_today['batter_id'] = df_today[id_col_today].astype(str).str.strip().str.replace('.0', '', regex=False)  
    df_today = df_today.drop_duplicates(subset=["batter_id"])  
    df_today = parse_weather_fields(df_today)  
    log_diagnostic(f"Parsed weather data for today's lineup.")  
  
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
    log_diagnostic(f"Batter event rolling stats computed.")  
  
    inc("Computing pitcher rolling stats (vectorized, fast)...")  
    if 'pitcher_id' in df_today.columns and 'pitcher_id' in df_hist.columns:  
        pitcher_event = fast_rolling_stats(  
            df_hist.rename(columns={"pitcher_id": "batter_id", "batter_id": "unused"}),  
            "batter_id", "game_date", event_windows, main_pitch_types, prefix="p_"  
        )  
        pitcher_event = pitcher_event.set_index('batter_id')  
        pitcher_event.index.name = 'pitcher_id'  
        log_diagnostic(f"Pitcher event rolling stats computed.")  
    else:  
        pitcher_event = pd.DataFrame()  
        log_diagnostic("No pitcher stats found.")  
  
    inc("Merging batter and pitcher stats into today's data...")  
    df_today["_orig_idx"] = np.arange(len(df_today))  
    merged = df_today.set_index("batter_id").join(batter_event, how="left")  
    if not pitcher_event.empty and "pitcher_id" in df_today.columns:  
        merged = merged.reset_index()
        merged = merged.merge(  
            pitcher_event.reset_index(),  
            how="left",  
            left_on="pitcher_id",  
            right_on="pitcher_id",  
            suffixes=("", "_p")  
        )  
    else:  
        merged = merged.reset_index()  
  
    # Final formatting
    missing_cols = [col for col in output_columns if col not in merged.columns]  
    if missing_cols:  
        nan_df = pd.DataFrame(np.nan, index=merged.index, columns=missing_cols)  
        merged = pd.concat([merged, nan_df], axis=1)  
  
    merged = merged.sort_values("_orig_idx").drop(columns=["_orig_idx"], errors="ignore")  
    inc(f"🟢 Generated file: {merged.shape[0]} unique batters, {merged.shape[1]} columns (features).")  
    st.dataframe(merged.head(10))  
  
    # Diagnostic download
    diagnostic_log.append(f"🟢 Generated file with {merged.shape[0]} rows and {merged.shape[1]} columns.")
    diagnostic_text = "\n".join(diagnostic_log)
  
    st.download_button(  
        "⬇️ Download Diagnostic Log",  
        data=diagnostic_text,  
        file_name="diagnostic_log.txt"  
    )  
  
    st.download_button(  
        "⬇️ Download Today's Event-Level CSV",  
        data=merged.to_csv(index=False),  
        file_name="event_level_today_full.csv"  
    )  

else:  
    st.info("Please upload BOTH today's matchups/lineups and historical event-level CSV.")
