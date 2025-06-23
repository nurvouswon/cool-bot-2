import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config("🟦 Generate Today's Event-Level CSV", layout="wide")
st.title("🟦 Generate Today's Event-Level CSV (FAST, Optimized for Large Data)")

st.markdown("""
**Instructions:**
- Upload Today's Lineups/Matchups CSV (must have `batter_id` or `mlb_id`, `player_name`, etc).
- Upload Historical Event-Level CSV (must have `batter_id`, `pitcher_id`, `pitch_type`, `game_date`, and all stat columns).
- Output: ONE row per batter with ALL rolling/stat features (3/7/14/20, batter & pitcher, per pitch type and overall).
""")

today_file = st.file_uploader("Upload Today's Matchups/Lineups CSV", type=["csv"], key="today_csv")
hist_file = st.file_uploader("Upload Historical Event-Level CSV", type=["csv"], key="hist_csv")

event_windows = [3, 7, 14, 20]
main_pitch_types = ["ff", "sl", "cu", "ch", "si", "fc", "fs", "st", "sinker", "splitter", "sweeper"]  # expand if needed

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

def batter_pitcher_rolling(df, id_col, stat_cols, event_windows, pitch_types=None, prefix=''):
    df = df.copy()
    df = df.sort_values([id_col, 'game_date'])
    # Overall stats (no pitch_type split)
    for stat, info in stat_cols.items():
        func = info['func']
        base_col = info['base']
        for w in event_windows:
            colname = f"{prefix}{stat}_{w}"
            grouped = df.groupby(id_col)[base_col]
            if func == 'mean':
                stat_result = grouped.transform(lambda x: x.rolling(w, min_periods=1).mean())
            elif func == 'hard_hit':
                stat_result = grouped.transform(lambda x: (x >= 95).rolling(w, min_periods=1).mean())
            elif func == 'barrel':
                stat_result = grouped.transform(
                    lambda x: (
                        ((df.loc[x.index, 'launch_speed'] >= 98) &
                         (df.loc[x.index, 'launch_angle'] >= 26) &
                         (df.loc[x.index, 'launch_angle'] <= 30))
                        .rolling(w, min_periods=1).mean()
                    )
                )
            elif func == 'fb_rate':
                stat_result = grouped.transform(lambda x: (x >= 25).rolling(w, min_periods=1).mean())
            elif func == 'sweet_spot':
                stat_result = grouped.transform(lambda x: ((x >= 8) & (x <= 32)).rolling(w, min_periods=1).mean())
            df[colname] = stat_result
    # Per pitch type
    if pitch_types is not None and 'pitch_type' in df.columns:
        for pt in pitch_types:
            mask = (df['pitch_type'].str.lower() == pt)
            for stat, info in stat_cols.items():
                func = info['func']
                base_col = info['base']
                for w in event_windows:
                    colname = f"{prefix}{stat}_{pt}_{w}"
                    grouped = df[mask].groupby(id_col)[base_col]
                    if func == 'mean':
                        stat_result = grouped.transform(lambda x: x.rolling(w, min_periods=1).mean())
                    elif func == 'hard_hit':
                        stat_result = grouped.transform(lambda x: (x >= 95).rolling(w, min_periods=1).mean())
                    elif func == 'barrel':
                        stat_result = grouped.transform(
                            lambda x: (
                                ((df.loc[x.index, 'launch_speed'] >= 98) &
                                 (df.loc[x.index, 'launch_angle'] >= 26) &
                                 (df.loc[x.index, 'launch_angle'] <= 30))
                                .rolling(w, min_periods=1).mean()
                            )
                        )
                    elif func == 'fb_rate':
                        stat_result = grouped.transform(lambda x: (x >= 25).rolling(w, min_periods=1).mean())
                    elif func == 'sweet_spot':
                        stat_result = grouped.transform(lambda x: ((x >= 8) & (x <= 32)).rolling(w, min_periods=1).mean())
                    df.loc[mask, colname] = stat_result
    # Only keep last row per id
    result = df.sort_values('game_date').groupby(id_col).tail(1).set_index(id_col)
    return result

# Stats to compute for both batter and pitcher
stat_cols = {
    'avg_exit_velo': {'base': 'launch_speed', 'func': 'mean'},
    'hard_hit_rate': {'base': 'launch_speed', 'func': 'hard_hit'},
    'barrel_rate': {'base': 'launch_speed', 'func': 'barrel'},
    'fb_rate': {'base': 'launch_angle', 'func': 'fb_rate'},
    'sweet_spot_rate': {'base': 'launch_angle', 'func': 'sweet_spot'},
}

# The full output columns, built up for every stat, window, and pitch type
output_columns = [
    "team_code","game_date","game_number","mlb_id","player_name","batting_order","position",
    "weather","temp","wind_mph","wind_dir","condition","stadium","city","batter_id","p_throws",
]
# Add all rolling stat columns dynamically
for s in stat_cols:
    for w in event_windows:
        output_columns.append(f"{s}_{w}")
        output_columns.append(f"p_{s}_{w}")
    for pt in main_pitch_types:
        for w in event_windows:
            output_columns.append(f"{s}_{pt}_{w}")
            output_columns.append(f"p_{s}_{pt}_{w}")

if today_file and hist_file:
    df_today = pd.read_csv(today_file)
    df_hist = pd.read_csv(hist_file)

    st.success(f"Today's Matchups file loaded: {df_today.shape[0]} rows, {df_today.shape[1]} columns.")
    st.success(f"Historical Event-Level file loaded: {df_hist.shape[0]} rows, {df_hist.shape[1]} columns.")

    df_today.columns = [str(c).strip().lower().replace(" ", "_") for c in df_today.columns]
    df_hist.columns = [str(c).strip().lower().replace(" ", "_") for c in df_hist.columns]

    id_cols_today = ['batter_id', 'mlb_id']
    id_col_today = next((c for c in id_cols_today if c in df_today.columns), None)
    if id_col_today is None:
        st.error("Today's file must have 'batter_id' or 'mlb_id' column.")
        st.stop()
    if 'batter_id' not in df_hist.columns or 'pitcher_id' not in df_hist.columns:
        st.error("Historical file must have both 'batter_id' and 'pitcher_id' columns.")
        st.stop()
    df_today['batter_id'] = df_today[id_col_today].astype(str).str.strip().str.replace('.0', '', regex=False)
    df_hist['batter_id'] = df_hist['batter_id'].astype(str).str.strip().str.replace('.0', '', regex=False)
    df_hist['pitcher_id'] = df_hist['pitcher_id'].astype(str).str.strip().str.replace('.0', '', regex=False)
    df_today = df_today.drop_duplicates(subset=["batter_id"])
    df_today = parse_weather_fields(df_today)
    if 'game_date' not in df_hist.columns:
        st.error("Historical file must have a 'game_date' column.")
        st.stop()
    df_hist['game_date'] = pd.to_datetime(df_hist['game_date'], errors='coerce')

    # Coerce stat columns
    for stat in set([d['base'] for d in stat_cols.values()]):
        if stat in df_hist.columns:
            df_hist[stat] = pd.to_numeric(df_hist[stat], errors='coerce')
    if 'launch_angle' in df_hist.columns:
        df_hist['launch_angle'] = pd.to_numeric(df_hist['launch_angle'], errors='coerce')

    # ---- Batter event rolling, per pitch type and overall (FAST) ----
    st.info("Computing batter rolling stats...")
    batter_event = batter_pitcher_rolling(
        df_hist, "batter_id", stat_cols, event_windows, main_pitch_types, prefix=""
    )
    batter_event = batter_event.reset_index()
    st.success(f"Batter rolling stats shape: {batter_event.shape}")

    # ---- Pitcher event rolling, per pitch type and overall (FAST) ----
    st.info("Computing pitcher rolling stats...")
    df_hist_p = df_hist.rename(columns={"pitcher_id": "batter_id"})
    pitcher_event = batter_pitcher_rolling(
        df_hist_p, "batter_id", {f"p_{k}": {'base': v['base'], 'func': v['func']} for k, v in stat_cols.items()},
        event_windows, main_pitch_types, prefix="p_"
    )
    pitcher_event = pitcher_event.reset_index().rename(columns={'batter_id': 'pitcher_id'})
    st.success(f"Pitcher rolling stats shape: {pitcher_event.shape}")

    # ---- Merge batter and pitcher stats into today's lineups ----
    st.info("Merging data...")
    merged = df_today.merge(batter_event, on='batter_id', how='left', suffixes=('', '_batter'))
    if 'pitcher_id' in df_today.columns:
        df_today['pitcher_id'] = df_today['pitcher_id'].astype(str).str.strip().str.replace('.0', '', regex=False)
        merged = merged.merge(pitcher_event, left_on='pitcher_id', right_on='pitcher_id', how='left', suffixes=('', '_pitcher'))

    # ---- Add any missing columns all at once to avoid fragmentation ----
    missing_cols = [col for col in output_columns if col not in merged.columns]
    if missing_cols:
        nan_df = pd.DataFrame(np.nan, index=merged.index, columns=missing_cols)
        merged = pd.concat([merged, nan_df], axis=1)
    merged = merged.loc[:,~merged.columns.duplicated()]
    merged = merged.reindex(columns=output_columns)

    st.success(f"🟢 Generated file: {merged.shape[0]} unique batters, {merged.shape[1]} columns (features).")
    st.dataframe(merged.head(10))

    st.download_button(
        "⬇️ Download Today's Event-Level CSV (Exact Format)",
        data=merged.to_csv(index=False),
        file_name="event_level_today_full.csv"
    )
else:
    st.info("Please upload BOTH today's matchups/lineups and historical event-level CSV.")
