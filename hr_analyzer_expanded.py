import streamlit as st
import pandas as pd
import numpy as np
import re
import time

st.set_page_config("🟦 Generate Today's Event-Level CSV", layout="wide")
st.title("🟦 Generate Today's Event-Level CSV (w/ All Custom Features)")

st.markdown("""
**Instructions:**
- Upload Today's Lineups/Matchups CSV (must have batter_id or mlb_id, player_name, etc).
- Upload Historical Event-Level CSV (must have batter_id, pitcher_id, pitch_type, game_date, and all stat columns).
- Output: ONE row per batter with ALL rolling/stat features (3/7/14/20, batter & pitcher, per pitch type and overall).
""")

@st.cache_data(show_spinner=False)
def load_csv(file):
    return pd.read_csv(file)

today_file = st.file_uploader("Upload Today's Matchups/Lineups CSV", type=["csv"], key="today_csv")
hist_file = st.file_uploader("Upload Historical Event-Level CSV", type=["csv"], key="hist_csv")

# --- Config ---
event_windows = [3, 5, 7, 14, 20, 30]  # Add any window you use
main_pitch_types = ["ff", "sl", "cu", "ch", "si", "fc", "fs", "st", "sinker", "splitter", "sweeper"]

# --- Feature Definitions ---
rolling_feats = [
    # (col_base, stat_fn, default_nan)
    ("launch_speed", lambda x: x.mean(), np.nan),
    ("launch_angle", lambda x: x.mean(), np.nan),
    ("hit_distance_sc", lambda x: x.mean(), np.nan),
    ("woba_value", lambda x: x.mean(), np.nan),
    ("release_speed", lambda x: x.mean(), np.nan),
    ("release_spin_rate", lambda x: x.mean(), np.nan),
    ("spin_axis", lambda x: x.mean(), np.nan),
    ("pfx_x", lambda x: x.mean(), np.nan),
    ("pfx_z", lambda x: x.mean(), np.nan),
    ("hard_hit", lambda x: x.mean(), np.nan),
    ("sweet_spot", lambda x: x.mean(), np.nan),
]

# HR features
hr_cols = [
    "hr_outcome",      # must be present in your event-level csv
    "park",            # park/ballpark column
    "p_throws",        # pitcher handedness
    "stand"            # batter handedness
]

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

def fill_pitcher_id(df):
    df = df.copy()
    team_game_keys = ["team_code", "game_date", "stadium"]
    if not all(col in df.columns for col in team_game_keys + ["position", "mlb_id"]):
        st.warning("Cannot assign pitcher_id; missing required columns in today's file.")
        df["pitcher_id"] = np.nan
        return df
    sp_mask = df["position"].astype(str).str.upper().isin(["SP", "P"])
    df["pitcher_id"] = np.nan
    for key_vals, subdf in df.groupby(team_game_keys):
        sp_rows = subdf[sp_mask]
        if not sp_rows.empty:
            sp_id = sp_rows.iloc[0]["mlb_id"]
            idxs = subdf.index
            df.loc[idxs, "pitcher_id"] = str(sp_id)
    return df

@st.cache_data(show_spinner=False)
def compute_rolling_stats(df, id_col, date_col, windows, prefix=""):
    out = []
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values([id_col, date_col])
    df['hr'] = (df['events'] == "home_run") | (df.get('hr_outcome', '') == 1)  # fallback logic
    df['hard_hit'] = df['launch_speed'] >= 95
    df['sweet_spot'] = df['launch_angle'].between(8,32)
    gb = df.groupby(id_col)
    for name, group in gb:
        row = {}
        for w in windows:
            _group = group.tail(w)
            # Simple stats
            for col, fn, nan in rolling_feats:
                if col in _group:
                    row[f"{prefix}{col}_{w}"] = fn(_group[col].dropna()) if _group[col].notnull().any() else nan
            # HR rate
            row[f"{prefix}hr_rate_{w}"] = _group['hr'].mean() if len(_group) else np.nan
            row[f"{prefix}hard_hit_rate_{w}"] = _group['hard_hit'].mean() if len(_group) else np.nan
            row[f"{prefix}sweet_spot_rate_{w}"] = _group['sweet_spot'].mean() if len(_group) else np.nan
            # Park HR (placeholder: overall in window)
            if 'park' in _group and 'hr' in _group:
                park_group = _group.groupby('park')['hr'].mean()
                row[f"{prefix}park_hr_{w}"] = park_group.mean() if not park_group.empty else np.nan
        # Add by-pitch-type stats if present
        if "pitch_type" in group.columns:
            for pt in main_pitch_types:
                pt_group = group[group['pitch_type'] == pt]
                for w in windows:
                    if not pt_group.empty:
                        row[f"{prefix}hr_rate_{pt}_{w}"] = pt_group['hr'].tail(w).mean()
                    else:
                        row[f"{prefix}hr_rate_{pt}_{w}"] = np.nan
        row[id_col] = name
        out.append(row)
    return pd.DataFrame(out)

@st.cache_data(show_spinner=False)
def compute_hand_hr(df, id_col, group_col, hand_col, windows, prefix=""):
    # e.g. for b_vsp_hand_hr_7 (batter vs pitcher hand)
    out = []
    df = df.copy()
    df['hr'] = (df['events'] == "home_run") | (df.get('hr_outcome', '') == 1)
    gb = df.groupby(id_col)
    for name, group in gb:
        row = {}
        for w in windows:
            for hand in ['R', 'L']:
                _group = group[group[hand_col] == hand].tail(w)
                row[f"{prefix}{group_col}_hand_{hand}_HR_{w}"] = _group['hr'].mean() if len(_group) else np.nan
        row[id_col] = name
        out.append(row)
    return pd.DataFrame(out)

def join_features(df_today, batter_df, pitcher_df, batter_hand_df=None, pitcher_hand_df=None):
    df = df_today.copy()
    df = df.set_index('batter_id').join(batter_df.set_index('batter_id'), how='left')
    df = df.reset_index()
    df = df.set_index('pitcher_id').join(pitcher_df.set_index('pitcher_id'), how='left', rsuffix='_pitcher')
    df = df.reset_index()
    if batter_hand_df is not None:
        df = df.set_index('batter_id').join(batter_hand_df.set_index('batter_id'), how='left', rsuffix='_bh')
        df = df.reset_index()
    if pitcher_hand_df is not None:
        df = df.set_index('pitcher_id').join(pitcher_hand_df.set_index('pitcher_id'), how='left', rsuffix='_ph')
        df = df.reset_index()
    return df

# --- MAIN APP LOGIC ---
if today_file and hist_file:
    st.info("Loaded Today's matchups and historical event data.")

    df_today = load_csv(today_file)
    df_hist = load_csv(hist_file)

    st.write("Today's Data Sample:", df_today.head(2))
    st.write("Historical Data Sample:", df_hist.head(2))

    # Standardize columns
    df_today.columns = [str(c).strip().lower().replace(" ", "_") for c in df_today.columns]
    df_hist.columns = [str(c).strip().lower().replace(" ", "_") for c in df_hist.columns]
    st.write("Standardized columns.")

    for col in ['batter_id', 'pitcher_id', 'mlb_id']:
        if col in df_today.columns:
            df_today[col] = df_today[col].astype(str).str.replace('.0', '', regex=False).str.strip()
        if col in df_hist.columns:
            df_hist[col] = df_hist[col].astype(str).str.replace('.0', '', regex=False).str.strip()

    # Assign pitcher_id to today's batters
    df_today = fill_pitcher_id(df_today)
    st.write("Pitcher_id column filled. Sample:", df_today[["team_code", "game_date", "game_number", "position", "mlb_id", "pitcher_id"]].head(8))

    df_today = parse_weather_fields(df_today)
    st.write("Parsed weather data for today's lineup.")

    # Rolling stats for batters
    batter_event = compute_rolling_stats(df_hist, "batter_id", "game_date", event_windows, prefix="")
    batter_event = batter_event.set_index('batter_id')
    st.write("Batter event rolling stats computed.")

    # Rolling stats for pitchers
    pitcher_event = compute_rolling_stats(
        df_hist.rename(columns={"pitcher_id": "batter_id", "batter_id": "unused"}),
        "batter_id", "game_date", event_windows, prefix="p_"
    )
    pitcher_event = pitcher_event.set_index('batter_id')
    pitcher_event.index.name = 'pitcher_id'
    st.write("Pitcher rolling stats computed. Sample:", pitcher_event.head(4))

    # Handedness HR stats (batter vs pitcher hand, etc.)
    if 'stand' in df_hist.columns and 'p_throws' in df_hist.columns:
        batter_hand_event = compute_hand_hr(df_hist, "batter_id", "vsp", "p_throws", event_windows, prefix="b_")
        pitcher_hand_event = compute_hand_hr(
            df_hist.rename(columns={"pitcher_id": "batter_id", "batter_id": "unused"}),
            "batter_id", "vsb", "stand", event_windows, prefix="p_"
        )
        batter_hand_event = batter_hand_event.set_index('batter_id')
        pitcher_hand_event = pitcher_hand_event.set_index('batter_id')
        pitcher_hand_event.index.name = 'pitcher_id'
    else:
        batter_hand_event = None
        pitcher_hand_event = None

    # Merge all features
    merged = join_features(df_today, batter_event, pitcher_event, batter_hand_event, pitcher_hand_event)

    # Fill missing columns with NaN
    final_cols = list(df_today.columns) + list(batter_event.columns) + list(pitcher_event.columns)
    if batter_hand_event is not None:
        final_cols += list(batter_hand_event.columns)
    if pitcher_hand_event is not None:
        final_cols += list(pitcher_hand_event.columns)
    merged = merged.loc[:, ~merged.columns.duplicated()]

    st.success(f"🟢 Generated file with {merged.shape[0]} rows and {merged.shape[1]} columns.")
    st.dataframe(merged.head(10))

    st.download_button(
        "⬇️ Download Today's Event-Level CSV (Exact Format)",
        data=merged.to_csv(index=False),
        file_name="event_level_today_full.csv"
    )

else:
    st.info("Please upload BOTH today's matchups/lineups and historical event-level CSV.")
