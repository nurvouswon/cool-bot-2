import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybaseball import statcast
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import requests
import pybaseball
pybaseball.cache.enable()

st.set_page_config(page_title="⚾ MLB HR Analyzer: Statcast + Context Power", layout="wide")

# --- MLB Team & Park Info (update for completeness) ---
team_code_to_park = {
    'PHI': 'citizens_bank_park', 'ATL': 'truist_park', 'NYM': 'citi_field', 'BOS': 'fenway_park',
    'NYY': 'yankee_stadium', 'CHC': 'wrigley_field', 'LAD': 'dodger_stadium', 'OAK': 'sutter_health_park',
    'CIN': 'great_american_ball_park', 'DET': 'comerica_park', 'HOU': 'minute_maid_park', 'MIA': 'loan_depot_park',
    'TB': 'tropicana_field', 'MIL': 'american_family_field', 'SD': 'petco_park', 'SF': 'oracle_park',
    'TOR': 'rogers_centre', 'CLE': 'progressive_field', 'MIN': 'target_field', 'KC': 'kauffman_stadium',
    'CWS': 'guaranteed_rate_field', 'LAA': 'angel_stadium', 'SEA': 't-mobile_park', 'TEX': 'globe_life_field',
    'ARI': 'chase_field', 'COL': 'coors_field', 'PIT': 'pnc_park', 'STL': 'busch_stadium', 'BAL': 'camden_yards',
    'WSH': 'nationals_park', 'ATH': 'sutter_health_park'
}
# Normalize alternate spellings
def normalize_park(park):
    if park in ["tmobile_park", "t-mobile_park"]:
        return "t-mobile_park"
    return park

# --- Weather API: For best results, use your real key via st.secrets["weather"]["api_key"]
@st.cache_data(show_spinner=False)
def get_weather(city, date, api_key):
    url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={date}"
    try:
        r = requests.get(url, timeout=12)
        if r.status_code == 200:
            data = r.json()
            # Closest to 19:00 (typical first pitch time)
            hour = min(data['forecast']['forecastday'][0]['hour'], key=lambda h: abs(int(h['time'].split()[1][:2]) - 19))
            return {
                'temp': hour['temp_f'],
                'wind_mph': hour['wind_mph'],
                'wind_dir': hour['wind_dir'],
                'humidity': hour['humidity']
            }
    except Exception:
        pass
    return {'temp': np.nan, 'wind_mph': np.nan, 'wind_dir': None, 'humidity': np.nan}

# -------------------- Streamlit UI -------------------
st.title("⚾ Statcast MLB HR Analyzer — All Strong Context Edges, No CSVs")
st.markdown("""
Fetches MLB Statcast batted ball events and **engineers advanced rolling, park, pitch, matchup, weather, and context features** for HR prediction.
- **No extra uploads needed**
- Exports event/player CSVs
- All edge features: rolling advanced stats, xwOBA/xSLG/xBA, park/handed/pitch HR rates, weather, pitch mix %, velocity, spin, and more.
""")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=60))
with col2:
    end_date = st.date_input("End Date", value=datetime.today())

if st.button("Fetch Statcast Data & Analyze", type="primary"):
    st.info("Pulling Statcast data... (can take several minutes for 30-60 day windows)")
    df = statcast(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    st.success(f"Loaded {len(df)} events from {start_date} to {end_date}")

    # --- Filter to batted ball events ---
    target_events = ['single', 'double', 'triple', 'home_run', 'field_out']
    df = df[df['events'].isin(target_events)].copy()

    # --- Add derived columns ---
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['park'] = df['home_team'].map(team_code_to_park).fillna(df['home_team'].str.lower().str.replace(" ", "_"))
    df['park'] = df['park'].apply(normalize_park)
    df['batter_id'] = df['batter']
    df['pitcher_id'] = df['pitcher']

    # --- Weather (API per park/date) ---
    if "weather" in st.secrets:
        api_key = st.secrets["weather"]["api_key"]
    else:
        st.warning("No weather API key found in st.secrets['weather']['api_key']! Weather columns will be NaN.")
        api_key = None
    weather_cols = ['temp', 'wind_mph', 'wind_dir', 'humidity']
    df['weather_key'] = df['park'] + "_" + df['game_date'].dt.strftime("%Y%m%d")
    wx_progress = st.progress(0, text="Weather (0%)")
    wx_keys = df['weather_key'].unique()
    for i, wx_key in enumerate(wx_keys):
        city = df[df['weather_key'] == wx_key].iloc[0]['park'].replace("_", " ").replace("t-mobile", "Seattle").replace("sutter health", "Las Vegas").title()
        date = df[df['weather_key'] == wx_key].iloc[0]['game_date'].strftime("%Y-%m-%d")
        weather = get_weather(city, date, api_key) if api_key else {k: np.nan for k in weather_cols}
        for col in weather_cols:
            df.loc[df['weather_key'] == wx_key, col] = weather[col]
        wx_progress.progress((i + 1) / len(wx_keys), text=f"Weather {i+1}/{len(wx_keys)}")
    wx_progress.empty()

    # --- Rolling features for batter/pitcher ---
    st.write("Engineering rolling stat features (batter/pitcher, 3/5/7/14)...")
    ROLL = [3, 5, 7, 14]
    batter_stats = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value', 'iso_value', 'xwoba', 'estimated_slg_using_speedangle', 'xba']
    pitcher_stats = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'woba_value', 'iso_value', 'xwoba', 'estimated_slg_using_speedangle', 'xba', 'release_speed', 'release_spin_rate']

    def add_rolling(df, group, stats, windows, prefix):
        for stat in stats:
            for w in windows:
                col = f"{prefix}_{stat}_{w}"
                df[col] = df.groupby(group)[stat].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        return df

    df = add_rolling(df, 'batter_id', batter_stats, ROLL, 'B')
    df = add_rolling(df, 'pitcher_id', pitcher_stats, ROLL, 'P')

    # --- Rolling pitch mix % (by batter/pitcher, by pitch_type) ---
    st.write("Calculating rolling pitch mix %...")
    pitch_types = df['pitch_type'].dropna().unique()
    for role, group in [('B', 'batter_id'), ('P', 'pitcher_id')]:
        for pt in pitch_types:
            for w in ROLL:
                col = f"{role}_PitchMix_{pt}_{w}"
                def pmix_func(x):
                    return (x.shift(1).rolling(w, min_periods=1)
                              .apply(lambda y: (y == pt).sum() / max(len(y), 1)))
                df[col] = df.groupby(group)['pitch_type'].transform(pmix_func)

    # --- Rolling park HR rate (season-to-date in current Statcast df) ---
    st.write("Calculating rolling park HR rate...")
    df['park_hr_rolling'] = (df.groupby('park')['events']
                               .transform(lambda x: x.shift(1).rolling(60, min_periods=1)
                               .apply(lambda y: (y == 'home_run').sum() / max(len(y), 1))))

    # --- Rolling handedness HR rate (batter + stand + p_throws) ---
    st.write("Calculating rolling handedness HR rate...")
    df['handed_context'] = df['stand'].astype(str) + "_" + df['p_throws'].astype(str)
    df['handed_hr_rolling'] = (df.groupby(['batter_id', 'handed_context'])['events']
                                 .transform(lambda x: x.shift(1).rolling(60, min_periods=1)
                                 .apply(lambda y: (y == 'home_run').sum() / max(len(y), 1))))

    # --- Rolling pitch type HR rate ---
    st.write("Calculating rolling pitch-type HR rate...")
    df['pitch_type_hr_rolling'] = (df.groupby(['batter_id', 'pitch_type'])['events']
                                     .transform(lambda x: x.shift(1).rolling(60, min_periods=1)
                                     .apply(lambda y: (y == 'home_run').sum() / max(len(y), 1))))

    # --- Batted ball directionality & context ---
    df['pull_side'] = (df['hc_x'] < 125).astype(int)
    df['pull_air'] = ((df['bb_type'] == 'fly_ball') & (df['hc_x'] < 125)).astype(int)
    df['flyball'] = (df['bb_type'] == 'fly_ball').astype(int)
    df['line_drive'] = (df['bb_type'] == 'line_drive').astype(int)
    df['groundball'] = (df['bb_type'] == 'ground_ball').astype(int)
    df['platoon'] = (df['stand'] != df['p_throws']).astype(int)
    df['hr_outcome'] = (df['events'] == 'home_run').astype(int)

    st.success("Feature engineering complete.")

    # --- Export event-level feature table (de-duplicate columns) ---
    export_cols = (
        ['game_date', 'batter', 'batter_id', 'pitcher', 'pitcher_id', 'events', 'description', 'stand', 'p_throws',
         'park', 'park_hr_rolling', 'temp', 'wind_mph', 'wind_dir', 'humidity', 'handed_context', 'handed_hr_rolling', 'pitch_type', 'pitch_type_hr_rolling']
        + [f"B_{stat}_{w}" for stat in batter_stats for w in ROLL]
        + [f"P_{stat}_{w}" for stat in pitcher_stats for w in ROLL]
        + [f"B_PitchMix_{pt}_{w}" for pt in pitch_types for w in ROLL]
        + [f"P_PitchMix_{pt}_{w}" for pt in pitch_types for w in ROLL]
        + ['pull_air', 'flyball', 'line_drive', 'groundball', 'pull_side', 'platoon', 'hr_outcome']
    )
    event_df = df.loc[:, ~df.columns.duplicated()].copy()
    event_df = event_df[[c for c in export_cols if c in event_df.columns]]

    st.markdown("#### Download Event-Level CSV (all features, 1 row per batted ball event):")
    st.dataframe(event_df.head(20))
    st.download_button("⬇️ Download Event-Level CSV", data=event_df.to_csv(index=False), file_name="event_level_hr_features.csv")

    # --- Player-level table (last row per batter, all advanced stats) ---
    st.markdown("#### Download Player-Level Rolling Feature CSV (1 row per batter):")
    player_cols = ['batter_id', 'batter'] + [c for c in event_df.columns if c.startswith('B_')]
    player_df = event_df.groupby(['batter_id', 'batter']).tail(1)[player_cols].reset_index(drop=True)
    st.dataframe(player_df.head(20))
    st.download_button("⬇️ Download Player-Level CSV", data=player_df.to_csv(index=False), file_name="player_level_hr_features.csv")

    # --- Logistic Regression: all context, advanced, and edge features ---
    st.markdown("#### Logistic Regression Weights (Standardized Features)")
    logit_features = [
        c for c in event_df.columns
        if (
            any(s in c for s in [
                'park_hr_rolling', 'temp', 'wind_mph', 'humidity',
                'B_', 'P_', '_PitchMix_', 'pull_air', 'flyball', 'line_drive',
                'groundball', 'pull_side', 'platoon', 'handed_hr_rolling', 'pitch_type_hr_rolling'
            ])
        ) and event_df[c].dtype in [np.float64, np.int64, float, int]
    ]
    model_df = event_df.dropna(subset=logit_features + ['hr_outcome'])
    X = model_df[logit_features].astype(float)
    y = model_df['hr_outcome'].astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=600)
    with st.spinner("Training logistic regression..."):
        model.fit(X_scaled, y)
    weights = pd.Series(model.coef_[0], index=logit_features).sort_values(ascending=False)
    weights_df = pd.DataFrame({'feature': weights.index, 'weight': weights.values})
    st.dataframe(weights_df.head(40))
    auc = roc_auc_score(y, model.predict_proba(X_scaled)[:, 1])
    st.write(f"Model ROC-AUC: **{auc:.3f}**")

    st.success("Full analysis done! All possible context edges and advanced stats are included.")

else:
    st.info("Set your date range and click 'Fetch Statcast Data & Analyze' to begin.")
