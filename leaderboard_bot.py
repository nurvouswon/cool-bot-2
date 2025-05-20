import streamlit as st
import pandas as pd
import requests
import pickle
import unicodedata
import difflib
import pytz
from datetime import datetime
from pybaseball import statcast, playerid_lookup, playerid_reverse_lookup
from pybaseball.lahman import people
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# ====== CONFIGURATION ======
API_KEY = st.secrets["general"]["weather_api"]

# Ballpark dictionary for park factors & city mapping
PARK_FACTORS = {
    "Yankee Stadium": 1.19, "Fenway Park": 0.97, "Coors Field": 1.30, "TBD": 1.0
}
BALLPARK_CITY = {
    "Yankee Stadium": "New York",
    "Fenway Park": "Boston",
    "Coors Field": "Denver",
    # ...add more mappings for every park...
}
DEFAULT_CITY = "TBD"

# ====== UTILITY FUNCTIONS ======
def normalize_name(name):
    if not isinstance(name, str): return ""
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = name.lower().replace('.', '').replace('-', ' ').replace("’", "'").strip()
    if ',' in name:
        last, first = name.split(',', 1)
        name = f"{first.strip()} {last.strip()}"
    return ' '.join(name.split())

# Handedness lookup with MLB API, Lahman, and manual fallback
MANUAL_HANDEDNESS = {
    # 'player name': ('Bats', 'Throws'),  # add manual fixes here as needed
}
def get_handedness(name):
    clean_name = normalize_name(name)
    parts = clean_name.split()
    if len(parts) >= 2:
        first, last = parts[0], parts[-1]
    else:
        first, last = clean_name, ""
    # 1. Try MLB API
    try:
        info = playerid_lookup(last.capitalize(), first.capitalize())
        if not info.empty and 'key_mlbam' in info.columns:
            mlbam_id = info.iloc[0]['key_mlbam']
            url = f'https://statsapi.mlb.com/api/v1/people/{mlbam_id}'
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                hand = data['people'][0]
                bats = hand['batSide']['code']  # "R", "L", or "S"
                throws = hand['pitchHand']['code']
                if bats and throws:
                    return bats, throws
    except Exception:
        pass
    # 2. Manual
    if clean_name in MANUAL_HANDEDNESS:
        return MANUAL_HANDEDNESS[clean_name]
    # 3. Lahman
    try:
        df = people()
        df['nname'] = (df['name_first'].fillna('') + ' ' + df['name_last'].fillna('')).map(normalize_name)
        match = df[df['nname'] == clean_name]
        if not match.empty:
            return match.iloc[0].get('bats', "UNK"), match.iloc[0].get('throws', "UNK")
        close = difflib.get_close_matches(clean_name, df['nname'].tolist(), n=1, cutoff=0.85)
        if close:
            row = df[df['nname'] == close[0]].iloc[0]
            return row.get('bats', "UNK"), row.get('throws', "UNK")
    except Exception:
        pass
    return "UNK", "UNK"

def fetch_schedule_df(date_str):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
    resp = requests.get(url)
    schedule = []
    try:
        games = resp.json()["dates"][0]["games"]
        for g in games:
            game_pk = g["gamePk"]
            venue = g["venue"]["name"]
            city = g["venue"].get("location") or BALLPARK_CITY.get(venue, DEFAULT_CITY)
            utc_dt = datetime.strptime(g["gameDate"][:19], "%Y-%m-%dT%H:%M:%S")
            utc_dt = pytz.utc.localize(utc_dt)
            pacific_dt = utc_dt.astimezone(pytz.timezone("US/Pacific"))
            start_time_pst = pacific_dt.strftime("%Y-%m-%d %H:%M")
            home_team = g["teams"]["home"]["team"]["name"]
            away_team = g["teams"]["away"]["team"]["name"]
            schedule.append({
                "game_pk": game_pk,
                "venue": venue,
                "city": city,
                "home_team": home_team,
                "away_team": away_team,
                "game_time_pst": start_time_pst,
            })
    except Exception as e:
        print(f"Error fetching schedule for {date_str}: {e}")
    return pd.DataFrame(schedule)

def get_weather(city, date, time, api_key=API_KEY):
    try:
        url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={date}"
        resp = requests.get(url)
        data = resp.json()
        hours = data['forecast']['forecastday'][0]['hour']
        game_hour = int(time.split(":")[0]) if time and ':' in time else 14
        weather_hour = min(hours, key=lambda h: abs(int(h['time'].split(' ')[1].split(':')[0]) - game_hour))
        return {
            "temp_f": weather_hour.get('temp_f', None),
            "wind_mph": weather_hour.get('wind_mph', None),
            "wind_dir": weather_hour.get('wind_dir', None),
            "humidity": weather_hour.get('humidity', None),
            "condition": weather_hour.get('condition', {}).get('text', None),
        }
    except Exception:
        return {"temp_f": None, "wind_mph": None, "wind_dir": None, "humidity": None, "condition": None}

def feature_prepare(df):
    df = df.copy()
    df['batter_hand'] = df['batter_hand'].map({'R':1, 'L':-1, 'S':0, 'UNK':0}).fillna(0)
    df['pitcher_hand'] = df['pitcher_hand'].map({'R':1, 'L':-1, 'S':0, 'UNK':0}).fillna(0)
    for col in ['temp_f','wind_mph','humidity','park_factor','launch_speed','launch_angle',
                'estimated_ba_using_speedangle','estimated_woba_using_speedangle']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

# ====== STREAMLIT UI ======
st.title("MLB HR Backtest + ML + Leaderboard Bot (Cool Bot 2, Batted Ball, Names, Weather, Handedness)")

mode = st.selectbox("Mode", [
    "Build CSV (Backtest Builder)",
    "Train Model",
    "Predict Today",
    "Leaderboard"
])

# ====== BUILD DATASET ======
if mode.startswith("Build"):
    st.header("Build Historical Dataset (Backtest Builder)")
    start = st.text_input("Start Date", "2025-05-12")
    end = st.text_input("End Date", "2025-05-19")
    if st.button("Build CSV"):
        st.info("Fetching Statcast and weather data (this may take several minutes for large ranges)...")
        try:
            df_all = statcast(start, end)
        except Exception as e:
            st.error(f"Error fetching statcast: {e}")
            st.stop()
        if df_all.empty:
            st.error("No Statcast data fetched for given date range.")
            st.stop()
        batted_df = df_all[df_all['type'] == 'X'].copy()
        if batted_df.empty:
            st.error("No batted ball events for selected date range.")
            st.stop()
        # Add batter names
        batter_ids = batted_df['batter'].unique().tolist()
        batter_lookup = playerid_reverse_lookup(batter_ids, key_type='mlbam')
        batted_df = batted_df.merge(batter_lookup[['key_mlbam','name_first','name_last']],
                                   left_on='batter', right_on='key_mlbam', how='left')
        batted_df['batter_name'] = batted_df['name_first'].fillna('') + ' ' + batted_df['name_last'].fillna('')
        # Add pitcher names
        pitcher_ids = batted_df['pitcher'].unique().tolist()
        pitcher_lookup = playerid_reverse_lookup(pitcher_ids, key_type='mlbam')
        pitcher_lookup = pitcher_lookup.rename(columns={'key_mlbam': 'pitcher', 'name_first': 'pitcher_first', 'name_last': 'pitcher_last'})
        batted_df = batted_df.merge(pitcher_lookup[['pitcher','pitcher_first','pitcher_last']],
                                    on='pitcher', how='left')
        batted_df['pitcher_name'] = batted_df['pitcher_first'].fillna('') + ' ' + batted_df['pitcher_last'].fillna('')
        # Schedule merge
        date_range = pd.date_range(start, end)
        sched_frames = []
        for date in date_range:
            sched_frames.append(fetch_schedule_df(date.strftime("%Y-%m-%d")))
        schedule_df = pd.concat(sched_frames, ignore_index=True)
        df = batted_df.merge(schedule_df, on="game_pk", how="left")
        # Features
        weather_rows, batter_hands, pitcher_hands, pf_list = [], [], [], []
        progress_bar = st.progress(0)
        progress_txt = st.empty()
        total = len(df)
        for idx, row in df.iterrows():
            pf = PARK_FACTORS.get(row['venue'], 1.0)
            pf_list.append(pf)
            bats, _ = get_handedness(row['batter_name'])
            _, throws = get_handedness(row['pitcher_name'])
            batter_hands.append(bats if bats else "UNK")
            pitcher_hands.append(throws if throws else "UNK")
            city_val = row['city'] if pd.notnull(row['city']) and row['city'] != "" else BALLPARK_CITY.get(row['venue'], DEFAULT_CITY)
            date_val = str(row['game_time_pst'])[:10] if pd.notnull(row['game_time_pst']) else str(row['game_date'])[:10]
            time_val = str(row['game_time_pst'])[11:16] if pd.notnull(row['game_time_pst']) else "14:00"
            weather = get_weather(city_val, date_val, time_val, API_KEY)
            weather_rows.append(weather)
            pct = int((idx + 1) / total * 100)
            progress_bar.progress((idx + 1) / total)
            progress_txt.markdown(f"**Progress:** {pct}% ({idx + 1} of {total})")
        df['park_factor'] = pf_list
        df['batter_hand'] = batter_hands
        df['pitcher_hand'] = pitcher_hands
        df['temp_f'] = [w['temp_f'] for w in weather_rows]
        df['wind_mph'] = [w['wind_mph'] for w in weather_rows]
        df['wind_dir'] = [w['wind_dir'] for w in weather_rows]
        df['humidity'] = [w['humidity'] for w in weather_rows]
        df['condition'] = [w['condition'] for w in weather_rows]
        df['is_hr'] = (df['events'] == "home_run").astype(int)
        # Save only columns of interest
        cols = [
            'game_date','game_pk','home_team','away_team','venue','city','game_time_pst',
            'batter','batter_name','batter_hand',
            'pitcher','pitcher_name','pitcher_hand',
            'events','bb_type','launch_speed','launch_angle',
            'estimated_ba_using_speedangle','estimated_woba_using_speedangle','hc_x','hc_y',
            'park_factor','temp_f','wind_mph','wind_dir','humidity','condition',
            'is_hr'
        ]
        present_cols = [c for c in cols if c in df.columns]
        df = df[present_cols]
        st.dataframe(df.head(20))
        csv = df.to_csv(index=False).encode()
        st.download_button("Download Backtest CSV", csv, "mlb_hr_batted_balls.csv")
        st.success("Done! Batted ball CSV (with names, advanced stats, weather, handedness) ready.")

# ====== TRAIN MODEL ======
elif mode == "Train Model":
    st.header("Train ML Model")
    uploaded = st.file_uploader("Upload CSV built from 'Build CSV' step", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Data preview:", df.head())
        features = [
            'batter_hand', 'pitcher_hand', 'park_factor', 'temp_f', 'wind_mph', 'humidity',
            'launch_speed', 'launch_angle', 'estimated_ba_using_speedangle', 'estimated_woba_using_speedangle'
        ]
        df = feature_prepare(df)
        X = df[features]
        y = df['is_hr']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_choice = st.radio("Choose model:", ["RandomForest", "LightGBM" if LIGHTGBM_AVAILABLE else "LightGBM (not installed)"])
        if model_choice == "RandomForest" or not LIGHTGBM_AVAILABLE:
            model = RandomForestClassifier(n_estimators=80, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        else:
            model = lgb.LGBMClassifier(n_estimators=120, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        auc = roc_auc_score(y_test, preds)
        acc = accuracy_score(y_test, preds)
        st.write(f"Test ROC AUC: {auc:.3f}  |  Accuracy: {acc:.3f}")
        with open("mlb_hr_model.pkl", "wb") as f:
            pickle.dump(model, f)
        st.success("Model trained and saved as mlb_hr_model.pkl")

# ====== PREDICT TODAY ======
elif mode == "Predict Today":
    st.header("Predict Today's Home Runs")
    uploaded = st.file_uploader("Upload new games CSV (with required columns)", type="csv")
    if uploaded:
        try:
            with open("mlb_hr_model.pkl", "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            st.error("Model not found. Please train model first!")
            st.stop()
        df_pred = pd.read_csv(uploaded)
        df_pred = feature_prepare(df_pred)
        features = [
            'batter_hand', 'pitcher_hand', 'park_factor', 'temp_f', 'wind_mph', 'humidity',
            'launch_speed', 'launch_angle', 'estimated_ba_using_speedangle', 'estimated_woba_using_speedangle'
        ]
        pred_probs = model.predict_proba(df_pred[features])[:, 1]
        df_pred['HR_Prob'] = pred_probs
        st.dataframe(df_pred.sort_values("HR_Prob", ascending=False).head(20))
        csv = df_pred.to_csv(index=False).encode()
        st.download_button("Download Prediction CSV", csv, "mlb_hr_predicted.csv")
        st.success("Predictions complete.")

# ====== LEADERBOARD ======
elif mode == "Leaderboard":
    st.header("Home Run Leaderboard (Sample)")
    uploaded = st.file_uploader("Upload predictions CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        if 'HR_Prob' not in df.columns:
            st.error("No HR_Prob column in uploaded CSV!")
        else:
            st.write(df.sort_values("HR_Prob", ascending=False).head(25)[['batter_name','pitcher_name','venue','HR_Prob']])
