import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import lightgbm as lgb
import unicodedata
import os
import joblib

# ---------- SETUP ----------

API_KEY = st.secrets.get("weather_api", "YOUR_WEATHER_API_KEY_HERE")

VENUE_TO_CITY = {
    "Wrigley Field": "Chicago",
    "Yankee Stadium": "New York",
    "Fenway Park": "Boston",
    "Dodger Stadium": "Los Angeles",
    "Petco Park": "San Diego",
    "Coors Field": "Denver",
    "Globe Life Field": "Arlington",
    "Oracle Park": "San Francisco",
    "Minute Maid Park": "Houston",
    "Chase Field": "Phoenix",
    "Target Field": "Minneapolis",
    "Citi Field": "New York",
    "Great American Ball Park": "Cincinnati",
    "Camden Yards": "Baltimore",
    "Busch Stadium": "St. Louis",
    "Guaranteed Rate Field": "Chicago",
    "Nationals Park": "Washington",
    "Comerica Park": "Detroit",
    "T-Mobile Park": "Seattle",
    "Truist Park": "Atlanta",
    "LoanDepot Park": "Miami",
    "Progressive Field": "Cleveland",
    "American Family Field": "Milwaukee",
    "Rogers Centre": "Toronto",
    "Tropicana Field": "St. Petersburg",
    "Kauffman Stadium": "Kansas City",
    "Oakland Coliseum": "Oakland",
    "Angel Stadium": "Anaheim",
    "PNC Park": "Pittsburgh",
    "Citizens Bank Park": "Philadelphia"
    # ...add more as needed...
}

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = ''.join(
        c for c in unicodedata.normalize('NFD', name)
        if unicodedata.category(c) != 'Mn'
    )
    name = name.lower().replace('.', '').replace('-', ' ').replace("’", "'").strip()
    if ',' in name:
        last, first = name.split(',', 1)
        name = f"{first.strip()} {last.strip()}"
    return ' '.join(name.split())

def get_hand_from_mlb(player_id):
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{int(player_id)}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            d = resp.json()
            person = d.get("people", [{}])[0]
            return person.get("batSide", {}).get("code", "UNK"), person.get("pitchHand", {}).get("code", "UNK")
    except Exception:
        pass
    return "UNK", "UNK"

def fetch_schedule_df(start_date, end_date):
    rows = []
    current = start_date
    while current <= end_date:
        url = f"https://statsapi.mlb.com/api/v1/schedule?date={current.strftime('%Y-%m-%d')}&sportId=1"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        for date_data in data.get("dates", []):
            for g in date_data.get("games", []):
                rows.append({
                    "game_pk": g.get("gamePk"),
                    "venue": g.get("venue", {}).get("name"),
                    "city": g.get("venue", {}).get("city", None),
                    "game_time_utc": g.get("gameDate"),
                    "home_team": g.get("teams", {}).get("home", {}).get("team", {}).get("name"),
                    "away_team": g.get("teams", {}).get("away", {}).get("team", {}).get("name"),
                })
        current += timedelta(days=1)
    df = pd.DataFrame(rows)
    df["city"] = df.apply(lambda x: x["city"] or VENUE_TO_CITY.get(x["venue"], "UNK"), axis=1)
    def utc_to_pst(utc_str):
        try:
            dt = datetime.strptime(utc_str, "%Y-%m-%dT%H:%M:%SZ")
            dt_pst = dt - timedelta(hours=8)
            return dt_pst.strftime("%H:%M")
        except Exception:
            return ""
    df["game_time_pst"] = df["game_time_utc"].map(utc_to_pst)
    return df

def fetch_weather(city, date, game_time, api_key=API_KEY):
    try:
        url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={date}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        hours = data['forecast']['forecastday'][0]['hour']
        game_hour = int(game_time.split(":")[0]) if game_time else 14
        weather_hour = min(hours, key=lambda h: abs(int(h['time'].split(' ')[1].split(':')[0]) - game_hour))
        return {
            "temp_f": weather_hour.get("temp_f"),
            "wind_mph": weather_hour.get("wind_mph"),
            "wind_dir": weather_hour.get("wind_dir"),
            "humidity": weather_hour.get("humidity"),
            "condition": weather_hour.get("condition", {}).get("text")
        }
    except Exception:
        return {"temp_f": None, "wind_mph": None, "wind_dir": None, "humidity": None, "condition": None}

# ---------- MAIN APP ----------
st.title("MLB HR Backtest + ML + Leaderboard Bot (Cool Bot 2, Batted Ball, Names, Weather, Handedness)")

mode = st.selectbox("Mode", [
    "Build CSV (Backtest Builder)",
    "Train Model",
    "Predict Today",
    "Leaderboard"
])

MODEL_PATH = "hr_model_lgb.pkl"
HIST_CSV_PATH = "mlb_hr_backtest.csv"
TODAY_CSV_PATH = "today_games.csv"

if mode == "Build CSV (Backtest Builder)":
    st.header("Build Historical Dataset (Backtest Builder)")
    start = st.date_input("Start Date", datetime(2025, 5, 12))
    end = st.date_input("End Date", datetime(2025, 5, 19))
    if st.button("Build Dataset"):
        st.info("Fetching Statcast and weather data (this may take several minutes for large ranges)...")
        schedule_df = fetch_schedule_df(start, end)
        all_rows = []
        date_list = pd.date_range(start, end)
        total_events = 0
        # Pre-scan for progress
        for date in date_list:
            url = f"https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfPT=&hfAB=&hfGT=R%7CPO%7C&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&hfPull=&hfC=&hfSea={date.year}&hfSit=&player_type=batter&hfOuts=&opponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt={date.strftime('%Y-%m-%d')}&game_date_lt={date.strftime('%Y-%m-%d')}&team=&position=&hfRO=&home_road=&hfFlag=&metric_1=&metric_2=&hfInn=&min_pitches=0&min_results=0&chk_statcast_data=on"
            try:
                df = pd.read_csv(url)
                df = df[df["type"] == "X"] # Batted ball events only
                all_rows.append(df)
                total_events += len(df)
            except Exception:
                pass
        if not all_rows:
            st.error("No Statcast data fetched for given date range.")
            st.stop()
        df = pd.concat(all_rows, ignore_index=True)
        merged = df.merge(schedule_df, left_on="game_pk", right_on="game_pk", how="left")
        merged["city"] = merged["city"].fillna(merged["venue"].map(VENUE_TO_CITY)).fillna("UNK")
        merged["game_time_pst"] = merged["game_time_pst"].fillna("")
        merged["home_team"] = merged["home_team"].fillna("UNK")
        merged["away_team"] = merged["away_team"].fillna("UNK")
        # Handedness by MLB API player_id
        batter_hands, pitcher_hands = [], []
        progress = st.progress(0, text="Fetching handedness (player_id)...")
        for idx, row in merged.iterrows():
            b_bat, _ = get_hand_from_mlb(row["batter"]) if pd.notnull(row["batter"]) else ("UNK", "UNK")
            _, p_throw = get_hand_from_mlb(row["pitcher"]) if pd.notnull(row["pitcher"]) else ("UNK", "UNK")
            batter_hands.append(b_bat)
            pitcher_hands.append(p_throw)
            if idx % 50 == 0 or idx == len(merged) - 1:
                pct = int(100 * (idx + 1) / len(merged))
                progress.progress((idx + 1) / len(merged), text=f"Handedness: {pct}% ({idx + 1} of {len(merged)})")
        merged["batter_hand"] = batter_hands
        merged["pitcher_hand"] = pitcher_hands
        # Weather (game time)
        weather_list = []
        progress = st.progress(0, text="Fetching weather...")
        for idx, row in merged.iterrows():
            if pd.notnull(row["city"]) and row["game_date"] and row["game_time_pst"]:
                w = fetch_weather(row["city"], row["game_date"], row["game_time_pst"])
            else:
                w = {"temp_f": None, "wind_mph": None, "wind_dir": None, "humidity": None, "condition": None}
            weather_list.append(w)
            if idx % 50 == 0 or idx == len(merged) - 1:
                pct = int(100 * (idx + 1) / len(merged))
                progress.progress((idx + 1) / len(merged), text=f"Weather: {pct}% ({idx + 1} of {len(merged)})")
        merged["temp_f"] = [w["temp_f"] for w in weather_list]
        merged["wind_mph"] = [w["wind_mph"] for w in weather_list]
        merged["wind_dir"] = [w["wind_dir"] for w in weather_list]
        merged["humidity"] = [w["humidity"] for w in weather_list]
        merged["condition"] = [w["condition"] for w in weather_list]
        # Columns for export
        cols = [
            "game_date","game_pk","home_team","away_team","venue","city","game_time_pst",
            "batter","batter_name","batter_hand","pitcher","pitcher_name","pitcher_hand",
            "events","bb_type","launch_speed","launch_angle","estimated_ba_using_speedangle",
            "estimated_woba_using_speedangle","hc_x","hc_y","park_factor","temp_f","wind_mph",
            "wind_dir","humidity","condition","is_hr"
        ]
        present_cols = [c for c in cols if c in merged.columns]
        missing = [c for c in cols if c not in merged.columns]
        st.write("Available columns:", merged.columns.tolist())
        st.write("Missing columns:", missing)
        df_export = merged[present_cols]
        df_export.to_csv(HIST_CSV_PATH, index=False)
        st.success("Export ready! Download your full backtest CSV below:")
        st.dataframe(df_export.head(50))
        st.download_button("Download CSV", df_export.to_csv(index=False).encode(), "mlb_hr_backtest.csv")
        st.stop()

if mode == "Train Model":
    st.header("Train ML Model on Historical Data")
    if not os.path.exists(HIST_CSV_PATH):
        st.warning("Please build historical dataset first.")
        st.stop()
    df = pd.read_csv(HIST_CSV_PATH)
    df = df[df["events"].notnull() & df["launch_speed"].notnull() & df["launch_angle"].notnull()]
    # Binary label: is_hr (1 if event == home_run)
    df["is_hr"] = (df["events"] == "home_run").astype(int)
    # Features for ML (customize as you like)
    features = [
        "launch_speed","launch_angle","estimated_ba_using_speedangle",
        "estimated_woba_using_speedangle","temp_f","wind_mph","humidity"
    ]
    X = df[features].fillna(0)
    y = df["is_hr"]
    dtrain = lgb.Dataset(X, y)
    params = dict(objective="binary", metric="auc", verbosity=-1, random_state=42)
    st.info("Training LightGBM model (may take a minute)...")
    model = lgb.train(params, dtrain, num_boost_round=200)
    joblib.dump(model, MODEL_PATH)
    st.success("Model trained and saved! (hr_model_lgb.pkl)")

if mode == "Predict Today":
    st.header("Predict HRs for Today's Games")
    st.info("Fetching today's games...")
    today = datetime.now().date()
    schedule_df = fetch_schedule_df(today, today)
    all_rows = []
    url = f"https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfPT=&hfAB=&hfGT=R%7CPO%7C&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&hfPull=&hfC=&hfSea={today.year}&hfSit=&player_type=batter&hfOuts=&opponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt={today.strftime('%Y-%m-%d')}&game_date_lt={today.strftime('%Y-%m-%d')}&team=&position=&hfRO=&home_road=&hfFlag=&metric_1=&metric_2=&hfInn=&min_pitches=0&min_results=0&chk_statcast_data=on"
    try:
        df = pd.read_csv(url)
        df = df[df["type"] == "X"]
        all_rows.append(df)
    except Exception:
        st.error("No Statcast events found for today.")
        st.stop()
    if not all_rows:
        st.error("No Statcast data for today.")
        st.stop()
    df = pd.concat(all_rows, ignore_index=True)
    merged = df.merge(schedule_df, left_on="game_pk", right_on="game_pk", how="left")
    merged["city"] = merged["city"].fillna(merged["venue"].map(VENUE_TO_CITY)).fillna("UNK")
    merged["game_time_pst"] = merged["game_time_pst"].fillna("")
    merged["home_team"] = merged["home_team"].fillna("UNK")
    merged["away_team"] = merged["away_team"].fillna("UNK")
    batter_hands, pitcher_hands = [], []
    progress = st.progress(0, text="Fetching handedness...")
    for idx, row in merged.iterrows():
        b_bat, _ = get_hand_from_mlb(row["batter"]) if pd.notnull(row["batter"]) else ("UNK", "UNK")
        _, p_throw = get_hand_from_mlb(row["pitcher"]) if pd.notnull(row["pitcher"]) else ("UNK", "UNK")
        batter_hands.append(b_bat)
        pitcher_hands.append(p_throw)
        if idx % 20 == 0 or idx == len(merged) - 1:
            pct = int(100 * (idx + 1) / len(merged))
            progress.progress((idx + 1) / len(merged), text=f"Handedness: {pct}% ({idx + 1} of {len(merged)})")
    merged["batter_hand"] = batter_hands
    merged["pitcher_hand"] = pitcher_hands
    # Weather (optional, for current games)
    weather_list = []
    for idx, row in merged.iterrows():
        if pd.notnull(row["city"]) and row["game_date"] and row["game_time_pst"]:
            w = fetch_weather(row["city"], row["game_date"], row["game_time_pst"])
        else:
            w = {"temp_f": None, "wind_mph": None, "wind_dir": None, "humidity": None, "condition": None}
        weather_list.append(w)
    merged["temp_f"] = [w["temp_f"] for w in weather_list]
    merged["wind_mph"] = [w["wind_mph"] for w in weather_list]
    merged["wind_dir"] = [w["wind_dir"] for w in weather_list]
    merged["humidity"] = [w["humidity"] for w in weather_list]
    merged["condition"] = [w["condition"] for w in weather_list]
    # Load model and predict
    if not os.path.exists(MODEL_PATH):
        st.warning("Train the model first.")
        st.stop()
    model = joblib.load(MODEL_PATH)
    features = [
        "launch_speed","launch_angle","estimated_ba_using_speedangle",
        "estimated_woba_using_speedangle","temp_f","wind_mph","humidity"
    ]
    pred_df = merged[features].fillna(0)
    preds = model.predict(pred_df)
    merged["HR_Prob"] = preds
    st.success("Predictions ready!")
    merged = merged.sort_values("HR_Prob", ascending=False)
    st.dataframe(merged[["game_date","home_team","away_team","venue","batter_name","pitcher_name","batter_hand","pitcher_hand","HR_Prob","events","launch_speed","launch_angle"]].head(50))
    st.download_button("Download Today Predictions CSV", merged.to_csv(index=False).encode(), "today_predictions.csv")

if mode == "Leaderboard":
    st.header("HR Leaderboard")
    # Use either last predictions or historical
    files = [HIST_CSV_PATH, TODAY_CSV_PATH]
    use_file = st.selectbox("Choose dataset", files)
    if not os.path.exists(use_file):
        st.warning("Dataset not found. Please build CSV or predict today first.")
        st.stop()
    df = pd.read_csv(use_file)
    # If model predictions available, show leaderboard
    pred_col = "HR_Prob" if "HR_Prob" in df.columns else None
    leaderboard_cols = [
        "game_date","home_team","away_team","venue","batter_name","pitcher_name",
        "batter_hand","pitcher_hand","events","launch_speed","launch_angle",
        "estimated_ba_using_speedangle","estimated_woba_using_speedangle",
        "temp_f","wind_mph","wind_dir","humidity","condition"
    ]
    if pred_col:
        leaderboard_cols.insert(8, pred_col)
        df = df.sort_values(pred_col, ascending=False)
        st.subheader("Top Predicted HR Events (by ML Probability)")
        st.dataframe(df[leaderboard_cols].head(50))
    else:
        df = df[df["events"] == "home_run"]
        st.subheader("Historical Home Runs (sorted by EV)")
        df = df.sort_values("launch_speed", ascending=False)
        st.dataframe(df[leaderboard_cols].head(50))
    st.download_button("Download Leaderboard CSV", df.to_csv(index=False).encode(), "leaderboard.csv")
