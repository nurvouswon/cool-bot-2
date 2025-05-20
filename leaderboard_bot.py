import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from io import BytesIO
import joblib

# --- Park Factors & City Map (expand as needed)
PARK_FACTORS = {
    "Yankee Stadium": 1.19, "Fenway Park": 0.97, "Tropicana Field": 0.85,
    "Camden Yards": 1.13, "Rogers Centre": 1.10, "Comerica Park": 0.96,
    "Progressive Field": 1.01, "Target Field": 1.04, "Kauffman Stadium": 0.98,
    "Guaranteed Rate Field": 1.18, "Angel Stadium": 1.05, "Minute Maid Park": 1.06,
    "Oakland Coliseum": 0.82, "T-Mobile Park": 0.86, "Globe Life Field": 1.00,
    "Dodger Stadium": 1.10, "Chase Field": 1.06, "Coors Field": 1.30,
    "Oracle Park": 0.82, "Wrigley Field": 1.12, "Great American Ball Park": 1.26,
    "American Family Field": 1.17, "PNC Park": 0.87, "Busch Stadium": 0.87,
    "Truist Park": 1.06, "LoanDepot Park": 0.86, "Citi Field": 1.05,
    "Nationals Park": 1.05, "Petco Park": 0.85, "Citizens Bank Park": 1.19
}
PARK_CITY = {
    "Yankee Stadium": "Bronx", "Fenway Park": "Boston", "Tropicana Field": "St. Petersburg",
    "Camden Yards": "Baltimore", "Rogers Centre": "Toronto", "Comerica Park": "Detroit",
    "Progressive Field": "Cleveland", "Target Field": "Minneapolis", "Kauffman Stadium": "Kansas City",
    "Guaranteed Rate Field": "Chicago", "Angel Stadium": "Anaheim", "Minute Maid Park": "Houston",
    "Oakland Coliseum": "Oakland", "T-Mobile Park": "Seattle", "Globe Life Field": "Arlington",
    "Dodger Stadium": "Los Angeles", "Chase Field": "Phoenix", "Coors Field": "Denver",
    "Oracle Park": "San Francisco", "Wrigley Field": "Chicago", "Great American Ball Park": "Cincinnati",
    "American Family Field": "Milwaukee", "PNC Park": "Pittsburgh", "Busch Stadium": "St. Louis",
    "Truist Park": "Atlanta", "LoanDepot Park": "Miami", "Citi Field": "Queens",
    "Nationals Park": "Washington", "Petco Park": "San Diego", "Citizens Bank Park": "Philadelphia"
}
API_KEY = st.secrets["general"]["weather_api"]

def get_stadium_name(game_pk):
    try:
        url = f"https://statsapi.mlb.com/api/v1.1/game/{int(game_pk)}/feed/live"
        resp = requests.get(url)
        data = resp.json()
        return data['gameData']['venue']['name']
    except Exception:
        return "TBD"

def get_game_time(game_pk):
    try:
        url = f"https://statsapi.mlb.com/api/v1.1/game/{int(game_pk)}/feed/live"
        resp = requests.get(url)
        data = resp.json()
        dt = data['gameData']['datetime']['dateTime']
        game_time = dt.split('T')[1][:5]
        return game_time
    except Exception:
        return "14:00"

def get_weather(city, date, time_str, api_key):
    try:
        url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={city}&dt={date}"
        resp = requests.get(url)
        data = resp.json()
        if "forecast" in data:
            hours = data['forecast']['forecastday'][0]['hour']
            game_hour = int(time_str.split(":")[0]) if time_str else 14
            hour_data = min(hours, key=lambda h: abs(int(h['time'].split(' ')[1].split(':')[0]) - game_hour))
            return {
                "temp_f": hour_data.get("temp_f"),
                "wind_mph": hour_data.get("wind_mph"),
                "wind_dir": hour_data.get("wind_dir"),
                "humidity": hour_data.get("humidity"),
                "condition": hour_data.get("condition", {}).get("text", "")
            }
    except Exception:
        pass
    return {
        "temp_f": None, "wind_mph": None, "wind_dir": None, "humidity": None, "condition": None
    }

def get_hand_from_mlb(player_id):
    """Fetch handedness from MLB API for a player_id (bats, throws)."""
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{int(player_id)}"
        resp = requests.get(url)
        p = resp.json()['people'][0]
        bats = p.get("batSide", {}).get("code", "UNK")
        throws = p.get("pitchHand", {}).get("code", "UNK")
        return bats, throws
    except Exception:
        return "UNK", "UNK"

def fetch_statcast_data(start_date, end_date):
    from pybaseball import statcast
    df = statcast(start_date, end_date)
    if df.empty:
        return df
    cols_needed = [
        'game_date', 'batter', 'pitcher', 'home_team', 'away_team', 'game_pk',
        'events', 'launch_angle', 'launch_speed'
    ]
    for c in cols_needed:
        if c not in df.columns:
            df[c] = None
    return df[cols_needed]

def build_backtest_csv(start, end):
    df_all = fetch_statcast_data(start, end)
    if df_all.empty:
        st.error("No Statcast data fetched for given date range.")
        return
    stadiums, park_factors, is_hr, weathers, cities, times = [], [], [], [], [], []
    batter_hands, pitcher_hands = [], []
    for idx, row in df_all.iterrows():
        stadium = get_stadium_name(row['game_pk'])
        stadiums.append(stadium)
        pf = PARK_FACTORS.get(stadium, 1.0)
        park_factors.append(pf)
        city = PARK_CITY.get(stadium, "TBD")
        cities.append(city)
        time_str = get_game_time(row['game_pk'])
        times.append(time_str)
        weather = get_weather(city, str(row['game_date']), time_str, API_KEY)
        weathers.append(weather)
        is_hr.append(1 if str(row['events']) == "home_run" else 0)
        # Handedness fetch
        bats, _ = get_hand_from_mlb(row['batter'])
        _, throws = get_hand_from_mlb(row['pitcher'])
        batter_hands.append(bats)
        pitcher_hands.append(throws)
        if idx % 200 == 0:
            st.write(f"Processed {idx} rows...")
    df_all['stadium'] = stadiums
    df_all['park_factor'] = park_factors
    df_all['city'] = cities
    df_all['game_time'] = times
    df_all['is_hr'] = is_hr
    df_all['batter_hand'] = batter_hands
    df_all['pitcher_hand'] = pitcher_hands
    df_all['temp_f'] = [w['temp_f'] for w in weathers]
    df_all['wind_mph'] = [w['wind_mph'] for w in weathers]
    df_all['wind_dir'] = [w['wind_dir'] for w in weathers]
    df_all['humidity'] = [w['humidity'] for w in weathers]
    df_all['condition'] = [w['condition'] for w in weathers]
    st.dataframe(df_all.head(20))
    csv = df_all.to_csv(index=False).encode()
    st.download_button("Download Backtest CSV", csv, "mlb_hr_backtest.csv")
    st.success("Done! Your historical backtest CSV is ready.")

def encode_hand(hand):
    # One-hot encoding for handedness: 'R'->0, 'L'->1, 'S'->2, 'UNK'->-1
    return {'R':0, 'L':1, 'S':2, 'UNK':-1}.get(str(hand).upper(), -1)

def train_model(uploaded_file):
    st.write("Training model (RandomForest)...")
    df = pd.read_csv(uploaded_file)
    # Feature columns (add handedness as encoded)
    for col in ['batter_hand', 'pitcher_hand']:
        if col in df.columns:
            df[col + "_num"] = df[col].map(encode_hand)
    feature_cols = ['launch_angle', 'launch_speed', 'park_factor', 'temp_f', 'wind_mph', 'humidity',
                    'batter_hand_num', 'pitcher_hand_num']
    feature_cols = [col for col in feature_cols if col in df.columns]
    if not feature_cols or 'is_hr' not in df.columns:
        st.error("Missing required columns for training!")
        return
    df = df.dropna(subset=feature_cols + ['is_hr'])
    X = df[feature_cols]
    y = df['is_hr']
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    acc = model.score(X, y)
    st.success(f"Training accuracy: {acc:.3f}")
    buffer = BytesIO()
    joblib.dump(model, buffer)
    st.download_button("Download Trained Model", buffer.getvalue(), "mlb_hr_rf_model.joblib")
    st.write("Feature importance:", dict(zip(feature_cols, model.feature_importances_)))

def predict_today(uploaded_file, model_file):
    df = pd.read_csv(uploaded_file)
    # Encode handedness
    for col in ['batter_hand', 'pitcher_hand']:
        if col in df.columns:
            df[col + "_num"] = df[col].map(encode_hand)
    feature_cols = ['launch_angle', 'launch_speed', 'park_factor', 'temp_f', 'wind_mph', 'humidity',
                    'batter_hand_num', 'pitcher_hand_num']
    feature_cols = [col for col in feature_cols if col in df.columns]
    if not feature_cols:
        st.error("Missing required columns in uploaded file!")
        return
    if model_file is None:
        st.error("Please upload a trained model file!")
        return
    buffer = BytesIO(model_file.read())
    model = joblib.load(buffer)
    df['hr_prob'] = model.predict_proba(df[feature_cols])[:, 1]
    st.dataframe(df.sort_values('hr_prob', ascending=False).head(20))
    csv = df.to_csv(index=False).encode()
    st.download_button("Download Predictions", csv, "mlb_hr_predictions.csv")
    st.success("Prediction complete!")

# --- Streamlit UI
st.title("MLB HR Backtest + ML + Leaderboard Bot (2025-ready, Handedness)")

mode = st.selectbox("Mode", ["Build CSV", "Train Model", "Predict Today"])

if mode == "Build CSV":
    st.header("Build Historical Dataset (Backtest Builder)")
    start = st.date_input("Start Date", datetime.now() - timedelta(days=7))
    end = st.date_input("End Date", datetime.now())
    if st.button("Run Backtest Builder"):
        st.info("Fetching Statcast and weather data (can take several minutes for large ranges)...")
        build_backtest_csv(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

elif mode == "Train Model":
    st.header("Train ML Model on HR Backtest CSV")
    uploaded_file = st.file_uploader("Upload CSV (from Build CSV step)", type=["csv"])
    if uploaded_file is not None:
        train_model(uploaded_file)

elif mode == "Predict Today":
    st.header("Predict HR Outcomes for Today's Matchups")
    uploaded_file = st.file_uploader("Upload Today's Data CSV (with features)", type=["csv"])
    model_file = st.file_uploader("Upload Trained Model (.joblib)", type=["joblib"])
    if uploaded_file is not None and model_file is not None:
        predict_today(uploaded_file, model_file)
