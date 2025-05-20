import streamlit as st
st.write("DEBUG secrets:", st.secrets)
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from pybaseball import statcast, playerid_reverse_lookup, playerid_lookup, statcast_batter, statcast_pitcher
from pybaseball.lahman import people
import requests, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import unicodedata

# ====== SETTINGS ======
API_KEY = st.secrets["general"]["weather_api"]  # <-- uses .streamlit/secrets.toml

# Update this with any other park factors you use
PARK_FACTORS = {
    "Coors Field": 1.30,
    "Great American Ball Park": 1.26,
    "Yankee Stadium": 1.19,
    "Fenway Park": 0.97,
    # Add more as needed...
}
# All feature columns for ML
FEATURES = ['launch_speed','launch_angle','ParkFactor','Temp','Wind','Bats_R','Throws_R']

# ====== HELPERS ======

def normalize_name(name):
    if not isinstance(name, str): return ""
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = name.lower().replace('.', '').replace('-', ' ').replace("’", "'").strip()
    return ' '.join(name.split())

@st.cache_data(show_spinner=False)
def get_weather(city, date, hour):
    try:
        url = f"http://api.weatherapi.com/v1/history.json?key={API_KEY}&q={city}&dt={date}"
        r = requests.get(url)
        data = r.json()
        hours = data['forecast']['forecastday'][0]['hour']
        target_hour = min(hours, key=lambda h: abs(int(h['time'].split(' ')[1].split(':')[0]) - hour))
        return {
            "Temp": target_hour.get('temp_f'),
            "Wind": target_hour.get('wind_mph'),
            "Humidity": target_hour.get('humidity'),
            "Condition": target_hour.get('condition', {}).get('text')
        }
    except:
        return {"Temp": None, "Wind": None, "Humidity": None, "Condition": None}

@st.cache_data(show_spinner=False)
def get_game_time(game_pk):
    try:
        url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
        r = requests.get(url)
        data = r.json()
        datetime_str = data['gameData']['datetime']['dateTime']
        return datetime.fromisoformat(datetime_str[:-1])
    except:
        return None

@st.cache_data(show_spinner=False)
def get_handedness(name):
    clean_name = normalize_name(name)
    parts = clean_name.split()
    if len(parts) >= 2:
        first, last = parts[0], parts[-1]
    else:
        first, last = clean_name, ""
    try:
        lookup = playerid_lookup(last.capitalize(), first.capitalize())
        if not lookup.empty:
            bats = lookup.iloc[0].get('bats')
            throws = lookup.iloc[0].get('throws')
            if pd.notnull(bats) and pd.notnull(throws): return bats, throws
    except: pass
    try:
        df = people()
        df['nname'] = (df['name_first'].fillna('') + ' ' + df['name_last'].fillna('')).map(normalize_name)
        match = df[df['nname'] == clean_name]
        if not match.empty:
            return match.iloc[0].get('bats'), match.iloc[0].get('throws')
    except: pass
    return "UNK", "UNK"

@st.cache_data(show_spinner=False)
def get_rolling_stats(pid, days, is_batter=True):
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    try:
        df = statcast_batter(start, end, pid) if is_batter else statcast_pitcher(start, end, pid)
        df = df[df['launch_speed'].notnull() & df['launch_angle'].notnull()]
        if df.empty: return None, None
        barrels = df[(df['launch_speed'] > 95) & (df['launch_angle'].between(20, 35))].shape[0]
        total = len(df)
        barrel_rate = round(barrels / total, 3) if total > 0 else 0
        ev = round(df['launch_speed'].mean(), 1) if total > 0 else None
        return barrel_rate, ev
    except:
        return None, None

# ====== STREAMLIT UI ======

st.title("MLB HR Backtest + ML + Leaderboard Bot")

mode = st.radio("Mode", ["Build CSV", "Train Model", "Predict Today"])

if mode == "Build CSV":
    st.header("Build Historical Dataset (Backtest Builder)")
    start_date = st.date_input("Start Date", datetime(2025, 4, 1))
    end_date = st.date_input("End Date", datetime(2025, 4, 7))
    run_button = st.button("Run Backtest Builder")
    if run_button:
        st.info("Fetching Statcast data (can take several minutes for large ranges)...")
        all_data = []
        loop_dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
        for date in loop_dates:
            day = date.strftime('%Y-%m-%d')
            try:
                df = statcast(day, day)
                df = df[df['events'].notnull() & df['batter'].notnull() & df['pitcher'].notnull()]
                df['Hit_HR'] = (df['events'] == 'home_run').astype(int)
                df = df[['game_date','batter','pitcher','home_team','away_team','stadium','game_pk',
                         'Hit_HR', 'launch_angle', 'launch_speed']].copy()
                all_data.append(df)
            except Exception as e:
                st.warning(f"{day} error: {e}")
            time.sleep(1)
        if all_data:
            df_all = pd.concat(all_data, ignore_index=True)
            df_all.rename(columns={'game_date': 'Date'}, inplace=True)
            st.info("Mapping player names and handedness...")
            batters = playerid_reverse_lookup(df_all['batter'].unique(), key_type='mlbam')
            pitchers = playerid_reverse_lookup(df_all['pitcher'].unique(), key_type='mlbam')
            bdict = dict(zip(batters['key_mlbam'], batters['name_first'] + ' ' + batters['name_last']))
            pdict = dict(zip(pitchers['key_mlbam'], pitchers['name_first'] + ' ' + pitchers['name_last']))
            df_all['Batter'] = df_all['batter'].map(bdict)
            df_all['Pitcher'] = df_all['pitcher'].map(pdict)
            weather_rows, park_rows, time_rows, handed_b, handed_p, barrels, evs = [], [], [], [], [], [], []
            for _, row in df_all.iterrows():
                city, date, game_pk = row['home_team'], row['Date'], row['game_pk']
                park_factor = PARK_FACTORS.get(row['stadium'], 1.0)
                game_time = get_game_time(game_pk)
                hour = game_time.hour if game_time else 13
                weather = get_weather(city, date, hour)
                park_rows.append({"ParkFactor": park_factor})
                weather_rows.append(weather)
                time_rows.append({"GameTime": game_time.strftime('%H:%M') if game_time else "TBD"})
                b, _ = get_handedness(row['Batter'])
                _, p = get_handedness(row['Pitcher'])
                handed_b.append(b or 'UNK')
                handed_p.append(p or 'UNK')
                pid = row['batter']
                pid2 = row['pitcher']
                bbr, bev = get_rolling_stats(pid, 7, True)
                pbr, pev = get_rolling_stats(pid2, 7, False)
                barrels.append({'BarrelRate_Batter_7': bbr, 'EV_Batter_7': bev, 'BarrelRate_Pitcher_7': pbr, 'EV_Pitcher_7': pev})
            df_all = pd.concat([
                df_all.reset_index(drop=True), 
                pd.DataFrame(weather_rows), 
                pd.DataFrame(park_rows), 
                pd.DataFrame(time_rows), 
                pd.DataFrame(barrels)
            ], axis=1)
            df_all['BatterHandedness'] = handed_b
            df_all['PitcherHandedness'] = handed_p
            st.success("Done! Preview below.")
            st.dataframe(df_all.head(50))
            csv = df_all.to_csv(index=False).encode()
            st.download_button("Download CSV", csv, "hr_backtest.csv")
        else:
            st.warning("No Statcast data fetched for given date range.")

elif mode == "Train Model":
    st.header("Train Model")
    uploaded_file = st.file_uploader("Upload Clean CSV for Training", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        # Drop rows missing core features
        df.dropna(subset=['launch_speed','launch_angle','ParkFactor','Temp','Wind'], inplace=True)
        df['Bats_R'] = (df['BatterHandedness'] == 'R').astype(int)
        df['Throws_R'] = (df['PitcherHandedness'] == 'R').astype(int)
        X = df[FEATURES]
        y = df['Hit_HR']
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        st.code(classification_report(y_test, preds))
        # Save model
        joblib.dump(model, 'hr_model.pkl')
        st.success("Model trained and saved as hr_model.pkl. Ready to predict today’s matchups!")

elif mode == "Predict Today":
    st.header("Predict Today's Matchups")
    uploaded_daily = st.file_uploader("Upload Today's Matchup CSV", type="csv")
    model = None
    try:
        model = joblib.load('hr_model.pkl')
    except:
        st.warning("Trained model (hr_model.pkl) not found. Train model first.")
    if uploaded_daily and model is not None:
        df_leaderboard = pd.read_csv(uploaded_daily)
        # Ensure features are present
        df_leaderboard['Bats_R'] = (df_leaderboard['BatterHandedness'] == 'R').astype(int)
        df_leaderboard['Throws_R'] = (df_leaderboard['PitcherHandedness'] == 'R').astype(int)
        missing = [col for col in FEATURES if col not in df_leaderboard.columns]
        if missing:
            st.warning(f"Missing columns in upload: {missing}")
        else:
            X_new = df_leaderboard[FEATURES]
            df_leaderboard['ML_HR_Prob'] = model.predict_proba(X_new)[:, 1]
            if 'HR_Score' in df_leaderboard.columns:
                df_leaderboard['BlendedScore'] = 0.5 * df_leaderboard['HR_Score'] + 0.5 * df_leaderboard['ML_HR_Prob']
            st.dataframe(df_leaderboard.sort_values('ML_HR_Prob', ascending=False).head(20))
            st.download_button("Download with ML HR Prob", df_leaderboard.to_csv(index=False).encode(), "today_predicted.csv")
