import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybaseball import statcast
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

st.set_page_config(layout="wide")
st.title("MLB Statcast Downloader & Analyzer — Rolling Features & HR Context Exports")

#### ============ SECTION 1: DATA DOWNLOAD ==============

st.header("📥 Download Statcast Batted Ball Events")
num_days = st.slider("Number of days to fetch", 7, 60, 30)
download_btn = st.button("Download Most Recent Data")

if download_btn:
    today = datetime.now().date()
    start_date = today - timedelta(days=num_days)
    with st.spinner(f"Downloading Statcast data from {start_date} to {today}..."):
        df = statcast(start_date.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"))
        df = df[df['type'] == 'X']  # Only in-play
    st.success(f"Fetched {len(df)} batted ball events!")
    csv = df.to_csv(index=False).encode()
    st.download_button("Download as CSV", csv, file_name=f"statcast_{num_days}days.csv")
    st.write(df.head())

#### ============ SECTION 2: UPLOAD FOR ANALYSIS ==============

st.header("📤 Upload Statcast Batted Ball CSV for Rolling Feature Analysis")

csv_file = st.file_uploader("Upload Statcast Batted Ball Events CSV", type=["csv"])
if not csv_file:
    st.info("Upload a CSV or download one above to begin.")
    st.stop()

df = pd.read_csv(csv_file)
df.columns = [c.lower().replace(" ", "_") for c in df.columns]

if 'game_date' in df.columns:
    df['game_date'] = pd.to_datetime(df['game_date'])
else:
    st.error("Missing 'game_date' column in CSV.")
    st.stop()

# --------- Basic Cleaning and HR Tagging -----------
hr_events = ['home_run', 'home run', 'hr']
if 'events' in df.columns:
    df['hr_outcome'] = df['events'].str.lower().isin(hr_events).astype(int)
elif 'result' in df.columns:
    df['hr_outcome'] = df['result'].str.lower().isin(hr_events).astype(int)
else:
    df['hr_outcome'] = 0

df['batter_id'] = df.get('batter', df.get('batter_id', np.nan))
df['pitcher_id'] = df.get('pitcher', df.get('pitcher_id', np.nan))
df['exit_velocity'] = df.get('launch_speed', df.get('exit_velocity', np.nan))
df['launch_angle'] = df.get('launch_angle', np.nan)
df['release_spin_rate'] = df.get('release_spin_rate', np.nan)
df['release_speed'] = df.get('release_speed', np.nan)
df['p_throws'] = df.get('p_throws', df.get('pitcher_hand', 'NA')).fillna('NA')
df['stand'] = df.get('stand', df.get('batter_hand', 'NA')).fillna('NA')

# xwOBA: Use statcast "estimated_woba_using_speedangle" or fallback to "woba_value"
if 'estimated_woba_using_speedangle' in df.columns:
    df['xwoba'] = df['estimated_woba_using_speedangle']
elif 'woba_value' in df.columns:
    df['xwoba'] = df['woba_value']
else:
    df['xwoba'] = np.nan

# xBA: Use "estimated_ba_using_speedangle" if available
df['xba'] = df['estimated_ba_using_speedangle'] if 'estimated_ba_using_speedangle' in df.columns else np.nan

#### ------------- ROLLING FEATURE GENERATION -------------

windows = [3, 5, 7, 14, 30]

def rolling_batter_features(group):
    group = group.sort_values('game_date')
    feats = {}
    for w in windows:
        lastN = group.tail(w)
        pa = len(lastN)
        barrels = lastN[(lastN['exit_velocity'] > 95) & (lastN['launch_angle'].between(20, 35))].shape[0]
        sweet = lastN[lastN['launch_angle'].between(8, 32)].shape[0]
        hard = lastN[lastN['exit_velocity'] >= 95].shape[0]
        slg = np.nansum([
            (lastN['events'] == 'single').sum(),
            2 * (lastN['events'] == 'double').sum(),
            3 * (lastN['events'] == 'triple').sum(),
            4 * (lastN['events'] == 'home_run').sum()
        ])
        ab = sum((lastN['events'] == x).sum() for x in ['single', 'double', 'triple', 'home_run', 'field_out', 'force_out', 'other_out'])
        # xSLG and xISO
        single = (lastN['xba'] >= 0.5) & (lastN['launch_angle'] < 15)
        double = (lastN['xba'] >= 0.5) & (lastN['launch_angle'].between(15, 30))
        triple = (lastN['xba'] >= 0.5) & (lastN['launch_angle'].between(30, 40))
        hr = (lastN['launch_angle'] > 35) & (lastN['exit_velocity'] > 100)
        xsingle = single.sum()
        xdouble = double.sum()
        xtriple = triple.sum()
        xhr = hr.sum()
        xab = xsingle + xdouble + xtriple + xhr
        xslg = (1 * xsingle + 2 * xdouble + 3 * xtriple + 4 * xhr) / xab if xab else np.nan
        xba = lastN['xba'].mean() if 'xba' in lastN.columns and not lastN['xba'].isnull().all() else np.nan
        xiso = (xslg - xba) if xslg is not np.nan and xba is not np.nan else np.nan

        feats[f'B_BarrelRate_{w}'] = barrels / pa if pa else np.nan
        feats[f'B_EV_{w}'] = lastN['exit_velocity'].mean() if pa else np.nan
        feats[f'B_SLG_{w}'] = slg / ab if ab else np.nan
        feats[f'B_xSLG_{w}'] = xslg
        feats[f'B_xISO_{w}'] = xiso
        feats[f'B_xwoba_{w}'] = lastN['xwoba'].mean() if pa else np.nan
        feats[f'B_sweet_spot_pct_{w}'] = sweet / pa if pa else np.nan
        feats[f'B_hardhit_pct_{w}'] = hard / pa if pa else np.nan
    return pd.Series(feats)

def rolling_pitcher_features(group):
    group = group.sort_values('game_date')
    feats = {}
    for w in windows:
        lastN = group.tail(w)
        total = len(lastN)
        barrels = lastN[(lastN['exit_velocity'] > 95) & (lastN['launch_angle'].between(20, 35))].shape[0]
        sweet = lastN[lastN['launch_angle'].between(8, 32)].shape[0]
        hard = lastN[lastN['exit_velocity'] >= 95].shape[0]
        slg = np.nansum([
            (lastN['events'] == 'single').sum(),
            2 * (lastN['events'] == 'double').sum(),
            3 * (lastN['events'] == 'triple').sum(),
            4 * (lastN['events'] == 'home_run').sum()
        ])
        ab = sum((lastN['events'] == x).sum() for x in ['single', 'double', 'triple', 'home_run', 'field_out', 'force_out', 'other_out'])
        # xSLG and xISO
        single = (lastN['xba'] >= 0.5) & (lastN['launch_angle'] < 15)
        double = (lastN['xba'] >= 0.5) & (lastN['launch_angle'].between(15, 30))
        triple = (lastN['xba'] >= 0.5) & (lastN['launch_angle'].between(30, 40))
        hr = (lastN['launch_angle'] > 35) & (lastN['exit_velocity'] > 100)
        xsingle = single.sum()
        xdouble = double.sum()
        xtriple = triple.sum()
        xhr = hr.sum()
        xab = xsingle + xdouble + xtriple + xhr
        xslg = (1 * xsingle + 2 * xdouble + 3 * xtriple + 4 * xhr) / xab if xab else np.nan
        xba = lastN['xba'].mean() if 'xba' in lastN.columns and not lastN['xba'].isnull().all() else np.nan
        xiso = (xslg - xba) if xslg is not np.nan and xba is not np.nan else np.nan

        feats[f'P_BarrelRateAllowed_{w}'] = barrels / total if total else np.nan
        feats[f'P_EVAllowed_{w}'] = lastN['exit_velocity'].mean() if total else np.nan
        feats[f'P_SLG_{w}'] = slg / ab if ab else np.nan
        feats[f'P_xSLG_{w}'] = xslg
        feats[f'P_xISO_{w}'] = xiso
        feats[f'P_xwoba_{w}'] = lastN['xwoba'].mean() if total else np.nan
        feats[f'P_sweet_spot_pct_{w}'] = sweet / total if total else np.nan
        feats[f'P_hardhit_pct_{w}'] = hard / total if total else np.nan
    return pd.Series(feats)

st.subheader("🔄 Generating rolling averages (batter and pitcher)...")

# --- Batter features ---
batter_feats = df.groupby('batter_id').apply(rolling_batter_features).reset_index()
df = df.merge(batter_feats, on='batter_id', how='left')

# --- Pitcher features ---
pitcher_feats = df.groupby('pitcher_id').apply(rolling_pitcher_features).reset_index()
df = df.merge(pitcher_feats, on='pitcher_id', how='left')

st.success("Rolling features added!")

#### -------------- ML LOGISTIC REGRESSION --------------

feature_cols = []
for base in ['B_BarrelRate', 'B_EV', 'B_SLG', 'B_xSLG', 'B_xISO', 'B_xwoba', 'B_sweet_spot_pct', 'B_hardhit_pct',
             'P_BarrelRateAllowed', 'P_EVAllowed', 'P_SLG', 'P_xSLG', 'P_xISO', 'P_xwoba', 'P_sweet_spot_pct', 'P_hardhit_pct']:
    for w in windows:
        feature_cols.append(f"{base}_{w}")

feature_cols = [c for c in feature_cols if c in df.columns]

st.subheader("Logistic Regression Weights (Predictor-Compatible)")
X = df[feature_cols].fillna(0)
y = df['hr_outcome']
if X.shape[0] > 100 and y.nunique() == 2:
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    weights = pd.Series(model.coef_[0], index=feature_cols).sort_values(ascending=False)
    weights_df = pd.DataFrame({'feature': weights.index, 'weight': weights.values})
    st.write(weights_df)
    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    st.markdown(f"**In-sample AUC:** `{auc:.3f}`")
    st.download_button(
        "Download Predictor-Compatible Logit Weights CSV",
        data=weights_df.to_csv(index=False).encode(),
        file_name="logit_feature_weights.csv"
    )
else:
    st.warning("Not enough data for regression.")

st.subheader("Sample Data with Features")
st.dataframe(df[feature_cols + ['hr_outcome']].head(20))

#### -------------- CONTEXTUAL HR CSV EXPORTS --------------

st.subheader("Download Contextual HR Rate CSVs")

# HR by Handedness
if 'stand' in df.columns and 'p_throws' in df.columns:
    hand_hr_df = df.groupby(['stand', 'p_throws'])['hr_outcome'].mean().reset_index()
    st.write("HR Rate by Batter vs Pitcher Handedness")
    st.dataframe(hand_hr_df)
    st.download_button(
        "Download HR Rate by Handedness",
        data=hand_hr_df.to_csv(index=False).encode(),
        file_name="hr_rate_by_hand.csv"
    )

# HR by Pitch Type
if 'pitch_type' in df.columns:
    pitch_hr_df = df.groupby('pitch_type')['hr_outcome'].mean().reset_index()
    st.write("HR Rate by Pitch Type")
    st.dataframe(pitch_hr_df)
    st.download_button(
        "Download HR Rate by Pitch Type",
        data=pitch_hr_df.to_csv(index=False).encode(),
        file_name="hr_rate_by_pitch_type.csv"
    )

# HR by Ballpark (home_team)
if 'home_team' in df.columns:
    park_hr_df = df.groupby('home_team')['hr_outcome'].mean().reset_index()
    st.write("HR Rate by Ballpark")
    st.dataframe(park_hr_df)
    st.download_button(
        "Download HR Rate by Ballpark",
        data=park_hr_df.to_csv(index=False).encode(),
        file_name="hr_rate_by_park.csv"
        )
