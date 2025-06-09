import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

st.set_page_config(layout="wide")
st.title("MLB Statcast Super Analyzer (All Features)")

st.markdown("""
Upload a Statcast batted-ball event CSV (in-play only, as rich as possible).
This analyzer will:
- Compute rolling averages for ALL core batter & pitcher Statcast metrics (3,5,7,14 days)
- Engineer advanced batted-ball profile rates (batter & pitcher)
- Calculate whiff rates (batter/pitcher) and pitcher spin
- Compute contextual HR rates (handedness, pitch type, park)
- Run logistic regression on ALL engineered features for logit weights
- Export full feature CSV and rate tables ready for leaderboard models
""")

# =============== File Upload ===============
csv_file = st.file_uploader("Upload Statcast Batted Ball Events CSV", type=["csv"])
if not csv_file:
    st.info("Upload a CSV to begin analysis.")
    st.stop()

# =============== Load & Prep Data ===============
df = pd.read_csv(csv_file)
df.columns = [c.lower().replace(" ", "_") for c in df.columns]

# Game date
if 'game_date' in df.columns:
    df['game_date'] = pd.to_datetime(df['game_date'])
else:
    st.error("Missing 'game_date' column in CSV.")
    st.stop()

# HR tag
hr_events = ['home_run', 'home run', 'hr']
if 'events' in df.columns:
    df['hr_outcome'] = df['events'].str.lower().isin(hr_events).astype(int)
elif 'result' in df.columns:
    df['hr_outcome'] = df['result'].str.lower().isin(hr_events).astype(int)
else:
    df['hr_outcome'] = 0

# IDs, exit velo, angle, spins
df['batter_id'] = df.get('batter_id', df.get('batter', np.nan))
df['pitcher_id'] = df.get('pitcher_id', df.get('pitcher', np.nan))
df['exit_velocity'] = df.get('launch_speed', df.get('exit_velocity', np.nan))
df['launch_angle'] = df.get('launch_angle', np.nan)
df['release_spin_rate'] = df.get('release_spin_rate', np.nan)
df['release_speed'] = df.get('release_speed', np.nan)
df['p_throws'] = df.get('p_throws', df.get('pitcher_hand', 'NA')).fillna('NA')
df['stand'] = df.get('stand', df.get('batter_hand', 'NA')).fillna('NA')
df['pitch_type'] = df.get('pitch_type', np.nan)

# xwOBA & xBA
df['xwoba'] = (
    df['estimated_woba_using_speedangle']
    if 'estimated_woba_using_speedangle' in df.columns else
    (df['woba_value'] if 'woba_value' in df.columns else np.nan)
)
df['xba'] = df['estimated_ba_using_speedangle'] if 'estimated_ba_using_speedangle' in df.columns else np.nan

# Weather (if present)
df['temp'] = df.get('temp', np.nan)
df['wind'] = df.get('wind', np.nan)
df['humidity'] = df.get('humidity', np.nan)
df['wind_dir'] = df.get('wind_dir', np.nan)

# =============== Rolling Window Feature Engineering ===============
windows = [3, 5, 7, 14]

def rolling_batter_features(group):
    group = group.sort_values('game_date')
    feats = {}
    for w in windows:
        lastN = group.tail(w)
        pa = len(lastN)
        barrels = lastN[(lastN['exit_velocity'] > 95) & (lastN['launch_angle'].between(20, 35))].shape[0]
        sweet = lastN[lastN['launch_angle'].between(8, 32)].shape[0]
        hard = lastN[lastN['exit_velocity'] >= 95].shape[0]
        # SLG
        slg = np.nansum([
            (lastN['events'] == 'single').sum(),
            2 * (lastN['events'] == 'double').sum(),
            3 * (lastN['events'] == 'triple').sum(),
            4 * (lastN['events'] == 'home_run').sum()
        ])
        ab = sum((lastN['events'] == x).sum() for x in ['single', 'double', 'triple', 'home_run', 'field_out', 'force_out', 'other_out'])
        # xSLG/xISO
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
        xiso = (xslg - xba) if (not pd.isnull(xslg) and not pd.isnull(xba)) else np.nan

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
        xiso = (xslg - xba) if (not pd.isnull(xslg) and not pd.isnull(xba)) else np.nan

        feats[f'P_BarrelRateAllowed_{w}'] = barrels / total if total else np.nan
        feats[f'P_EVAllowed_{w}'] = lastN['exit_velocity'].mean() if total else np.nan
        feats[f'P_SLG_{w}'] = slg / ab if ab else np.nan
        feats[f'P_xSLG_{w}'] = xslg
        feats[f'P_xISO_{w}'] = xiso
        feats[f'P_xwoba_{w}'] = lastN['xwoba'].mean() if total else np.nan
        feats[f'P_sweet_spot_pct_{w}'] = sweet / total if total else np.nan
        feats[f'P_hardhit_pct_{w}'] = hard / total if total else np.nan
    return pd.Series(feats)

st.subheader("Rolling Averages (Batter/Pitcher)")
batter_feats = df.groupby('batter_id').apply(rolling_batter_features).reset_index()
df = df.merge(batter_feats, on='batter_id', how='left')

pitcher_feats = df.groupby('pitcher_id').apply(rolling_pitcher_features).reset_index()
df = df.merge(pitcher_feats, on='pitcher_id', how='left')
st.success("Rolling features added!")

# =============== Whiff Rate Features ===============
def whiff_rate_batter(group):
    swing_mask = group['description'].isin(['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip']) if 'description' in group else pd.Series(False, index=group.index)
    whiff_mask = group['description'].isin(['swinging_strike', 'swinging_strike_blocked']) if 'description' in group else pd.Series(False, index=group.index)
    swings = swing_mask.sum()
    whiffs = whiff_mask.sum()
    return pd.Series({'B_WhiffRate': whiffs / swings if swings > 0 else np.nan})

def whiff_rate_pitcher(group):
    swing_mask = group['description'].isin(['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip']) if 'description' in group else pd.Series(False, index=group.index)
    whiff_mask = group['description'].isin(['swinging_strike', 'swinging_strike_blocked']) if 'description' in group else pd.Series(False, index=group.index)
    swings = swing_mask.sum()
    whiffs = whiff_mask.sum()
    return pd.Series({'P_WhiffRate': whiffs / swings if swings > 0 else np.nan})

b_whiff = df.groupby('batter_id').apply(whiff_rate_batter).reset_index()
p_whiff = df.groupby('pitcher_id').apply(whiff_rate_pitcher).reset_index()
df = df.merge(b_whiff, on='batter_id', how='left').merge(p_whiff, on='pitcher_id', how='left')

# =============== Pitcher Spin Rate (FF) ===============
def spin_rate_pitcher(group):
    fb = group[group['pitch_type'] == 'FF']
    return pd.Series({'P_FF_Spin': fb['release_spin_rate'].mean() if not fb.empty else np.nan})

p_spin = df.groupby('pitcher_id').apply(spin_rate_pitcher).reset_index()
df = df.merge(p_spin, on='pitcher_id', how='left')

# =============== Batted Ball Profile Features ===============
def batted_ball_profile_batter(group):
    total = len(group)
    def rate(mask): return mask.sum() / total if total > 0 else np.nan
    # Angles: GB < 10°, LD 10–25°, FB 25–50°, PU > 50°
    return pd.Series({
        'gb_rate': rate(group['launch_angle'] < 10),
        'ld_rate': rate((group['launch_angle'] >= 10) & (group['launch_angle'] < 25)),
        'fb_rate': rate((group['launch_angle'] >= 25) & (group['launch_angle'] < 50)),
        'pu_rate': rate(group['launch_angle'] >= 50),
        'air_rate': rate(group['launch_angle'] >= 10),
        # Pull/Straight/Oppo: pull < 0°, oppo > 0°
        'pull_rate': rate(group['hc_x'] < 125) if 'hc_x' in group else np.nan,
        'straight_rate': rate((group['hc_x'] >= 125) & (group['hc_x'] <= 175)) if 'hc_x' in group else np.nan,
        'oppo_rate': rate(group['hc_x'] > 175) if 'hc_x' in group else np.nan,
    })

def batted_ball_profile_pitcher(group):
    total = len(group)
    def rate(mask): return mask.sum() / total if total > 0 else np.nan
    return pd.Series({
        'gb_rate_pbb': rate(group['launch_angle'] < 10),
        'ld_rate_pbb': rate((group['launch_angle'] >= 10) & (group['launch_angle'] < 25)),
        'fb_rate_pbb': rate((group['launch_angle'] >= 25) & (group['launch_angle'] < 50)),
        'pu_rate_pbb': rate(group['launch_angle'] >= 50),
        'air_rate_pbb': rate(group['launch_angle'] >= 10),
        'pull_rate_pbb': rate(group['hc_x'] < 125) if 'hc_x' in group else np.nan,
        'straight_rate_pbb': rate((group['hc_x'] >= 125) & (group['hc_x'] <= 175)) if 'hc_x' in group else np.nan,
        'oppo_rate_pbb': rate(group['hc_x'] > 175) if 'hc_x' in group else np.nan,
    })

b_bb = df.groupby('batter_id').apply(batted_ball_profile_batter).reset_index()
p_bb = df.groupby('pitcher_id').apply(batted_ball_profile_pitcher).reset_index()
df = df.merge(b_bb, on='batter_id', how='left').merge(p_bb, on='pitcher_id', how='left')

# =============== Contextual HR Rates ===============
st.subheader("Contextual HR Rate Tables")

if 'stand' in df.columns and 'p_throws' in df.columns:
    hand_hr_df = (
        df.groupby(['stand', 'p_throws'])['hr_outcome'].mean().reset_index()
        .rename(columns={
            'stand': 'BatterHandedness',
            'p_throws': 'PitcherHandedness',
            'hr_outcome': 'HandedHRRate'
        })
    )
    st.write("HR Rate by Batter vs Pitcher Handedness")
    st.dataframe(hand_hr_df)
    st.download_button(
        "Download HR Rate by Handedness",
        data=hand_hr_df.to_csv(index=False).encode(),
        file_name="hr_rate_by_hand.csv"
    )

if 'pitch_type' in df.columns:
    pitch_hr_df = (
        df.groupby('pitch_type')['hr_outcome'].mean()
        .reset_index()
        .rename(columns={
            'pitch_type': 'pitch_type',
            'hr_outcome': 'PitchTypeHRRate'
        })
    )
    st.write("HR Rate by Pitch Type")
    st.dataframe(pitch_hr_df)
    st.download_button(
        "Download HR Rate by Pitch Type",
        data=pitch_hr_df.to_csv(index=False).encode(),
        file_name="hr_rate_by_pitch_type.csv"
    )

if 'home_team' in df.columns:
    park_hr_df = (
        df.groupby('home_team')['hr_outcome'].mean()
        .reset_index()
        .rename(columns={
            'home_team': 'park',
            'hr_outcome': 'ParkHRRate'
        })
    )
    st.write("HR Rate by Ballpark")
    st.dataframe(park_hr_df)
    st.download_button(
        "Download HR Rate by Ballpark",
        data=park_hr_df.to_csv(index=False).encode(),
        file_name="hr_rate_by_park.csv"
    )

# =============== Logistic Regression on All Features ===============
st.subheader("Logistic Regression (ALL Features)")

ignore_cols = [
    'batter_id', 'pitcher_id', 'game_date', 'events', 'result', 'description', 'home_team',
    'stand', 'p_throws', 'batter', 'pitcher', 'hr_outcome', 'pitch_type', 'hc_x', 'hc_y'
]
feature_cols = [col for col in df.columns if col not in ignore_cols and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]

X = df[feature_cols].fillna(0)
y = df['hr_outcome']
if X.shape[0] > 100 and y.nunique() == 2:
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    weights = pd.Series(model.coef_[0], index=feature_cols).sort_values(ascending=False)
    weights_df = pd.DataFrame({'feature': weights.index, 'weight': weights.values})
    st.write("**Feature Weights:**")
    st.dataframe(weights_df)

    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    st.markdown(f"**In-sample AUC:** `{auc:.3f}`")

    st.download_button(
        "Download Logit Weights CSV",
        data=weights_df.to_csv(index=False).encode(),
        file_name="logit_feature_weights.csv"
    )

    # Optionally, download ALL rows with all engineered features for leaderboard model
    out_df = df[['batter_id', 'pitcher_id', 'game_date', 'hr_outcome'] + feature_cols]
    st.download_button(
        "Download Full Feature Export (for Predictor)",
        data=out_df.to_csv(index=False).encode(),
        file_name="all_features_export.csv"
    )
else:
    st.warning("Not enough data for regression or not enough HR outcomes in data.")

st.subheader("Sample Data (First 20 Rows with All Features)")
st.dataframe(df[['batter_id', 'pitcher_id', 'game_date', 'hr_outcome'] + feature_cols].head(20))
