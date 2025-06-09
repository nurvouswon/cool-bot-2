# BLOCK 1: Imports, helper functions, config
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

st.set_page_config(layout="wide")
st.title("MLB Statcast All-in-One Analyzer & HR Predictor")

@st.cache_data
def normalize_name(name):
    import unicodedata
    if not isinstance(name, str):
        return ""
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = name.lower().replace('.', '').replace('-', ' ').replace("’", "'").strip()
    if ',' in name:
        last, first = name.split(',', 1)
        name = f"{first.strip()} {last.strip()}"
    return ' '.join(name.split())

def get_hand(p):
    p = str(p).strip().upper()
    return p if p in {"R", "L", "S"} else "NA"

windows_default = [3, 5, 7, 14]
# BLOCK 2: Feature engineering from raw batted ball CSV

st.header("1️⃣ Generate Rolling Batted Ball Feature CSVs")
raw_csv = st.file_uploader("Upload Raw Statcast Batted Ball CSV", type="csv", key="rawcsv")
win_sel = st.multiselect("Select rolling windows (games)", windows_default, default=windows_default, key="win_sel")

def engineer_features(df, windows):
    df = df.copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    # Standardize and robustify key columns
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['hr_outcome'] = df.get('events', df.get('result', '')).str.lower().isin(['home_run', 'home run', 'hr']).astype(int)
    df['batter_id'] = df.get('batter_id', df.get('batter', np.nan)).astype(str)
    df['pitcher_id'] = df.get('pitcher_id', df.get('pitcher', np.nan)).astype(str)
    df['exit_velocity'] = df.get('launch_speed', df.get('exit_velocity', np.nan))
    df['launch_angle'] = df.get('launch_angle', np.nan)
    df['release_spin_rate'] = df.get('release_spin_rate', np.nan)
    df['release_speed'] = df.get('release_speed', np.nan)
    df['p_throws'] = df.get('p_throws', df.get('pitcher_hand', 'NA')).fillna('NA').apply(get_hand)
    df['stand'] = df.get('stand', df.get('batter_hand', 'NA')).fillna('NA').apply(get_hand)
    df['pitch_type'] = df.get('pitch_type', np.nan)
    df['home_team'] = df.get('home_team', df.get('park', np.nan))

    # xwOBA/xBA handling
    df['xwoba'] = df.get('estimated_woba_using_speedangle', df.get('woba_value', np.nan))
    df['xba'] = df.get('estimated_ba_using_speedangle', np.nan)
    # Rolling feature generation
    all_feats = []
    for idcol, groupby in [('batter_id', 'B_'), ('pitcher_id', 'P_')]:
        for idval, group in df.groupby(idcol):
            group = group.sort_values('game_date')
            for i in range(len(group)):
                row = group.iloc[i].to_dict()
                for w in windows:
                    tail = group.iloc[max(0, i-w+1):i+1]
                    pa = len(tail)
                    barrels = tail[(tail['exit_velocity'] > 95) & (tail['launch_angle'].between(20, 35))].shape[0]
                    sweet = tail[tail['launch_angle'].between(8, 32)].shape[0]
                    hard = tail[tail['exit_velocity'] >= 95].shape[0]
                    slg = np.nansum([
                        (tail['events'] == 'single').sum(),
                        2 * (tail['events'] == 'double').sum(),
                        3 * (tail['events'] == 'triple').sum(),
                        4 * (tail['events'] == 'home_run').sum()
                    ])
                    ab = sum((tail['events'] == x).sum() for x in ['single', 'double', 'triple', 'home_run', 'field_out', 'force_out', 'other_out'])
                    xsingle = ((tail['xba'] >= 0.5) & (tail['launch_angle'] < 15)).sum()
                    xdouble = ((tail['xba'] >= 0.5) & (tail['launch_angle'].between(15, 30))).sum()
                    xtriple = ((tail['xba'] >= 0.5) & (tail['launch_angle'].between(30, 40))).sum()
                    xhr = ((tail['launch_angle'] > 35) & (tail['exit_velocity'] > 100)).sum()
                    xab = xsingle + xdouble + xtriple + xhr
                    xslg = (1 * xsingle + 2 * xdouble + 3 * xtriple + 4 * xhr) / xab if xab else np.nan
                    xba = tail['xba'].mean() if 'xba' in tail.columns and not tail['xba'].isnull().all() else np.nan
                    xiso = (xslg - xba) if pd.notnull(xslg) and pd.notnull(xba) else np.nan
                    prefix = groupby
                    row[f'{prefix}BarrelRate_{w}'] = barrels / pa if pa else np.nan
                    row[f'{prefix}EV_{w}'] = tail['exit_velocity'].mean() if pa else np.nan
                    row[f'{prefix}SLG_{w}'] = slg / ab if ab else np.nan
                    row[f'{prefix}xSLG_{w}'] = xslg
                    row[f'{prefix}xISO_{w}'] = xiso
                    row[f'{prefix}xwoba_{w}'] = tail['xwoba'].mean() if pa else np.nan
                    row[f'{prefix}sweet_spot_pct_{w}'] = sweet / pa if pa else np.nan
                    row[f'{prefix}hardhit_pct_{w}'] = hard / pa if pa else np.nan
                all_feats.append(row)
    feats_df = pd.DataFrame(all_feats)
    # Deduplicate: keep last event per (game_date, batter_id, pitcher_id)
    feats_df = feats_df.sort_values('game_date').drop_duplicates(['game_date', 'batter_id', 'pitcher_id'], keep='last')
    return feats_df

if raw_csv and win_sel:
    st.write("Processing features...")
    df_feat = engineer_features(pd.read_csv(raw_csv), win_sel)
    st.write(df_feat.head())
    st.download_button("Download Feature CSV", data=df_feat.to_csv(index=False), file_name="batted_ball_features.csv")
else:
    df_feat = None
    # BLOCK 3: Contextual HR rates & logit weights

st.header("2️⃣ Export Contextual HR Rate & Logit Weight CSVs")

if df_feat is not None:
    # --- Handedness HR Rate ---
    if 'stand' in df_feat.columns and 'p_throws' in df_feat.columns:
        hand_hr_df = (
            df_feat.groupby(['stand', 'p_throws'])['hr_outcome'].mean().reset_index()
            .rename(columns={
                'stand': 'BatterHandedness',
                'p_throws': 'PitcherHandedness',
                'hr_outcome': 'HandedHRRate'
            })
        )
        st.subheader("HR Rate by Batter vs Pitcher Handedness")
        st.dataframe(hand_hr_df)
        st.download_button(
            "Download HR Rate by Handedness",
            data=hand_hr_df.to_csv(index=False).encode(),
            file_name="hr_rate_by_hand.csv"
        )

    # --- Pitch Type HR Rate ---
    if 'pitch_type' in df_feat.columns:
        pitch_hr_df = (
            df_feat.groupby('pitch_type')['hr_outcome'].mean()
            .reset_index()
            .rename(columns={
                'pitch_type': 'pitch_type',
                'hr_outcome': 'PitchTypeHRRate'
            })
        )
        st.subheader("HR Rate by Pitch Type")
        st.dataframe(pitch_hr_df)
        st.download_button(
            "Download HR Rate by Pitch Type",
            data=pitch_hr_df.to_csv(index=False).encode(),
            file_name="hr_rate_by_pitch_type.csv"
        )

    # --- Ballpark HR Rate ---
    if 'home_team' in df_feat.columns:
        park_hr_df = (
            df_feat.groupby('home_team')['hr_outcome'].mean()
            .reset_index()
            .rename(columns={
                'home_team': 'park',
                'hr_outcome': 'ParkHRRate'
            })
        )
        st.subheader("HR Rate by Ballpark")
        st.dataframe(park_hr_df)
        st.download_button(
            "Download HR Rate by Ballpark",
            data=park_hr_df.to_csv(index=False).encode(),
            file_name="hr_rate_by_park.csv"
        )

    # --- LOGIT FEATURE WEIGHTS (All features present) ---
    # Detect all feature columns for regression
    skip_cols = {'game_date', 'batter_id', 'pitcher_id', 'exit_velocity', 'launch_angle',
                 'release_spin_rate', 'release_speed', 'events', 'result', 'pitch_type',
                 'home_team', 'stand', 'p_throws', 'xwoba', 'xba', 'hr_outcome'}
    feature_cols = [c for c in df_feat.columns if c not in skip_cols and df_feat[c].dtype != 'O']

    X = df_feat[feature_cols].fillna(0)
    y = df_feat['hr_outcome']
    if X.shape[0] > 100 and y.nunique() == 2:
        st.subheader("Logistic Regression Weights (All Features)")
        model = LogisticRegression(max_iter=200)
        model.fit(X, y)
        weights = pd.Series(model.coef_[0], index=feature_cols).sort_values(ascending=False)
        weights_df = pd.DataFrame({'feature': weights.index, 'weight': weights.values})
        st.write(weights_df)
        auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
        st.markdown(f"**In-sample AUC:** `{auc:.3f}`")
        st.download_button(
            "Download All-Features Logit Weights CSV",
            data=weights_df.to_csv(index=False).encode(),
            file_name="logit_feature_weights.csv"
        )
    else:
        st.warning("Not enough labeled data for regression.")

    # --- Output example (for debugging) ---
    st.subheader("Sample Data with Features")
    st.dataframe(df_feat.head(20))

else:
    st.info("Upload raw data & process features first to unlock these outputs.")
    # BLOCK 4: HR Leaderboard Scoring from Exported Feature/Context/Logit CSVs

st.header("3️⃣ Leaderboard: Upload & Score Today's Matchups")

st.markdown("""
Upload your game-day **lineup/matchup** CSV and the exported **logit weights, contextual HR rates, batted ball profiles, etc.**  
The app will:
- Merge all factors for each matchup
- Compute a normalized logit score using the weights
- Rank and output the top HR matchups!
""")

uploaded_lineup = st.file_uploader("Lineup/Matchup CSV (game day)", type=["csv"])
uploaded_logit = st.file_uploader("Logit Weights CSV (from Analyzer)", type=["csv"])
uploaded_hand = st.file_uploader("Handedness HR Rate CSV", type=["csv"])
uploaded_pitchtype = st.file_uploader("Pitch Type HR Rate CSV", type=["csv"])
uploaded_park = st.file_uploader("Park HR Rate CSV", type=["csv"])

all_inputs = [uploaded_lineup, uploaded_logit, uploaded_hand, uploaded_pitchtype, uploaded_park]
if all(all_inputs):
    df_lineup = pd.read_csv(uploaded_lineup)
    df_logit = pd.read_csv(uploaded_logit)
    hand_df = pd.read_csv(uploaded_hand)
    pitchtype_df = pd.read_csv(uploaded_pitchtype)
    park_df = pd.read_csv(uploaded_park)

    # Merge contextual rates onto lineup
    if 'BatterHandedness' in hand_df.columns and 'PitcherHandedness' in hand_df.columns:
        df_lineup = df_lineup.merge(
            hand_df,
            left_on=['stand', 'p_throws'],
            right_on=['BatterHandedness', 'PitcherHandedness'],
            how='left'
        )
    if 'pitch_type' in pitchtype_df.columns:
        df_lineup = df_lineup.merge(
            pitchtype_df,
            on='pitch_type',
            how='left'
        )
    if 'park' in park_df.columns:
        # Try to normalize ballpark name field
        df_lineup['park'] = df_lineup['park'].str.lower().str.replace(' ', '_')
        park_df['park'] = park_df['park'].str.lower().str.replace(' ', '_')
        df_lineup = df_lineup.merge(
            park_df,
            on='park',
            how='left'
        )

    # Use logit weights to compute score
    logit_dict = dict(zip(df_logit['feature'], df_logit['weight']))
    def compute_logit_score(row):
        score = 0
        for feat, weight in logit_dict.items():
            v = row.get(feat, 0)
            if pd.isnull(v):
                v = 0
            score += v * weight
        return score

    df_lineup['HR_Score'] = df_lineup.apply(compute_logit_score, axis=1)
    df_lineup['HR_Rank'] = df_lineup['HR_Score'].rank(ascending=False, method='min')
    df_lineup = df_lineup.sort_values('HR_Score', ascending=False).reset_index(drop=True)
    st.success("Leaderboard created!")
    st.dataframe(df_lineup.head(25))

    # Download button for leaderboard
    st.download_button(
        "Download HR Matchup Leaderboard",
        data=df_lineup.to_csv(index=False).encode(),
        file_name="hr_leaderboard.csv"
    )
else:
    st.info("Upload all 5 required CSVs to see leaderboard.")
