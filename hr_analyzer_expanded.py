import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybaseball import statcast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

st.header("📅 Generate Batted Ball Events (Last N Days) — In-Play Only")

@st.cache_data
def get_statcast_data(start_date, end_date):
    return statcast(start_date, end_date)

# Select number of days
num_days = st.slider("Select number of past days to collect Statcast data", 7, 60, 30)

if st.button(f"Generate and Download {num_days}-Day Batted Ball Events CSV"):
    today = datetime.now().date()
    start_date = today - timedelta(days=num_days)
    with st.spinner(f"Fetching batted ball data from {start_date} to {today}..."):
        df_events = get_statcast_data(start_date.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"))
        df_events = df_events[df_events['type'] == 'X']  # in-play only
    st.success(f"Downloaded {len(df_events)} batted ball events.")
    csv_bytes = df_events.to_csv(index=False).encode()
    st.download_button("Download CSV", csv_bytes, file_name=f"batted_ball_{num_days}days.csv")
    st.write(df_events.head())

st.title("MLB HR Analyzer — Batted Ball Events & Feature Weighting")

st.markdown("""
Upload a CSV of Statcast batted ball events (in-play only).  
This app will:
- Tag HRs
- Engineer advanced features
- Show logistic regression weights & mean differences
- Compute AUC score
- Visualize HR trends by pitch type, handedness, and park
- 📤 Export logistic weights + contextual HR patterns for your HR bot
""")

csv = st.file_uploader("Upload Batted Ball Events CSV", type=["csv"])

def engineer_features(df):
    cols = [c.lower().replace(' ', '_') for c in df.columns]
    df.columns = cols

    df = df[df['type'] == 'X']

    hr_events = ['home_run', 'home run', 'hr']
    if 'events' in df.columns:
        df['hr_outcome'] = df['events'].str.lower().isin(hr_events).astype(int)
    elif 'result' in df.columns:
        df['hr_outcome'] = df['result'].str.lower().isin(hr_events).astype(int)
    else:
        df['hr_outcome'] = 0

    df['exit_velocity'] = df.get('launch_speed', df.get('exit_velocity', np.nan))
    df['launch_angle'] = df.get('launch_angle', np.nan)
    df['spin_rate'] = df.get('release_spin_rate', np.nan)
    df['pitch_velocity'] = df.get('release_speed', np.nan)
    df['pitch_type'] = df.get('pitch_type', df.get('pitch_name', 'NA')).fillna('NA')
    df['plate_x'] = df.get('plate_x', np.nan)
    df['plate_z'] = df.get('plate_z', np.nan)
    df['batter_hand'] = df.get('stand', df.get('batter_hand', 'NA')).fillna('NA')
    df['pitcher_hand'] = df.get('p_throws', df.get('pitcher_hand', 'NA')).fillna('NA')
    df['balls'] = df.get('balls', 0)
    df['strikes'] = df.get('strikes', 0)
    df['description'] = df.get('description', '')

    df['zone'] = (
        pd.cut(df['plate_x'], [-2, -0.7, 0.7, 2], labels=['left', 'middle', 'right']).astype(str) + "_" +
        pd.cut(df['plate_z'], [0, 2, 3.5, 5], labels=['low', 'middle', 'high']).astype(str)
    ).fillna('unknown')

    df['barrel'] = ((df['exit_velocity'] > 98) & df['launch_angle'].between(26, 30)).astype(int)
    df['is_whiff'] = df['description'].str.contains('miss', case=False, na=False).astype(int)

    if 'hit_direction' in df.columns:
        df['pull'] = (df['hit_direction'].str.lower() == 'pull').astype(int)
        df['oppo'] = (df['hit_direction'].str.lower() == 'opposite').astype(int)
        df['straight'] = (df['hit_direction'].str.lower() == 'straight').astype(int)

    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])

    return df

def mean_diff_weights(df, feature_cols):
    means_hr = df[df['hr_outcome'] == 1][feature_cols].mean()
    means_no = df[df['hr_outcome'] == 0][feature_cols].mean()
    return (means_hr - means_no).sort_values(ascending=False)

def logistic_regression_weights(df, feature_cols):
    X = df[feature_cols].fillna(0)
    y = df['hr_outcome']
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    return pd.Series(model.coef_[0], index=feature_cols).sort_values(ascending=False)

def evaluate_model(df, feature_cols):
    X = df[feature_cols].fillna(0)
    y = df['hr_outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred)

if csv:
    df = pd.read_csv(csv)
    df = engineer_features(df)

    st.subheader("Sample Data (Tagged & In-Play Only)")
    st.dataframe(df.head(10))

    feature_cols = [
        'exit_velocity', 'launch_angle', 'spin_rate', 'pitch_velocity',
        'barrel', 'is_whiff', 'plate_x', 'plate_z', 'balls', 'strikes'
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    st.subheader("Mean Difference (HR vs Non-HR)")
    mean_diff = mean_diff_weights(df, feature_cols)
    st.bar_chart(mean_diff)
    st.write(mean_diff)

    st.subheader("Logistic Regression Feature Weights")
    logit_weights = logistic_regression_weights(df, feature_cols)
    st.bar_chart(logit_weights)
    st.write(logit_weights)

    auc = evaluate_model(df, feature_cols)
    st.markdown(f"**Logistic Regression AUC:** `{auc:.3f}`")

    st.subheader("Pitch Type HR Rate")
    if 'pitch_type' in df.columns:
        pitch_hr = df.groupby('pitch_type')['hr_outcome'].mean().sort_values(ascending=False)
        st.bar_chart(pitch_hr)
        st.write(pitch_hr)

    st.subheader("HR Rate by Batter vs Pitcher Handedness")
    if 'batter_hand' in df.columns and 'pitcher_hand' in df.columns:
        hand_hr = df.groupby(['batter_hand', 'pitcher_hand'])['hr_outcome'].mean().unstack().fillna(0)
        st.dataframe(hand_hr.style.format("{:.3f}").highlight_max(axis=1))

    st.subheader("HR Rate by Ballpark (Home Team)")
    if 'home_team' in df.columns:
        park_hr = df.groupby('home_team')['hr_outcome'].mean().sort_values(ascending=False)
        st.bar_chart(park_hr)
        st.write(park_hr)

    st.subheader("Feature Distributions (HR vs Non-HR)")
    for c in feature_cols:
        st.write(f"**{c}**")
        fig, ax = plt.subplots()
        ax.hist(df[df['hr_outcome'] == 1][c].dropna(), bins=20, alpha=0.5, label='HR')
        ax.hist(df[df['hr_outcome'] == 0][c].dropna(), bins=20, alpha=0.5, label='No HR')
        ax.set_title(f'{c} distribution')
        ax.legend()
        st.pyplot(fig)

    # Export: Logistic Weights CSV (Predictor-compatible)
    logit_weights_df = pd.DataFrame({
        'feature': logit_weights.index,
        'weight': logit_weights.values
    })
    st.download_button(
        label="Download Logistic Feature Weights",
        data=logit_weights_df.to_csv(index=False).encode(),
        file_name="logit_feature_weights.csv"
    )
    # Export: HR Rate by Handedness
    if 'batter_hand' in df.columns and 'pitcher_hand' in df.columns:
        hand_hr_df = df.groupby(['batter_hand', 'pitcher_hand'])['hr_outcome'].mean().reset_index()
        hand_hr_csv = hand_hr_df.to_csv(index=False).encode()
        st.download_button(
            label="📤 Download HR Rate by Batter vs Pitcher Handedness",
            data=hand_hr_csv,
            file_name="hr_rate_by_hand.csv"
        )

    # Export: HR Rate by Pitch Type
    if 'pitch_type' in df.columns:
        pitch_hr_df = df.groupby('pitch_type')['hr_outcome'].mean().reset_index()
        pitch_hr_csv = pitch_hr_df.to_csv(index=False).encode()
        st.download_button(
            label="📤 Download HR Rate by Pitch Type",
            data=pitch_hr_csv,
            file_name="hr_rate_by_pitch_type.csv"
        )

    # Export: HR Rate by Ballpark
    if 'home_team' in df.columns:
        park_hr_df = df.groupby('home_team')['hr_outcome'].mean().reset_index()
        park_hr_csv = park_hr_df.to_csv(index=False).encode()
        st.download_button(
            label="📤 Download HR Rate by Ballpark (Home Team)",
            data=park_hr_csv,
            file_name="hr_rate_by_park.csv"
        )

else:
    st.info("Upload a CSV to begin analysis.")
