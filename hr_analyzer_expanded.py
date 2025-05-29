import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from pybaseball import statcast
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

st.header("1️⃣ Generate Last 7 Days Batted Ball Events CSV")

if st.button("Generate and Download 7-Day Batted Ball Events CSV"):
    today = datetime.now().date()
    seven_days_ago = today - timedelta(days=7)
    with st.spinner(f"Fetching batted ball data from {seven_days_ago} to {today}..."):
        df_events = statcast(seven_days_ago.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"))
    st.success(f"Downloaded {len(df_events)} batted ball events.")
    csv_bytes = df_events.to_csv(index=False).encode()
    st.download_button("Download CSV", csv_bytes, file_name="batted_ball_7days.csv")
    st.write(df_events.head())
    
st.title("MLB HR Analyzer — 7-Day Batted Ball Events & Feature Weighting")

st.markdown("""
Upload your 7-day batted ball events CSV (export from Statcast, Savant, or bot).  
**This app will:**
- Auto-tag HR outcomes
- Engineer advanced Statcast features
- Show mean difference and logistic regression weightings
- Provide quick visualizations of HR patterns  
""")

csv = st.file_uploader("Upload 7-day Batted Ball Events CSV", type=["csv"])

def engineer_features(df):
    # Normalize missing or alternate column names
    cols = [c.lower().replace(' ', '_') for c in df.columns]
    df.columns = cols

    # HR outcome auto-tag
    hr_events = ['home_run', 'home run', 'HR', 'hr']
    if 'events' in df.columns:
        df['hr_outcome'] = df['events'].str.lower().isin(hr_events).astype(int)
    elif 'result' in df.columns:
        df['hr_outcome'] = df['result'].str.lower().isin(hr_events).astype(int)
    else:
        df['hr_outcome'] = 0  # fallback

    # Statcast features (fillna for safety)
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

    # Zone (optional: bucket plate_x and plate_z into zones)
    df['zone'] = (
        pd.cut(df['plate_x'], bins=[-2, -0.7, 0.7, 2], labels=['left', 'middle', 'right'])
        .astype(str) + "_" +
        pd.cut(df['plate_z'], bins=[0, 2, 3.5, 5], labels=['low', 'middle', 'high']).astype(str)
    )

    # Barrel flag (Statcast)
    df['barrel'] = (
        (df['exit_velocity'] > 98) &
        (df['launch_angle'].between(26, 30))
    ).astype(int)

    # Whiff flag
    df['is_whiff'] = df['description'].str.contains('miss', case=False, na=False).astype(int)

    # Pull/opp/straight flags if hit_direction exists
    if 'hit_direction' in df.columns:
        df['pull'] = (df['hit_direction'].str.lower() == 'pull').astype(int)
        df['oppo'] = (df['hit_direction'].str.lower() == 'opposite').astype(int)
        df['straight'] = (df['hit_direction'].str.lower() == 'straight').astype(int)

    # Date normalization (if present)
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])

    return df

def mean_diff_weights(df, feature_cols):
    means_hr = df[df['hr_outcome'] == 1][feature_cols].mean()
    means_no = df[df['hr_outcome'] == 0][feature_cols].mean()
    mean_diff = means_hr - means_no
    return mean_diff.sort_values(ascending=False)

def logistic_regression_weights(df, feature_cols):
    X = df[feature_cols].fillna(0)
    y = df['hr_outcome']
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    weights = pd.Series(model.coef_[0], index=feature_cols)
    return weights.sort_values(ascending=False)

if csv:
    df = pd.read_csv(csv)
    df = engineer_features(df)

    # Display sample data
    st.subheader("Sample Data (Tagged)")
    st.dataframe(df.head(10))

    # Feature set for weighting (expand as needed)
    feature_cols = [
        'exit_velocity', 'launch_angle', 'spin_rate', 'pitch_velocity',
        'barrel', 'is_whiff',
        'plate_x', 'plate_z',
        'balls', 'strikes'
    ]
    # Only use if present
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Show mean difference weights
    st.subheader("Mean Difference (HR vs Non-HR)")
    mean_diff = mean_diff_weights(df, feature_cols)
    st.bar_chart(mean_diff)
    st.write(mean_diff)

    # Show logistic regression weights
    st.subheader("Logistic Regression Feature Weights")
    logit_weights = logistic_regression_weights(df, feature_cols)
    st.bar_chart(logit_weights)
    st.write(logit_weights)

    # Quick histograms for key features
    st.subheader("Feature Distributions (HR vs Non-HR)")
    for c in feature_cols:
        st.write(f"**{c}**")
        hr_hist = df[df['hr_outcome'] == 1][c].dropna()
        no_hr_hist = df[df['hr_outcome'] == 0][c].dropna()
        st.histogram([hr_hist, no_hr_hist], bins=20, legend=['HR', 'No HR'])
else:
    st.info("Upload a CSV to begin analysis.")
