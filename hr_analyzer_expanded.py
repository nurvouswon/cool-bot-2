# [PASTE YOUR CONTEXT MAPS AND DICTIONARIES HERE]

import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import gc
from datetime import datetime, timedelta

# ===================== CONTEXT MAPS & RATES =====================
park_hr_rate_map = {
    'angels_stadium': 1.05, 'angel_stadium': 1.05, 'minute_maid_park': 1.06, 'coors_field': 1.30,
    'yankee_stadium': 1.19, 'fenway_park': 0.97, 'rogers_centre': 1.10, 'tropicana_field': 0.85,
    'camden_yards': 1.13, 'guaranteed_rate_field': 1.18, 'progressive_field': 1.01,
    'comerica_park': 0.96, 'kauffman_stadium': 0.98, 'globe_life_field': 1.00, 'dodger_stadium': 1.10,
    'oakland_coliseum': 0.82, 't-mobile_park': 0.86, 'tmobile_park': 0.86, 'oracle_park': 0.82,
    'wrigley_field': 1.12, 'great_american_ball_park': 1.26, 'american_family_field': 1.17,
    'pnc_park': 0.87, 'busch_stadium': 0.87, 'truist_park': 1.06, 'loan_depot_park': 0.86,
    'loandepot_park': 0.86, 'citi_field': 1.05, 'nationals_park': 1.05, 'petco_park': 0.85,
    'chase_field': 1.06, 'citizens_bank_park': 1.19, 'sutter_health_park': 1.12, 'target_field': 1.05
}
park_altitude_map = {
    'coors_field': 5280, 'chase_field': 1100, 'dodger_stadium': 338, 'minute_maid_park': 50,
    'fenway_park': 19, 'wrigley_field': 594, 'great_american_ball_park': 489, 'oracle_park': 10,
    'petco_park': 62, 'yankee_stadium': 55, 'citizens_bank_park': 30, 'kauffman_stadium': 750,
    'guaranteed_rate_field': 600, 'progressive_field': 650, 'busch_stadium': 466, 'camden_yards': 40,
    'rogers_centre': 250, 'angel_stadium': 160, 'tropicana_field': 3, 'citi_field': 3,
    'oakland_coliseum': 50, 'globe_life_field': 560, 'pnc_park': 725, 'loan_depot_park': 7,
    'loandepot_park': 7, 'nationals_park': 25, 'american_family_field': 633, 'sutter_health_park': 20,
    'target_field': 830
}
roof_status_map = {
    'rogers_centre': 'closed', 'chase_field': 'open', 'minute_maid_park': 'open',
    'loan_depot_park': 'closed', 'loandepot_park': 'closed', 'globe_life_field': 'open',
    'tropicana_field': 'closed', 'american_family_field': 'open'
}
team_code_to_park = {
    'PHI': 'citizens_bank_park', 'ATL': 'truist_park', 'NYM': 'citi_field',
    'BOS': 'fenway_park', 'NYY': 'yankee_stadium', 'CHC': 'wrigley_field',
    'LAD': 'dodger_stadium', 'OAK': 'sutter_health_park', 'ATH': 'sutter_health_park',
    'CIN': 'great_american_ball_park', 'DET': 'comerica_park', 'HOU': 'minute_maid_park',
    'MIA': 'loandepot_park', 'TB': 'tropicana_field', 'MIL': 'american_family_field',
    'SD': 'petco_park', 'SF': 'oracle_park', 'TOR': 'rogers_centre', 'CLE': 'progressive_field',
    'MIN': 'target_field', 'KC': 'kauffman_stadium', 'CWS': 'guaranteed_rate_field',
    'CHW': 'guaranteed_rate_field', 'LAA': 'angel_stadium', 'SEA': 't-mobile_park',
    'TEX': 'globe_life_field', 'ARI': 'chase_field', 'AZ': 'chase_field', 'COL': 'coors_field', 'PIT': 'pnc_park',
    'STL': 'busch_stadium', 'BAL': 'camden_yards', 'WSH': 'nationals_park', 'WAS': 'nationals_park'
}
mlb_team_city_map = {
    'ANA': 'Anaheim', 'ARI': 'Phoenix', 'AZ': 'Phoenix', 'ATL': 'Atlanta', 'BAL': 'Baltimore', 'BOS': 'Boston',
    'CHC': 'Chicago', 'CIN': 'Cincinnati', 'CLE': 'Cleveland', 'COL': 'Denver', 'CWS': 'Chicago',
    'CHW': 'Chicago', 'DET': 'Detroit', 'HOU': 'Houston', 'KC': 'Kansas City', 'LAA': 'Anaheim',
    'LAD': 'Los Angeles', 'MIA': 'Miami', 'MIL': 'Milwaukee', 'MIN': 'Minneapolis', 'NYM': 'New York',
    'NYY': 'New York', 'OAK': 'Oakland', 'ATH': 'Oakland', 'PHI': 'Philadelphia', 'PIT': 'Pittsburgh',
    'SD': 'San Diego', 'SEA': 'Seattle', 'SF': 'San Francisco', 'STL': 'St. Louis', 'TB': 'St. Petersburg',
    'TEX': 'Arlington', 'TOR': 'Toronto', 'WSH': 'Washington', 'WAS': 'Washington'
}
park_hand_hr_rate_map = {
    'angels_stadium': {'L': 1.09, 'R': 1.02}, 'angel_stadium': {'L': 1.09, 'R': 1.02},
    'minute_maid_park': {'L': 1.13, 'R': 1.06}, 'coors_field': {'L': 1.38, 'R': 1.24},
    'yankee_stadium': {'L': 1.47, 'R': 0.98}, 'fenway_park': {'L': 1.04, 'R': 0.97},
    'rogers_centre': {'L': 1.08, 'R': 1.12}, 'tropicana_field': {'L': 0.84, 'R': 0.89},
    'camden_yards': {'L': 0.98, 'R': 1.27}, 'guaranteed_rate_field': {'L': 1.25, 'R': 1.11},
    'progressive_field': {'L': 0.99, 'R': 1.02}, 'comerica_park': {'L': 1.10, 'R': 0.91},
    'kauffman_stadium': {'L': 0.90, 'R': 1.03}, 'globe_life_field': {'L': 1.01, 'R': 0.98},
    'dodger_stadium': {'L': 1.02, 'R': 1.18}, 'oakland_coliseum': {'L': 0.81, 'R': 0.85},
    't-mobile_park': {'L': 0.81, 'R': 0.92}, 'tmobile_park': {'L': 0.81, 'R': 0.92},
    'oracle_park': {'L': 0.67, 'R': 0.99}, 'wrigley_field': {'L': 1.10, 'R': 1.16},
    'great_american_ball_park': {'L': 1.30, 'R': 1.23}, 'american_family_field': {'L': 1.25, 'R': 1.13},
    'pnc_park': {'L': 0.76, 'R': 0.92}, 'busch_stadium': {'L': 0.78, 'R': 0.91},
    'truist_park': {'L': 1.00, 'R': 1.09}, 'loan_depot_park': {'L': 0.83, 'R': 0.91},
    'loandepot_park': {'L': 0.83, 'R': 0.91}, 'citi_field': {'L': 1.11, 'R': 0.98},
    'nationals_park': {'L': 1.04, 'R': 1.06}, 'petco_park': {'L': 0.90, 'R': 0.88},
    'chase_field': {'L': 1.16, 'R': 1.05}, 'citizens_bank_park': {'L': 1.22, 'R': 1.20},
    'sutter_health_park': {'L': 1.12, 'R': 1.12}, 'target_field': {'L': 1.09, 'R': 1.01}
}
# ========== DEEP RESEARCH HR MULTIPLIERS: BATTER SIDE ===============
park_hr_percent_map_all = {
    'ARI': 0.98, 'AZ': 0.98, 'ATL': 0.95, 'BAL': 1.11, 'BOS': 0.84, 'CHC': 1.03, 'CHW': 1.25, 'CWS': 1.25,
    'CIN': 1.27, 'CLE': 0.96, 'COL': 1.06, 'DET': 0.96, 'HOU': 1.10, 'KC': 0.83, 'LAA': 1.01, 'LAD': 1.11,
    'MIA': 0.85, 'MIL': 1.14, 'MIN': 0.94, 'NYM': 1.07, 'NYY': 1.20, 'OAK': 0.90, 'ATH': 0.90,
    'PHI': 1.18, 'PIT': 0.83, 'SD': 1.02, 'SEA': 1.00, 'SF': 0.75, 'STL': 0.86, 'TB': 0.96, 'TEX': 1.07, 'TOR': 1.09,
    'WAS': 1.00, 'WSH': 1.00
}
park_hr_percent_map_rhb = {
    'ARI': 1.00, 'AZ': 1.00, 'ATL': 0.93, 'BAL': 1.09, 'BOS': 0.90, 'CHC': 1.09, 'CHW': 1.26, 'CWS': 1.26,
    'CIN': 1.27, 'CLE': 0.91, 'COL': 1.05, 'DET': 0.96, 'HOU': 1.10, 'KC': 0.83, 'LAA': 1.01, 'LAD': 1.11,
    'MIA': 0.84, 'MIL': 1.12, 'MIN': 0.95, 'NYM': 1.11, 'NYY': 1.15, 'OAK': 0.91, 'ATH': 0.91,
    'PHI': 1.18, 'PIT': 0.80, 'SD': 1.02, 'SEA': 1.03, 'SF': 0.76, 'STL': 0.84, 'TB': 0.94, 'TEX': 1.06, 'TOR': 1.11,
    'WAS': 1.02, 'WSH': 1.02
}
park_hr_percent_map_lhb = {
    'ARI': 0.98, 'AZ': 0.98, 'ATL': 0.99, 'BAL': 1.13, 'BOS': 0.75, 'CHC': 0.93, 'CHW': 1.23, 'CWS': 1.23,
    'CIN': 1.29, 'CLE': 1.01, 'COL': 1.07, 'DET': 0.96, 'HOU': 1.09, 'KC': 0.81, 'LAA': 1.00, 'LAD': 1.12,
    'MIA': 0.87, 'MIL': 1.19, 'MIN': 0.91, 'NYM': 1.06, 'NYY': 1.28, 'OAK': 0.87, 'ATH': 0.87,
    'PHI': 1.19, 'PIT': 0.90, 'SD': 0.98, 'SEA': 0.96, 'SF': 0.73, 'STL': 0.90, 'TB': 0.99, 'TEX': 1.11, 'TOR': 1.05,
    'WAS': 0.96, 'WSH': 0.96
}
# ========== DEEP RESEARCH HR MULTIPLIERS: PITCHER SIDE ===============
park_hr_percent_map_pitcher_all = {
    'ARI': 0.98, 'AZ': 0.98, 'ATL': 0.95, 'BAL': 1.11, 'BOS': 0.84, 'CHC': 1.03, 'CHW': 1.25, 'CWS': 1.25,
    'CIN': 1.27, 'CLE': 0.96, 'COL': 1.06, 'DET': 0.96, 'HOU': 1.10, 'KC': 0.83, 'LAA': 1.01, 'LAD': 1.11,
    'MIA': 0.85, 'MIL': 1.14, 'MIN': 0.94, 'NYM': 1.07, 'NYY': 1.20, 'OAK': 0.90, 'ATH': 0.90,
    'PHI': 1.18, 'PIT': 0.83, 'SD': 1.02, 'SEA': 1.00, 'SF': 0.75, 'STL': 0.86, 'TB': 0.96, 'TEX': 1.07, 'TOR': 1.09,
    'WAS': 1.00, 'WSH': 1.00
}
park_hr_percent_map_rhp = {
    'ARI': 0.97, 'AZ': 0.97, 'ATL': 1.01, 'BAL': 1.16, 'BOS': 0.84, 'CHC': 1.02, 'CHW': 1.28, 'CWS': 1.28,
    'CIN': 1.27, 'CLE': 0.98, 'COL': 1.06, 'DET': 0.95, 'HOU': 1.11, 'KC': 0.84, 'LAA': 1.01, 'LAD': 1.11,
    'MIA': 0.84, 'MIL': 1.14, 'MIN': 0.96, 'NYM': 1.07, 'NYY': 1.24, 'OAK': 0.90, 'ATH': 0.90,
    'PHI': 1.19, 'PIT': 0.85, 'SD': 1.02, 'SEA': 1.01, 'SF': 0.73, 'STL': 0.84, 'TB': 0.97, 'TEX': 1.10, 'TOR': 1.11,
    'WAS': 1.03, 'WSH': 1.03
}
park_hr_percent_map_lhp = {
    'ARI': 0.99, 'AZ': 0.99, 'ATL': 0.79, 'BAL': 0.97, 'BOS': 0.83, 'CHC': 1.03, 'CHW': 1.18, 'CWS': 1.18,
    'CIN': 1.27, 'CLE': 0.89, 'COL': 1.05, 'DET': 0.97, 'HOU': 1.07, 'KC': 0.79, 'LAA': 1.01, 'LAD': 1.11,
    'MIA': 0.90, 'MIL': 1.14, 'MIN': 0.89, 'NYM': 1.05, 'NYY': 1.12, 'OAK': 0.89, 'ATH': 0.89,
    'PHI': 1.16, 'PIT': 0.78, 'SD': 1.02, 'SEA': 0.97, 'SF': 0.82, 'STL': 0.96, 'TB': 0.94, 'TEX': 1.01, 'TOR': 1.06,
    'WAS': 0.90, 'WSH': 0.90
}
# ========== YOUR UTILITY FUNCTIONS ==========

def dedup_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

def downcast_numeric(df):
    for col in df.select_dtypes(include=['float']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def parse_custom_weather_string_v2(s):
    if pd.isna(s):
        return pd.Series([np.nan]*7, index=['temp','wind_vector','wind_field_dir','wind_mph','humidity','condition','wind_dir_string'])
    s = str(s)
    temp_match = re.search(r'(\d{2,3})\s*[OI°]?\s', s)
    temp = int(temp_match.group(1)) if temp_match else np.nan
    wind_vector_match = re.search(r'\d{2,3}\s*([OI])\s', s)
    wind_vector = wind_vector_match.group(1) if wind_vector_match else np.nan
    wind_field_dir_match = re.search(r'\s([A-Z]{2})\s*\d', s)
    wind_field_dir = wind_field_dir_match.group(1) if wind_field_dir_match else np.nan
    mph = re.search(r'(\d{1,3})\s*-\s*(\d{1,3})', s)
    if mph:
        wind_mph = (int(mph.group(1)) + int(mph.group(2))) / 2
    else:
        mph = re.search(r'([1-9][0-9]?)\s*(?:mph)?', s)
        wind_mph = int(mph.group(1)) if mph else np.nan
    humidity_match = re.search(r'(\d{1,3})%', s)
    humidity = int(humidity_match.group(1)) if humidity_match else np.nan
    condition = "outdoor" if "outdoor" in s.lower() else ("indoor" if "indoor" in s.lower() else np.nan)
    wind_dir_string = f"{wind_vector} {wind_field_dir}".strip()
    return pd.Series([temp, wind_vector, wind_field_dir, wind_mph, humidity, condition, wind_dir_string],
                     index=['temp','wind_vector','wind_field_dir','wind_mph','humidity','condition','wind_dir_string'])

def calculate_spray_angle(hc_x, hc_y):
    try:
        if np.isnan(hc_x) or np.isnan(hc_y):
            return np.nan
        angle = np.degrees(np.arctan2(hc_x - 125.42, 199.53 - hc_y))
        return angle
    except Exception:
        return np.nan

def rolling_apply(series, window, func):
    if len(series) < 1:
        return np.nan
    result = series.rolling(window, min_periods=1).apply(func)
    return result.iloc[-1] if not result.empty else np.nan

# ================= ADVANCED ROLLING FEATURE FUNCTIONS ================
@st.cache_data(show_spinner=True, max_entries=8, ttl=7200)
def rolling_features_hr(df, id_col, date_col, windows, group_batter=True):
    records = []
    err_list = []
    for id_val, group in df.groupby(id_col, sort=False):
        group = group.sort_values(date_col)
        hr_outcome = group['hr_outcome'] if 'hr_outcome' in group.columns else pd.Series([0]*len(group))
        outs = group['outs_when_up'] if 'outs_when_up' in group.columns else pd.Series([0]*len(group))
        for w in windows:
            try:
                roll_hr = hr_outcome.rolling(w, min_periods=1).sum().iloc[-1]
                roll_pa = len(hr_outcome.iloc[-w:])
                pa_per_hr = roll_pa / roll_hr if roll_hr > 0 else np.nan
                hr_per_pa = roll_hr / roll_pa if roll_pa > 0 else 0
                if not group_batter:
                    roll_outs = outs.rolling(w, min_periods=1).sum().iloc[-1]
                    hr9 = (roll_hr / (roll_outs/3)) * 9 if roll_outs > 0 else np.nan
                    hr_percent = roll_hr / roll_pa if roll_pa > 0 else 0
                # Defensive handling for last_hr_days
                if hr_outcome.ne(0).any():
                    reversed_nonzero = hr_outcome[::-1].ne(0)
                    last_hr_idx = reversed_nonzero.idxmax()
                    if isinstance(last_hr_idx, (pd.Series, np.ndarray)):
                        last_hr_idx = last_hr_idx.iloc[0] if hasattr(last_hr_idx, 'iloc') else last_hr_idx[0]
                    try:
                        last_hr_date = group[date_col].loc[last_hr_idx]
                        last_hr_days = (pd.Timestamp(group[date_col].iloc[-1]) - pd.Timestamp(last_hr_date)).days
                    except Exception:
                        last_hr_days = np.nan
                else:
                    last_hr_days = np.nan
                row = {
                    id_col: id_val,
                    f"{'b_' if group_batter else 'p_'}rolling_hr_{w}": roll_hr,
                    f"{'b_' if group_batter else 'p_'}rolling_pa_{w}": roll_pa,
                    f"{'b_' if group_batter else 'p_'}pa_per_hr_{w}": pa_per_hr if group_batter else np.nan,
                    f"{'b_' if group_batter else 'p_'}hr_per_pa_{w}": hr_per_pa if group_batter else np.nan,
                    f"{'b_' if group_batter else 'p_'}time_since_hr_{w}": last_hr_days,
                }
                if not group_batter:
                    row[f"p_rolling_hr9_{w}"] = hr9
                    row[f"p_rolling_hr_percent_{w}"] = hr_percent
                records.append(row)
            except Exception as e:
                err_list.append((id_val, w, str(e)))
    df_out = pd.DataFrame(records)
    return df_out, err_list

@st.cache_data(show_spinner=True, max_entries=8, ttl=7200)
def fast_rolling_stats(df, id_col, date_col, windows, pitch_types=None, prefix=""):
    results = []
    err_list = []
    try:
        df = df.copy()
        if id_col in df.columns and date_col in df.columns:
            df = df.drop_duplicates(subset=[id_col, date_col], keep='last')
            df = df.sort_values([id_col, date_col])
        if 'launch_speed' in df.columns:
            df['launch_speed'] = pd.to_numeric(df['launch_speed'], errors='coerce')
        if 'launch_angle' in df.columns:
            df['launch_angle'] = pd.to_numeric(df['launch_angle'], errors='coerce')
        if 'hc_x' in df.columns and 'hc_y' in df.columns:
            df['pull'] = ((df['hc_x'] > 200) & (df['hc_x'] < 300)).astype(float)
            df['spray_angle'] = [calculate_spray_angle(x, y) for x, y in zip(df['hc_x'], df['hc_y'])]
        if 'hit_distance_sc' in df.columns:
            df['hard_hit'] = (df['launch_speed'] >= 95).astype(float)
        for name, group in df.groupby(id_col, sort=False):
            try:
                out_row = {}
                ls = group['launch_speed'] if 'launch_speed' in group.columns else None
                la = group['launch_angle'] if 'launch_angle' in group.columns else None
                hd = group['hit_distance_sc'] if 'hit_distance_sc' in group.columns else None
                pull = group['pull'] if 'pull' in group.columns else None
                hard = group['hard_hit'] if 'hard_hit' in group.columns else None
                spray = group['spray_angle'] if 'spray_angle' in group.columns else None
                slg = group['slg_numeric'] if 'slg_numeric' in group.columns else None
                for w in windows:
                    if ls is not None:
                        out_row[f"{prefix}avg_exit_velo_{w}"] = ls.rolling(w, min_periods=1).mean().iloc[-1]
                        out_row[f"{prefix}hard_hit_rate_{w}"] = ls.rolling(w, min_periods=1).apply(lambda x: np.mean(x >= 95)).iloc[-1]
                    if la is not None:
                        out_row[f"{prefix}fb_rate_{w}"] = la.rolling(w, min_periods=1).apply(lambda x: np.mean(x >= 25)).iloc[-1]
                        out_row[f"{prefix}sweet_spot_rate_{w}"] = la.rolling(w, min_periods=1).apply(lambda x: np.mean((x >= 8) & (x <= 32))).iloc[-1]
                    if spray is not None:
                        out_row[f"{prefix}spray_angle_avg_{w}"] = spray.rolling(w, min_periods=1).mean().iloc[-1]
                        out_row[f"{prefix}spray_angle_std_{w}"] = spray.rolling(w, min_periods=1).std().iloc[-1]
                    if ls is not None and la is not None:
                        barrels = ((ls >= 98) & (la >= 26) & (la <= 30)).astype(float)
                        out_row[f"{prefix}barrel_rate_{w}"] = barrels.rolling(w, min_periods=1).mean().iloc[-1]
                    if hd is not None:
                        out_row[f"{prefix}hit_dist_avg_{w}"] = hd.rolling(w, min_periods=1).mean().iloc[-1]
                    if pull is not None:
                        out_row[f"{prefix}pull_rate_{w}"] = pull.rolling(w, min_periods=1).mean().iloc[-1]
                    if hard is not None:
                        out_row[f"{prefix}hard_contact_rate_{w}"] = hard.rolling(w, min_periods=1).mean().iloc[-1]
                    if slg is not None:
                        out_row[f"{prefix}slg_{w}"] = slg.rolling(w, min_periods=1).mean().iloc[-1]
                if pitch_types is not None and "pitch_type" in group.columns:
                    for pt in pitch_types:
                        pt_group = group[group['pitch_type'] == pt]
                        for w in windows:
                            key = f"{prefix}{pt}_"
                            if not pt_group.empty:
                                if 'launch_speed' in pt_group.columns:
                                    out_row[f"{key}avg_exit_velo_{w}"] = pt_group['launch_speed'].rolling(w, min_periods=1).mean().iloc[-1]
                                    out_row[f"{key}hard_hit_rate_{w}"] = pt_group['launch_speed'].rolling(w, min_periods=1).apply(lambda x: np.mean(x >= 95)).iloc[-1]
                                if 'launch_angle' in pt_group.columns:
                                    out_row[f"{key}fb_rate_{w}"] = pt_group['launch_angle'].rolling(w, min_periods=1).apply(lambda x: np.mean(x >= 25)).iloc[-1]
                                    out_row[f"{key}sweet_spot_rate_{w}"] = pt_group['launch_angle'].rolling(w, min_periods=1).apply(lambda x: np.mean((x >= 8) & (x <= 32))).iloc[-1]
                                if 'spray_angle' in pt_group.columns:
                                    out_row[f"{key}spray_angle_avg_{w}"] = pt_group['spray_angle'].rolling(w, min_periods=1).mean().iloc[-1]
                                    out_row[f"{key}spray_angle_std_{w}"] = pt_group['spray_angle'].rolling(w, min_periods=1).std().iloc[-1]
                                if 'launch_speed' in pt_group.columns and 'launch_angle' in pt_group.columns:
                                    barrel_flags = pd.concat([
                                        pt_group['launch_speed'].reset_index(drop=True),
                                        pt_group['launch_angle'].reset_index(drop=True)
                                    ], axis=1).apply(lambda row: (row[0] >= 98) & (26 <= row[1] <= 30), axis=1)
                                    out_row[f"{key}barrel_rate_{w}"] = rolling_apply(barrel_flags, w, np.mean)
                            else:
                                for feat in ['avg_exit_velo', 'hard_hit_rate', 'barrel_rate', 'fb_rate', 'sweet_spot_rate', 'spray_angle_avg', 'spray_angle_std']:
                                    out_row[f"{key}{feat}_{w}"] = np.nan
                out_row[id_col] = name
                results.append(out_row)
            except Exception as e:
                err_list.append((name, str(e)))
    except Exception as e:
        err_list.append(("General groupby error", str(e)))
    df_out = pd.DataFrame(results)
    return df_out, err_list

# ==================== STREAMLIT APP MAIN ====================
st.set_page_config("MLB HR Analyzer", layout="wide")
st.header("Fetch Statcast Data & Generate Features")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=7))
with col2:
    end_date = st.date_input("End Date", value=datetime.today())
fetch_btn = st.button("Fetch Statcast, Feature Engineer, and Download", type="primary")
progress = st.empty()

if fetch_btn:
    import pybaseball
    from pybaseball import statcast

    progress.progress(3, "Fetching Statcast data...")
    df = statcast(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    progress.progress(10, "Loaded Statcast")
    st.write(f"Loaded {len(df)} raw Statcast events.")
    if len(df) == 0:
        st.error("No data! Try different dates.")
        st.stop()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce').dt.strftime('%Y-%m-%d')
    progress.progress(12, "Loaded and formatted Statcast columns.")

    # =================== STATCAST EVENT-LEVEL ENGINEERING ===================
    progress.progress(18, "Adding park/city/context and cleaning Statcast event data...")

    for col in ['batter_id', 'mlb_id', 'pitcher_id', 'team_code']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('.0','',regex=False).str.strip()

    # Map park/city/context using your provided context maps
    if 'home_team_code' in df.columns:
        df['team_code'] = df['home_team_code'].str.upper()
        df['park'] = df['home_team_code'].str.lower().str.replace(' ', '_')
    if 'home_team' in df.columns and 'park' not in df.columns:
        df['park'] = df['home_team'].str.lower().str.replace(' ', '_')
    if 'team_code' not in df.columns and 'park' in df.columns:
        park_to_team = {v:k for k,v in team_code_to_park.items()}
        df['team_code'] = df['park'].map(park_to_team).str.upper()
    df['team_code'] = df['team_code'].astype(str).str.upper()
    df['park'] = df['team_code'].map(team_code_to_park).str.lower()
    df['park_hr_rate'] = df['park'].map(park_hr_rate_map).fillna(1.0)
    df['park_altitude'] = df['park'].map(park_altitude_map).fillna(0)
    df['roof_status'] = df['park'].map(roof_status_map).fillna("open")
    df['city'] = df['team_code'].map(mlb_team_city_map).fillna("")

    # ====== Handedness fields ======
    if 'stand' in df.columns:
        df['batter_hand'] = df['stand']
    else:
        df['batter_hand'] = np.nan
    if 'p_throws' in df.columns:
        df['pitcher_hand'] = df['p_throws']
    elif 'p_throw' in df.columns:
        df['pitcher_hand'] = df['p_throw']
    else:
        df['pitcher_hand'] = np.nan

    # ========== DEEP RESEARCH: Park HR percent columns by hand/team ==========
    df['park_hr_pct_all'] = df['team_code'].map(park_hr_percent_map_all).fillna(1.0)
    df['park_hr_pct_rhb'] = df['team_code'].map(park_hr_percent_map_rhb).fillna(1.0)
    df['park_hr_pct_lhb'] = df['team_code'].map(park_hr_percent_map_lhb).fillna(1.0)
    df['park_hr_pct_hand'] = [
        park_hr_percent_map_rhb.get(team, 1.0) if str(stand).upper() == "R"
        else park_hr_percent_map_lhb.get(team, 1.0) if str(stand).upper() == "L"
        else park_hr_percent_map_all.get(team, 1.0)
        for team, stand in zip(df['team_code'], df['batter_hand'])
    ]

    # ================== SLG NUMERIC FOR ROLLING ==================
    slg_map = {'single':1, 'double':2, 'triple':3, 'home_run':4, 'homerun':4}
    if 'events' in df.columns:
        df['events_clean'] = df['events'].astype(str).str.lower().str.replace(' ', '')
    else:
        df['events_clean'] = ""
    df['slg_numeric'] = df['events_clean'].map(slg_map).astype(float).fillna(0)
    if 'hr_outcome' not in df.columns:
        df['hr_outcome'] = df['events_clean'].isin(['homerun', 'home_run']).astype(int)

    valid_events = [
        'single', 'double', 'triple', 'homerun', 'home_run', 'field_out',
        'force_out', 'grounded_into_double_play', 'fielders_choice_out',
        'pop_out', 'lineout', 'flyout', 'sac_fly', 'sac_fly_double_play'
    ]
    df = df[df['events_clean'].isin(valid_events)].copy()

    # ========== ADVANCED ROLLING HR/9, HR%, TIME-SINCE-HR FEATURES ==========
    progress.progress(22, "Computing rolling Statcast features (batter & pitcher, pitch type, deep windows, spray angle, rolling HR/9, HR%, time-since-HR)...")
    roll_windows = [3, 5, 7, 14, 20, 30, 60]
    main_pitch_types = ["ff", "sl", "cu", "ch", "si", "fc", "fs", "st", "sinker", "splitter", "sweeper"]

    for col in ['batter', 'batter_id']:
        if col in df.columns:
            df['batter_id'] = df[col]
    for col in ['pitcher', 'pitcher_id']:
        if col in df.columns:
            df['pitcher_id'] = df[col]

    # ========== BATTER ROLLING FEATURES ==========
    with st.spinner("Diagnostics: Batter Rolling Feature Calculation..."):
        batter_event, batter_event_errs = fast_rolling_stats(df, "batter_id", "game_date", roll_windows, main_pitch_types, prefix="b_")
        st.subheader("Diagnostics: Batter Fast Rolling Feature Errors")
        if len(batter_event_errs) == 0:
            st.success(f"Fast Rolling: Successfully processed {batter_event.shape[0]} groups with no errors.")
        else:
            st.warning(f"Fast Rolling errors for batters: {batter_event_errs}")

        batter_hr_rolling, batter_hr_rolling_errs = rolling_features_hr(df, "batter_id", "game_date", roll_windows, group_batter=True)
        st.subheader("Diagnostics: Batter Rolling HR Feature Errors")
        if len(batter_hr_rolling_errs) == 0:
            st.success(f"Rolling HR: Successfully processed {batter_hr_rolling.shape[0]} groups with no errors.")
        else:
            st.warning(f"Rolling HR errors for batters: {batter_hr_rolling_errs}")

    if not batter_event.empty and not batter_hr_rolling.empty:
        batter_merged = pd.merge(batter_event, batter_hr_rolling, how="left", on="batter_id")
    else:
        batter_merged = batter_event

    # ========== PITCHER ROLLING FEATURES ==========
    df_for_pitchers = df.copy()
    if 'batter_id' in df_for_pitchers.columns:
        df_for_pitchers = df_for_pitchers.drop(columns=['batter_id'])
    df_for_pitchers = df_for_pitchers.rename(columns={"pitcher_id": "batter_id"})
    with st.spinner("Diagnostics: Pitcher Rolling Feature Calculation..."):
        pitcher_event, pitcher_event_errs = fast_rolling_stats(df_for_pitchers, "batter_id", "game_date", roll_windows, main_pitch_types, prefix="p_")
        st.subheader("Diagnostics: Pitcher Fast Rolling Feature Errors")
        if len(pitcher_event_errs) == 0:
            st.success(f"Fast Rolling: Successfully processed {pitcher_event.shape[0]} groups with no errors.")
        else:
            st.warning(f"Fast Rolling errors for pitchers: {pitcher_event_errs}")

        pitcher_hr_rolling, pitcher_hr_rolling_errs = rolling_features_hr(df, "pitcher_id", "game_date", roll_windows, group_batter=False)
        st.subheader("Diagnostics: Pitcher Rolling HR Feature Errors")
        if len(pitcher_hr_rolling_errs) == 0:
            st.success(f"Rolling HR: Successfully processed {pitcher_hr_rolling.shape[0]} groups with no errors.")
        else:
            st.warning(f"Rolling HR errors for pitchers: {pitcher_hr_rolling_errs}")

    if not pitcher_event.empty and not pitcher_hr_rolling.empty:
        pitcher_merged = pd.merge(pitcher_event, pitcher_hr_rolling, how="left", left_on="batter_id", right_on="pitcher_id")
        pitcher_merged = pitcher_merged.drop(columns=["pitcher_id"], errors='ignore')
    else:
        pitcher_merged = pitcher_event

    # Merge all advanced features into event-level dataset
    with st.spinner("Merging advanced features..."):
        try:
            df = pd.merge(df, batter_merged.reset_index(drop=True), how="left", left_on="batter_id", right_on="batter_id")
            df = pd.merge(df, pitcher_merged.reset_index(drop=True), how="left", left_on="pitcher_id", right_on="batter_id", suffixes=('', '_pitcherstat'))
            if 'batter_id_pitcherstat' in df.columns:
                df = df.drop(columns=['batter_id_pitcherstat'])
            df = dedup_columns(df)
            st.success("Merged advanced rolling features successfully!")
        except Exception as merge_error:
            st.error(f"Error merging advanced features: {merge_error}")
            st.stop()

    # ====== Add park_hand_hr_rate to event-level ======
    if 'stand' in df.columns and 'park' in df.columns:
        df['park_hand_hr_rate'] = [
            park_hand_hr_rate_map.get(str(park).lower(), {}).get(str(stand).upper(), 1.0)
            for park, stand in zip(df['park'], df['stand'])
        ]
    else:
        df['park_hand_hr_rate'] = 1.0

    # ====== DOWNCAST NUMERICS FOR RAM OPTIMIZATION ======
    df = downcast_numeric(df)
    progress.progress(80, "Event-level feature engineering/merges complete.")

    # =================== DIAGNOSTICS: event-level output preview ===================
    st.write("Event-level dataframe sample:", df.head(20))
    st.markdown("#### Download Event-Level Feature CSV / Parquet:")
    st.dataframe(df.head(20), use_container_width=True)
    st.download_button(
        "⬇️ Download Event-Level CSV",
        data=df.to_csv(index=False),
        file_name="event_level_hr_features.csv",
        key="download_event_csv"
    )
    event_parquet = io.BytesIO()
    df.to_parquet(event_parquet, index=False)
    st.download_button(
        "⬇️ Download Event-Level Parquet",
        data=event_parquet.getvalue(),
        file_name="event_level_hr_features.parquet",
        mime="application/octet-stream",
        key="download_event_parquet"
    )

    st.success("Event-level file(s) and debug outputs ready.")
    progress.progress(100, "All complete.")

    # ---- RAM Cleanup ----
    del df, batter_event, pitcher_event, batter_merged, pitcher_merged, batter_hr_rolling, pitcher_hr_rolling
    gc.collect()
