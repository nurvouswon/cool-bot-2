import streamlit as st
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import pandas as pd

# -------------- Connect & Set Schema --------------
schema_name = st.secrets["snowflake"]["schema"].upper()

conn = snowflake.connector.connect(
    user=st.secrets["snowflake"]["user"],
    password=st.secrets["snowflake"]["password"],
    account=st.secrets["snowflake"]["account"],
    warehouse=st.secrets["snowflake"]["warehouse"],
    database=st.secrets["snowflake"]["database"],
    schema=st.secrets["snowflake"]["schema"],  # set schema here, casing doesn't matter here
    autocommit=False,  # We'll manually commit DDL
)
cur = conn.cursor()

# Explicitly use schema to avoid ambiguity, uppercase as Snowflake expects
cur.execute(f"USE SCHEMA {schema_name}")
conn.commit()

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
    'NYY': 'New York', 'OAK': 'West Sacramento', 'ATH': 'West Sacramento', 'PHI': 'Philadelphia', 'PIT': 'Pittsburgh',
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

# Flatten nested dict
park_hand_hr_rows = []
for park, hands in park_hand_hr_rate_map.items():
    for hand, rate in hands.items():
        park_hand_hr_rows.append({'park': park, 'hand': hand, 'hr_rate': rate})

# Convert dicts to DataFrames
df_park_hr_rate = pd.DataFrame(list(park_hr_rate_map.items()), columns=['park', 'park_hr_rate'])
df_park_altitude = pd.DataFrame(list(park_altitude_map.items()), columns=['park', 'park_altitude'])
df_roof_status = pd.DataFrame(list(roof_status_map.items()), columns=['park', 'roof_status'])
df_team_code_to_park = pd.DataFrame(list(team_code_to_park.items()), columns=['team_code', 'park'])
df_mlb_team_city = pd.DataFrame(list(mlb_team_city_map.items()), columns=['team_code', 'city'])
df_park_hand_hr_rate = pd.DataFrame(park_hand_hr_rows)

df_park_hr_percent_map_all = pd.DataFrame(list(park_hr_percent_map_all.items()), columns=['team_code', 'hr_percent_all'])
df_park_hr_percent_map_rhb = pd.DataFrame(list(park_hr_percent_map_rhb.items()), columns=['team_code', 'hr_percent_rhb'])
df_park_hr_percent_map_lhb = pd.DataFrame(list(park_hr_percent_map_lhb.items()), columns=['team_code', 'hr_percent_lhb'])

df_park_hr_percent_map_pitcher_all = pd.DataFrame(list(park_hr_percent_map_pitcher_all.items()), columns=['team_code', 'hr_percent_pitcher_all'])
df_park_hr_percent_map_rhp = pd.DataFrame(list(park_hr_percent_map_rhp.items()), columns=['team_code', 'hr_percent_rhp'])
df_park_hr_percent_map_lhp = pd.DataFrame(list(park_hr_percent_map_lhp.items()), columns=['team_code', 'hr_percent_lhp'])

# === Creation SQL: no schema prefix, uppercase names ===
CREATE_TABLE_STATEMENTS = {
    'PARK_HR_RATE_MAP': """
        CREATE OR REPLACE TABLE PARK_HR_RATE_MAP (
            PARK VARCHAR PRIMARY KEY,
            PARK_HR_RATE FLOAT
        )
    """,
    'PARK_ALTITUDE_MAP': """
        CREATE OR REPLACE TABLE PARK_ALTITUDE_MAP (
            PARK VARCHAR PRIMARY KEY,
            PARK_ALTITUDE INTEGER
        )
    """,
    'ROOF_STATUS_MAP': """
        CREATE OR REPLACE TABLE ROOF_STATUS_MAP (
            PARK VARCHAR PRIMARY KEY,
            ROOF_STATUS VARCHAR
        )
    """,
    'TEAM_CODE_TO_PARK': """
        CREATE OR REPLACE TABLE TEAM_CODE_TO_PARK (
            TEAM_CODE VARCHAR PRIMARY KEY,
            PARK VARCHAR
        )
    """,
    'MLB_TEAM_CITY_MAP': """
        CREATE OR REPLACE TABLE MLB_TEAM_CITY_MAP (
            TEAM_CODE VARCHAR PRIMARY KEY,
            CITY VARCHAR
        )
    """,
    'PARK_HAND_HR_RATE_MAP': """
        CREATE OR REPLACE TABLE PARK_HAND_HR_RATE_MAP (
            PARK VARCHAR,
            HAND VARCHAR,
            HR_RATE FLOAT
        )
    """,
    'PARK_HR_PERCENT_MAP_ALL': """
        CREATE OR REPLACE TABLE PARK_HR_PERCENT_MAP_ALL (
            TEAM_CODE VARCHAR PRIMARY KEY,
            HR_PERCENT_ALL FLOAT
        )
    """,
    'PARK_HR_PERCENT_MAP_RHB': """
        CREATE OR REPLACE TABLE PARK_HR_PERCENT_MAP_RHB (
            TEAM_CODE VARCHAR PRIMARY KEY,
            HR_PERCENT_RHB FLOAT
        )
    """,
    'PARK_HR_PERCENT_MAP_LHB': """
        CREATE OR REPLACE TABLE PARK_HR_PERCENT_MAP_LHB (
            TEAM_CODE VARCHAR PRIMARY KEY,
            HR_PERCENT_LHB FLOAT
        )
    """,
    'PARK_HR_PERCENT_MAP_PITCHER_ALL': """
        CREATE OR REPLACE TABLE PARK_HR_PERCENT_MAP_PITCHER_ALL (
            TEAM_CODE VARCHAR PRIMARY KEY,
            HR_PERCENT_PITCHER_ALL FLOAT
        )
    """,
    'PARK_HR_PERCENT_MAP_RHP': """
        CREATE OR REPLACE TABLE PARK_HR_PERCENT_MAP_RHP (
            TEAM_CODE VARCHAR PRIMARY KEY,
            HR_PERCENT_RHP FLOAT
        )
    """,
    'PARK_HR_PERCENT_MAP_LHP': """
        CREATE OR REPLACE TABLE PARK_HR_PERCENT_MAP_LHP (
            TEAM_CODE VARCHAR PRIMARY KEY,
            HR_PERCENT_LHP FLOAT
        )
    """,
}

# List of (table_name, dataframe) tuples with uppercase table names
tables_and_dfs = [
    ('PARK_HR_RATE_MAP', df_park_hr_rate),
    ('PARK_ALTITUDE_MAP', df_park_altitude),
    ('ROOF_STATUS_MAP', df_roof_status),
    ('TEAM_CODE_TO_PARK', df_team_code_to_park),
    ('MLB_TEAM_CITY_MAP', df_mlb_team_city),
    ('PARK_HAND_HR_RATE_MAP', df_park_hand_hr_rate),
    ('PARK_HR_PERCENT_MAP_ALL', df_park_hr_percent_map_all),
    ('PARK_HR_PERCENT_MAP_RHB', df_park_hr_percent_map_rhb),
    ('PARK_HR_PERCENT_MAP_LHB', df_park_hr_percent_map_lhb),
    ('PARK_HR_PERCENT_MAP_PITCHER_ALL', df_park_hr_percent_map_pitcher_all),
    ('PARK_HR_PERCENT_MAP_RHP', df_park_hr_percent_map_rhp),
    ('PARK_HR_PERCENT_MAP_LHP', df_park_hr_percent_map_lhp),
]

def create_and_upload(table_name, df):
    print(f"Creating table {table_name} ...")
    cur.execute(CREATE_TABLE_STATEMENTS[table_name])
    conn.commit()
    
    print(f"Uploading data to {table_name} ({len(df)} rows) ...")
    success, nchunks, nrows, _ = write_pandas(conn, df, table_name)
    
    if success:
        print(f"Inserted {nrows} rows into {table_name}.\n")
    else:
        print(f"Failed to insert rows into {table_name}.\n")

# Run uploads for all tables
for table_name, df in tables_and_dfs:
    create_and_upload(table_name, df)

cur.close()
conn.close()
print("All mapping tables created and data uploaded successfully!")
