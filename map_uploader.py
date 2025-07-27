import streamlit as st
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import pandas as pd

# ----------------- Snowflake Connection -----------------
conn = snowflake.connector.connect(
    user=st.secrets["snowflake"]["user"],
    password=st.secrets["snowflake"]["password"],
    account=st.secrets["snowflake"]["account"],
    warehouse=st.secrets["snowflake"]["warehouse"],
    database=st.secrets["snowflake"]["database"],
    schema=st.secrets["snowflake"]["schema"]
)
cur = conn.cursor()

# Use schema explicitly (optional but safer)
schema_name = st.secrets["snowflake"]["schema"].upper()
cur.execute(f"USE SCHEMA {schema_name}")
conn.commit()

# === Your original dicts remain unchanged === #

# Flatten nested dict (park_hand_hr_rate_map)
park_hand_hr_rows = []
for park, hands in park_hand_hr_rate_map.items():
    for hand, rate in hands.items():
        park_hand_hr_rows.append({'park': park, 'hand': hand, 'hr_rate': rate})

# Convert all dicts to dataframes (keys lower case OK, column names remain same)
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

# Table names uppercase!
CREATE_TABLE_STATEMENTS = {
    'PARK_HR_RATE_MAP': f"""
        CREATE OR REPLACE TABLE {schema_name}.PARK_HR_RATE_MAP (
            PARK VARCHAR PRIMARY KEY,
            PARK_HR_RATE FLOAT
        )
    """,
    'PARK_ALTITUDE_MAP': f"""
        CREATE OR REPLACE TABLE {schema_name}.PARK_ALTITUDE_MAP (
            PARK VARCHAR PRIMARY KEY,
            PARK_ALTITUDE INTEGER
        )
    """,
    'ROOF_STATUS_MAP': f"""
        CREATE OR REPLACE TABLE {schema_name}.ROOF_STATUS_MAP (
            PARK VARCHAR PRIMARY KEY,
            ROOF_STATUS VARCHAR
        )
    """,
    'TEAM_CODE_TO_PARK': f"""
        CREATE OR REPLACE TABLE {schema_name}.TEAM_CODE_TO_PARK (
            TEAM_CODE VARCHAR PRIMARY KEY,
            PARK VARCHAR
        )
    """,
    'MLB_TEAM_CITY_MAP': f"""
        CREATE OR REPLACE TABLE {schema_name}.MLB_TEAM_CITY_MAP (
            TEAM_CODE VARCHAR PRIMARY KEY,
            CITY VARCHAR
        )
    """,
    'PARK_HAND_HR_RATE_MAP': f"""
        CREATE OR REPLACE TABLE {schema_name}.PARK_HAND_HR_RATE_MAP (
            PARK VARCHAR,
            HAND VARCHAR,
            HR_RATE FLOAT
        )
    """,
    'PARK_HR_PERCENT_MAP_ALL': f"""
        CREATE OR REPLACE TABLE {schema_name}.PARK_HR_PERCENT_MAP_ALL (
            TEAM_CODE VARCHAR PRIMARY KEY,
            HR_PERCENT_ALL FLOAT
        )
    """,
    'PARK_HR_PERCENT_MAP_RHB': f"""
        CREATE OR REPLACE TABLE {schema_name}.PARK_HR_PERCENT_MAP_RHB (
            TEAM_CODE VARCHAR PRIMARY KEY,
            HR_PERCENT_RHB FLOAT
        )
    """,
    'PARK_HR_PERCENT_MAP_LHB': f"""
        CREATE OR REPLACE TABLE {schema_name}.PARK_HR_PERCENT_MAP_LHB (
            TEAM_CODE VARCHAR PRIMARY KEY,
            HR_PERCENT_LHB FLOAT
        )
    """,
    'PARK_HR_PERCENT_MAP_PITCHER_ALL': f"""
        CREATE OR REPLACE TABLE {schema_name}.PARK_HR_PERCENT_MAP_PITCHER_ALL (
            TEAM_CODE VARCHAR PRIMARY KEY,
            HR_PERCENT_PITCHER_ALL FLOAT
        )
    """,
    'PARK_HR_PERCENT_MAP_RHP': f"""
        CREATE OR REPLACE TABLE {schema_name}.PARK_HR_PERCENT_MAP_RHP (
            TEAM_CODE VARCHAR PRIMARY KEY,
            HR_PERCENT_RHP FLOAT
        )
    """,
    'PARK_HR_PERCENT_MAP_LHP': f"""
        CREATE OR REPLACE TABLE {schema_name}.PARK_HR_PERCENT_MAP_LHP (
            TEAM_CODE VARCHAR PRIMARY KEY,
            HR_PERCENT_LHP FLOAT
        )
    """,
}

# List tables and their corresponding dataframes (use uppercase names)
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
    print(f"Creating table {table_name}...")
    cur.execute(CREATE_TABLE_STATEMENTS[table_name])
    conn.commit()  # important to commit the create table before loading data
    
    full_table_name = f'{schema_name}.{table_name}'
    print(f"Uploading data to {full_table_name} ({len(df)} rows)...")
    
    success, nchunks, nrows, _ = write_pandas(conn, df, full_table_name)
    if success:
        print(f"Inserted {nrows} rows into {full_table_name}.\n")
    else:
        print(f"Failed to insert rows into {full_table_name}.\n")

# Run uploads
for table, data in tables_and_dfs:
    create_and_upload(table, data)

cur.close()
conn.close()
print("All mapping tables created and data uploaded successfully!")
