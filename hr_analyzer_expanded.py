import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config("🟦 Generate Today's Event-Level CSV", layout="wide")
st.title("🟦 Generate Today's Event-Level CSV (All Rolling Stats: Batters + Pitchers, 1 Row Each)")

st.markdown("""
**Instructions:**
- Upload Today's Lineups/Matchups CSV (must have `batter_id` or `mlb_id`, `player_name`, etc).
- Upload Historical Event-Level CSV (must have `batter_id`, `pitcher_id`, `game_date`, plus raw stat columns like launch_speed, launch_angle, events, etc).
- Outputs: ONE row per batter (and per pitcher, if you want) with all rolling/stat features, windows: 3, 5, 7, 14, 20.
- Weather/park columns auto-merged if available.
- All previous debugging and dedupe/typing logic is retained!
""")

# ---- FILE UPLOAD ----
today_file = st.file_uploader("Upload Today's Matchups/Lineups CSV", type=["csv"], key="today_csv")
hist_file = st.file_uploader("Upload Historical Event-Level CSV", type=["csv"], key="hist_csv")

# ---- Output columns (can expand as needed) ----
output_columns = [
    # Batters, matchup meta
    "team_code","game_date","game_number","mlb_id","player_name","batting_order","position","weather","temp","wind_mph","wind_dir","condition",
    "stadium","city","batter_id","p_throws",
    # Rolling stats - main
    "hard_hit_rate_20","sweet_spot_rate_20","barrel_rate_20","fb_rate_20","avg_exit_velo_20",
    # All main HR/rolling windows
    "park_hand_hr_7","park_hand_hr_14","park_hand_hr_30",
    "b_vsp_hand_hr_3","p_vsb_hand_hr_3","b_vsp_hand_hr_5","p_vsb_hand_hr_5","b_vsp_hand_hr_7","p_vsb_hand_hr_7",
    "b_vsp_hand_hr_14","p_vsb_hand_hr_14",
    "b_pitchtype_hr_3","p_pitchtype_hr_3","b_pitchtype_hr_5","p_pitchtype_hr_5","b_pitchtype_hr_7","p_pitchtype_hr_7",
    "b_pitchtype_hr_14","p_pitchtype_hr_14",
    # Full 3/5/7/14/20 rolling windows for power/quality
    "b_launch_speed_3","b_launch_speed_5","b_launch_speed_7","b_launch_speed_14","b_launch_speed_20",
    "b_launch_angle_3","b_launch_angle_5","b_launch_angle_7","b_launch_angle_14","b_launch_angle_20",
    "b_hit_distance_sc_3","b_hit_distance_sc_5","b_hit_distance_sc_7","b_hit_distance_sc_14","b_hit_distance_sc_20",
    "b_woba_value_3","b_woba_value_5","b_woba_value_7","b_woba_value_14","b_woba_value_20",
    "b_release_speed_3","b_release_speed_5","b_release_speed_7","b_release_speed_14","b_release_speed_20",
    "b_release_spin_rate_3","b_release_spin_rate_5","b_release_spin_rate_7","b_release_spin_rate_14","b_release_spin_rate_20",
    "b_spin_axis_3","b_spin_axis_5","b_spin_axis_7","b_spin_axis_14","b_spin_axis_20",
    "b_pfx_x_3","b_pfx_x_5","b_pfx_x_7","b_pfx_x_14","b_pfx_x_20",
    "b_pfx_z_3","b_pfx_z_5","b_pfx_z_7","b_pfx_z_14","b_pfx_z_20",
    # Pitcher windows (mirrored)
    "p_launch_speed_3","p_launch_speed_5","p_launch_speed_7","p_launch_speed_14","p_launch_speed_20",
    "p_launch_angle_3","p_launch_angle_5","p_launch_angle_7","p_launch_angle_14","p_launch_angle_20",
    "p_hit_distance_sc_3","p_hit_distance_sc_5","p_hit_distance_sc_7","p_hit_distance_sc_14","p_hit_distance_sc_20",
    "p_woba_value_3","p_woba_value_5","p_woba_value_7","p_woba_value_14","p_woba_value_20",
    "p_release_speed_3","p_release_speed_5","p_release_speed_7","p_release_speed_14","p_release_speed_20",
    "p_release_spin_rate_3","p_release_spin_rate_5","p_release_spin_rate_7","p_release_spin_rate_14","p_release_spin_rate_20",
    "p_spin_axis_3","p_spin_axis_5","p_spin_axis_7","p_spin_axis_14","p_spin_axis_20",
    "p_pfx_x_3","p_pfx_x_5","p_pfx_x_7","p_pfx_x_14","p_pfx_x_20",
    "p_pfx_z_3","p_pfx_z_5","p_pfx_z_7","p_pfx_z_14","p_pfx_z_20",
    # Park/weather meta again (for stacking)
    "park","temp","wind_mph","wind_dir","humidity","condition"
]

# ---- MAIN LOGIC ----
if today_file and hist_file:
    df_today = pd.read_csv(today_file)
    df_hist = pd.read_csv(hist_file)
    st.success(f"Today's Matchups file loaded: {df_today.shape[0]} rows, {df_today.shape[1]} columns.")
    st.success(f"Historical Event-Level file loaded: {df_hist.shape[0]} rows, {df_hist.shape[1]} columns.")

    # ---- Clean/standardize column names ----
    df_today.columns = [str(c).strip().lower().replace(" ", "_") for c in df_today.columns]
    df_hist.columns = [str(c).strip().lower().replace(" ", "_") for c in df_hist.columns]

    # ---- Get batter_id/pitcher_id ----
    for id_col in ['batter_id', 'mlb_id']:
        if id_col in df_today.columns:
            df_today['batter_id'] = df_today[id_col].astype(str).str.strip().str.replace('.0', '', regex=False)
    if 'batter_id' in df_hist.columns:
        df_hist['batter_id'] = df_hist['batter_id'].astype(str).str.strip().str.replace('.0', '', regex=False)
    if 'pitcher_id' in df_hist.columns:
        df_hist['pitcher_id'] = df_hist['pitcher_id'].astype(str).str.strip().str.replace('.0', '', regex=False)

    # ---- Deduplicate today's batters ----
    df_today = df_today.drop_duplicates(subset=["batter_id"])

    # ---- Parse today's weather/park fields ----
    # Try to extract temp/wind/condition if "weather" field is present
    if "weather" in df_today.columns:
        weather_str = df_today["weather"].astype(str).str.lower()
        df_today["temp"] = weather_str.str.extract(r'(\d{2,3})')[0]
        df_today["temp"] = pd.to_numeric(df_today["temp"], errors="coerce")
        df_today["wind_mph"] = weather_str.str.extract(r'(\d+)\s*mph')[0]
        df_today["wind_mph"] = pd.to_numeric(df_today["wind_mph"], errors="coerce")
        # Try to get wind direction (single letters like N, S, etc)
        df_today["wind_dir"] = weather_str.str.extract(r'([nswecf]{1,2})\s', flags=re.I, expand=False)
        # Condition fallback
        df_today["condition"] = weather_str.str.extract(r'(indoor|outdoor|open|dome|cloudy|rain|sun)', expand=False)

    # ---- Function for custom rolling stats (returns a dict of {window: value}) ----
    def rolling_stats(group, col, stat="mean", windows=[3,5,7,14,20]):
        out = {}
        arr = group[col].dropna()
        for w in windows:
            key = f"{col}_{w}"
            if len(arr) >= w:
                if stat == "mean":
                    out[key] = arr[-w:].mean()
                elif stat == "sum":
                    out[key] = arr[-w:].sum()
                elif stat == "max":
                    out[key] = arr[-w:].max()
            else:
                out[key] = np.nan
        return out

    # ---- BARREL/HH%/FB/SWEETSPOT for each batter, each window ----
    windows = [3,5,7,14,20]

    batter_rows = []
    for batter_id, bgroup in df_hist.sort_values("game_date").groupby("batter_id"):
        row = {"batter_id": batter_id}
        # Pull from latest today's meta if exists
        today_row = df_today[df_today["batter_id"] == batter_id]
        for col in ["team_code","game_date","game_number","mlb_id","player_name","batting_order","position",
                    "weather","temp","wind_mph","wind_dir","condition","stadium","city","p_throws"]:
            if not today_row.empty and col in today_row.columns:
                row[col] = today_row.iloc[0][col]
            else:
                row[col] = np.nan
        # Rolling rates (batted ball events only, not all events)
        is_bbe = bgroup["launch_speed"].notnull()
        # Hard hit: launch_speed >=95
        for w in windows:
            last = bgroup[is_bbe].tail(w)
            row[f"hard_hit_rate_{w}"] = np.nan if last.empty else (last["launch_speed"]>=95).mean()
            row[f"barrel_rate_{w}"] = np.nan if last.empty else (
                ((last["launch_speed"]>=98) & (last["launch_angle"].between(26,30))).mean()
            )
            row[f"fb_rate_{w}"] = np.nan if last.empty else (last["launch_angle"]>=25).mean()
            row[f"sweet_spot_rate_{w}"] = np.nan if last.empty else (last["launch_angle"].between(8,32)).mean()
            row[f"avg_exit_velo_{w}"] = np.nan if last.empty else last["launch_speed"].mean()
        # Statcast/stat rolling metrics (all windows)
        for col in [
            "launch_speed", "launch_angle", "hit_distance_sc", "woba_value",
            "release_speed", "release_spin_rate", "spin_axis",
            "pfx_x", "pfx_z"
        ]:
            row.update(rolling_stats(bgroup, col, "mean", windows))
        # Existing rolling HR/park/pitchtype
        for stat in [
            "park_hand_hr_7","park_hand_hr_14","park_hand_hr_30",
            "b_vsp_hand_hr_3","p_vsb_hand_hr_3","b_vsp_hand_hr_5","p_vsb_hand_hr_5","b_vsp_hand_hr_7","p_vsb_hand_hr_7",
            "b_vsp_hand_hr_14","p_vsb_hand_hr_14","b_pitchtype_hr_3","p_pitchtype_hr_3","b_pitchtype_hr_5","p_pitchtype_hr_5",
            "b_pitchtype_hr_7","p_pitchtype_hr_7","b_pitchtype_hr_14","p_pitchtype_hr_14"
        ]:
            if stat in bgroup.columns:
                row[stat] = bgroup[stat].dropna().iloc[-1] if not bgroup[stat].dropna().empty else np.nan
            else:
                row[stat] = np.nan
        batter_rows.append(row)

    batters_df = pd.DataFrame(batter_rows)

    # ---- Pitcher version ----
    pitcher_rows = []
    if "pitcher_id" in df_hist.columns:
        for pitcher_id, pgroup in df_hist.sort_values("game_date").groupby("pitcher_id"):
            row = {"pitcher_id": pitcher_id}
            # Try to pull any "today" info if you want (future feature)
            # Rolling stat windows for pitcher metrics
            is_bbe = pgroup["launch_speed"].notnull()
            for w in windows:
                last = pgroup[is_bbe].tail(w)
                row[f"p_hard_hit_rate_{w}"] = np.nan if last.empty else (last["launch_speed"]>=95).mean()
                row[f"p_barrel_rate_{w}"] = np.nan if last.empty else (
                    ((last["launch_speed"]>=98) & (last["launch_angle"].between(26,30))).mean()
                )
                row[f"p_fb_rate_{w}"] = np.nan if last.empty else (last["launch_angle"]>=25).mean()
                row[f"p_sweet_spot_rate_{w}"] = np.nan if last.empty else (last["launch_angle"].between(8,32)).mean()
                row[f"p_avg_exit_velo_{w}"] = np.nan if last.empty else last["launch_speed"].mean()
            # Statcast windows
            for col in [
                "launch_speed", "launch_angle", "hit_distance_sc", "woba_value",
                "release_speed", "release_spin_rate", "spin_axis", "pfx_x", "pfx_z"
            ]:
                row.update(rolling_stats(pgroup, col, "mean", windows))
            # HR/park/pitchtype
            for stat in [
                "park_hand_hr_7","park_hand_hr_14","park_hand_hr_30",
                "b_vsp_hand_hr_3","p_vsb_hand_hr_3","b_vsp_hand_hr_5","p_vsb_hand_hr_5","b_vsp_hand_hr_7","p_vsb_hand_hr_7",
                "b_vsp_hand_hr_14","p_vsb_hand_hr_14","b_pitchtype_hr_3","p_pitchtype_hr_3","b_pitchtype_hr_5","p_pitchtype_hr_5",
                "b_pitchtype_hr_7","p_pitchtype_hr_7","b_pitchtype_hr_14","p_pitchtype_hr_14"
            ]:
                if stat in pgroup.columns:
                    row[stat] = pgroup[stat].dropna().iloc[-1] if not pgroup[stat].dropna().empty else np.nan
                else:
                    row[stat] = np.nan
            pitcher_rows.append(row)
        pitchers_df = pd.DataFrame(pitcher_rows)
    else:
        pitchers_df = pd.DataFrame()

    # ---- Merge batters with today's meta ----
    merged = df_today.merge(batters_df, on="batter_id", how="left", suffixes=('', '_hist'))
    merged = merged.drop_duplicates(subset=['batter_id'], keep='first')

    # ---- Reindex columns to final order (add missing as NaN) ----
    for col in output_columns:
        if col not in merged.columns:
            merged[col] = np.nan
    merged = merged.reindex(columns=output_columns)

    st.success(f"🟢 Generated file: {merged.shape[0]} unique batters, {merged.shape[1]} columns (features).")

    # ---- Preview output ----
    st.dataframe(merged.head(10))
    st.markdown("#### Batters/All Feature Null Report:")
    st.code(merged.isnull().sum().sort_values(ascending=False).to_string())

    # ---- Download button ----
    st.download_button(
        "⬇️ Download Today's Event-Level CSV (All Rolling Windows, Batters+Pitchers)",
        data=merged.to_csv(index=False),
        file_name="event_level_today_full.csv"
    )

    # ---- Optionally, download pitcher rolling stats (if desired) ----
    if not pitchers_df.empty:
        st.markdown("#### Download Pitcher Rolling Stats:")
        st.download_button(
            "⬇️ Download Pitcher Rolling Stats",
            data=pitchers_df.to_csv(index=False),
            file_name="event_level_today_pitchers.csv"
        )

else:
    st.info("Please upload BOTH today's matchups/lineups and historical event-level CSV.")
