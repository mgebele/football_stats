import glob
import pandas as pd
import numpy as np
import re
import plotly_express as px
import datetime
import warnings
import os
import json
import traceback
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(layout="wide")
pd.options.display.float_format = "{:,.1f}".format
warnings.filterwarnings('ignore')

# TEAMNAMES value in teamnamedict must match the htdatan teamname!
global teamnamedict
# C:\Users\mg\JupyterLabDir\Rest\Pr Winning\teamnamedict_streamlit.json
with open('scripts/streamlit_app/teamnamedict_streamlit.json') as f:
    teamnamedict = json.load(f)

global widthfig
widthfig = 700
heightfig = 600

title_x=.2 # alignment of title of plotly diagrams. 0 = left, 1 = right

# get all the gamestatistics from in dropdown specified league and season
# setup the database connection.  There's no need to setup cursors with pandas psql.
tables = list(glob.glob("data/htdatan/*"))

# take only the 0 part of the every list entry
global saissons
saissons = []

for x in range(0, len(tables)):
    saissons.append(Path(tables[x]).parts[2].split("_24102021.csv")[0])


cleaned_names_saissons = []
for saisson in saissons:
    saisson = saisson.replace("_", " ")
    saisson = saisson.replace(".csv", "")
    # saison = saison.strip()
    cleaned_names_saissons.append(saisson)

# map league shortcuts to real names:
shortcut_league_dict = {
    "ll": "La-Liga",
    "pl": "Premier-League",
    "sa": "Serie-A",
    "pl": "Premier-League",
    "l1": "League-1",
    "b": "Bundesliga",
}
# list(map(cleaned_names_saissons, shortcut_league_dict) )
cleaned_names_saissons = [e.replace(
    key, val) for e in cleaned_names_saissons for key, val in shortcut_league_dict.items() if key in e]

# sort list by int substring
cleaned_names_saissons = sorted(cleaned_names_saissons, key=lambda x: int(
    x.split(" ")[-1]), reverse=True)

global saison
saison = st.sidebar.selectbox("Saison", list(
    cleaned_names_saissons), 0)
print(saison)

def find_key(input_dict, value):
    for key, val in input_dict.items():
        if val == value:
            return key
    return "None"

saison = "{}_{}".format(find_key(shortcut_league_dict, saison.split(" ")[0]),
                        saison.split(" ")[1]
                        )

try:
    df_complete_saison = pd.read_csv(
        "data/htdatan/"+saison+".csv", index_col=0, encoding='utf-8')
except:
    df_complete_saison = pd.read_csv(
        "data/htdatan/"+saison+"_24102021.csv", index_col=0, encoding='utf-8')

df_complete_saison = df_complete_saison.replace(teamnamedict)
dfallteamnamesl = df_complete_saison.H_Teamnames.unique()




# Schritt 3: Zuweisen von Punkten basierend auf dem Halbzeit-Ergebnis
def assign_points(row, team_type):
    if team_type == 'H': # and row['BP-H'] > 59
        if row['Halbzeit_Ergebnis_H'] == 'Gewinn':
            return 1
        elif row['Halbzeit_Ergebnis_H'] == 'Verlust':
            return -1
        else:
            return 0
    elif team_type == 'A': #and row['BP-A'] > 59
        if row['Halbzeit_Ergebnis_A'] == 'Gewinn':
            return 1
        elif row['Halbzeit_Ergebnis_A'] == 'Verlust':
            return -1
        else:
            return 0
    return 0

def post_process_table(df_filtered):
    df_filtered['Punkte_H'] = df_filtered.apply(lambda row: assign_points(row, 'H'), axis=1)
    df_filtered['Punkte_A'] = df_filtered.apply(lambda row: assign_points(row, 'A'), axis=1)

    # Schritt 4: Aggregieren der Punkte für jedes Team
    home_points = df_filtered.groupby('Home')['Punkte_H'].sum()
    away_points = df_filtered.groupby('Opponent')['Punkte_A'].sum()

    # Zusammenführen der Punkte für Heim- und Auswärtsteams
    total_points = home_points.add(away_points, fill_value=0).reset_index()
    total_points.columns = ['Team', 'Gesamtpunkte']

    # Zählen der Halbzeiten für jedes Team
    home_counts = df_filtered['Home'].value_counts()
    away_counts = df_filtered['Opponent'].value_counts()
    total_counts = home_counts.add(away_counts, fill_value=0)

    # Summieren der xG-Werte für und gegen jedes Team
    home_xg_for = df_filtered.groupby('Home')['xg_halftime'].sum()
    away_xg_for = df_filtered.groupby('Opponent')['Axg_halftime'].sum()
    home_xg_against = df_filtered.groupby('Home')['Axg_halftime'].sum()
    away_xg_against = df_filtered.groupby('Opponent')['xg_halftime'].sum()

    total_xg_for = home_xg_for.add(away_xg_for, fill_value=0)
    total_xg_against = home_xg_against.add(away_xg_against, fill_value=0)

    # Zusammenführen der Punkte und Halbzeitenanzahl in einem DataFrame
    total_points['Halbzeitenanzahl'] = total_points['Team'].map(total_counts)
    total_points['xG Für'] = total_points['Team'].map(total_xg_for)
    total_points['xG Gegen'] = total_points['Team'].map(total_xg_against)

    total_points['xG diff'] = total_points['xG Für'] - total_points['xG Gegen'] 

    total_points = total_points.sort_values(by=['Gesamtpunkte'], ascending=False)

    return total_points

def get_table_ballpositionstyle(dfxg_df_merged_cleaned):
    # Schritt 2: Filtern der Spiele mit Ballbesitz > 59%
    df_filtered = dfxg_df_merged_cleaned[(dfxg_df_merged_cleaned['BP-H'] > 56) ]
    # st.write("bp:")
    # st.write(df_filtered)
    # calculate the metrics for this selection of halftimes-df
    table_ballpositionstyle = post_process_table(df_filtered)
    return table_ballpositionstyle

def get_table_counterstyle(dfxg_df_merged_cleaned):
    df_filtered = dfxg_df_merged_cleaned[(dfxg_df_merged_cleaned['BP-H'] < 44) ]
    # st.write("counter:")
    # st.write(df_filtered)
    # calculate the metrics for this selection of halftimes-df
    table_counterstyle = post_process_table(df_filtered)
    return table_counterstyle

def get_table_even(dfxg_df_merged_cleaned):
    df_filtered = dfxg_df_merged_cleaned[(dfxg_df_merged_cleaned['BP-H'] <= 56) & (dfxg_df_merged_cleaned['BP-H'] >= 44)]
    # st.write("counter:")
    # st.write(df_filtered)
    # calculate the metrics for this selection of halftimes-df
    table_evenstyle = post_process_table(df_filtered)
    return table_evenstyle

# Schritt 1: Berechnen des Halbzeit-Ergebnisses für das Heimteam
def calc_halftime_result_h(row):
    goal_diff = row['G-H'] - row['G-A']
    if goal_diff > 0:
        return 'Gewinn'
    elif goal_diff < 0:
        return 'Verlust'
    else:
        return 'Unentschieden'
    
def calc_halftime_result_a(row):
    goal_diff = row['G-A'] - row['G-H']
    if goal_diff > 0:
        return 'Gewinn'
    elif goal_diff < 0:
        return 'Verlust'
    else:
        return 'Unentschieden'
            
def page_league_table():
    
    # TODO: iteriere über alle teams und packe alle ergebnisse in eine seperate liste und die kommt
    # dann in eine neue seperate ergebnis df!
    result_table_ballpositionstyle_list = []
    result_table_counterstyle_list = []
    result_table_evenstyle_list = []
    error_list = []

    for team in df_complete_saison["H_Teamnames"].unique():
        try:
            # teamname corrected that it fits to htdatan teamnames
            # team = df_complete_saison["H_Teamnames"].unique()[0]
            dfxg = load_xg_season_stats_sql(saison)

            # convert xg teamnames to correct ones that are used in htdatan
            dfxg = process_team_names_of_df(dfxg)

            # rename columns for
            dfxg_rename = dfxg.rename(
                columns={'TEAMS': 'H_Teamnames', 'A_TEAMS': 'A_Teamnames'})

            dfxg_df_merged = pd.merge(
                df_complete_saison, dfxg_rename, on=["H_Teamnames", "A_Teamnames"])
            df = dfxg_df_merged.drop_duplicates()

            # the following is only so the functions have the expecting columns
            df["homexg_complete_game"] = ""
            df["awayxg_complete_game"] = ""
            df["last_game_minute"] = -1
            df["start_min_game"] = -1

            dfxg_df_merged_cleaned = df_cleaning_converting(df)

            df4Home, df4OpponentReversed = df_specific_team(
                dfxg_df_merged_cleaned, team)
            dfxg_df_merged_cleaned = create_df4Complete(df4Home, df4OpponentReversed)

            # Sortieren des DataFrames nach Index (falls erforderlich)
            dfxg_df_merged_cleaned = dfxg_df_merged_cleaned.sort_index()

            # Berechnen der xG-Werte für die zweite Halbzeit und Aktualisieren der 'xg_halftime'-Spalte
            for i in range(1, len(dfxg_df_merged_cleaned), 2):  # Start bei 1 und springe jede zweite Zeile
                if dfxg_df_merged_cleaned.iloc[i]['Home'] == dfxg_df_merged_cleaned.iloc[i-1]['Home'] and dfxg_df_merged_cleaned.iloc[i]['Opponent'] == dfxg_df_merged_cleaned.iloc[i-1]['Opponent']:
                    dfxg_df_merged_cleaned.at[i, 'xg_halftime'] = dfxg_df_merged_cleaned.iloc[i]['xG'] - dfxg_df_merged_cleaned.iloc[i-1]['xg_halftime']
                    dfxg_df_merged_cleaned.at[i, 'Axg_halftime'] = dfxg_df_merged_cleaned.iloc[i]['A_xG'] - dfxg_df_merged_cleaned.iloc[i-1]['Axg_halftime']
            
            dfxg_df_merged_cleaned['Halbzeit_Ergebnis_H'] = dfxg_df_merged_cleaned.apply(calc_halftime_result_h, axis=1)
            dfxg_df_merged_cleaned['Halbzeit_Ergebnis_A'] = dfxg_df_merged_cleaned.apply(calc_halftime_result_a, axis=1)

            # now it should not be called Home Away but Team and Opposition!

            # st.write(dfxg_df_merged_cleaned)

            table_ballpositionstyle = get_table_ballpositionstyle(dfxg_df_merged_cleaned)
            table_counterstyle = get_table_counterstyle(dfxg_df_merged_cleaned)
            table_evenstyle = get_table_even(dfxg_df_merged_cleaned)

            # st.dataframe(table_ballpositionstyle[table_ballpositionstyle["Team"]==team])
            # st.dataframe(table_counterstyle[table_counterstyle["Team"]==team])
            # st.dataframe(table_evenstyle[table_evenstyle["Team"]==team])

            result_table_ballpositionstyle_list.append(table_ballpositionstyle[table_ballpositionstyle["Team"]==team])
            result_table_counterstyle_list.append(table_counterstyle[table_counterstyle["Team"]==team])
            result_table_evenstyle_list.append(table_evenstyle[table_evenstyle["Team"]==team])

        except:
            print(f"error for team {team}")
            error_list.append(team)


    # Concatenate all DataFrame rows into a final DataFrame
    result_table_ballpositionstyle = pd.concat(result_table_ballpositionstyle_list, ignore_index=True)
    result_table_counterstyle = pd.concat(result_table_counterstyle_list, ignore_index=True)
    result_table_evenstyle = pd.concat(result_table_evenstyle_list, ignore_index=True)
    
    result_table_ballpositionstyle = result_table_ballpositionstyle.sort_values(by=['Gesamtpunkte'], ascending=False)
    result_table_counterstyle = result_table_counterstyle.sort_values(by=['Gesamtpunkte'], ascending=False)
    result_table_evenstyle = result_table_evenstyle.sort_values(by=['Gesamtpunkte'], ascending=False)

    # Titel der Seite
    st.title("League table per play style")

    # Erstellen von drei Spalten
    col1, col2, col3 = st.columns(3)

    # Anzeigen des zweiten DataFrames in der zweiten Spalte
    with col1:
        st.header("Counter Style")  # Titel für das zweite DataFrame
        st.dataframe(result_table_counterstyle)

    # Anzeigen des dritten DataFrames in der dritten Spalte
    with col2:
        st.header("Even Style")  # Titel für das dritte DataFrame
        st.dataframe(result_table_evenstyle)

    # Anzeigen des ersten DataFrames in der ersten Spalte
    with col3:
        st.header("Ball Position Style")  # Titel für das erste DataFrame
        st.dataframe(result_table_ballpositionstyle)

    st.write(error_list)

# Define Functions:
def process_team_names_of_df(x_df):
    x_df = x_df.replace(teamnamedict)
    return x_df

#######################################################
###  calculate table with two halftimes to one game ###
#######################################################

def df_cleaning_converting(df):
    df = df[['H_Teamnames', 'A_Teamnames', 'H_Goals', 'A_Goals', 'H_Ball Possession', 'A_Ball Possession', 'H_Goal Attempts', 'A_Goal Attempts',
             'H_Shots on Goal', 'A_Shots on Goal', 'H_Shots off Goal', 'A_Shots off Goal', 'H_Free Kicks',"H_Red Cards", "A_Red Cards",
             'A_Free Kicks', 'H_Corner Kicks', 'A_Corner Kicks', 'H_Offsides', 'A_Offsides', 'H_Goalkeeper Saves', 'A_Goalkeeper Saves',
             'H_Fouls', 'A_Fouls', 'A_gameinfo', 'A_datetime', 'xG', 'GOALS', 'xPTS', 'A_xG', 'A_GOALS', 'A_xPTS', 'timing_chart_xg', "homexg_complete_game", "awayxg_complete_game"]]
    df = df.drop_duplicates(subset=['H_Teamnames', 'A_Teamnames', 'H_Goals', 'A_Goals', 'H_Ball Possession', 'A_Ball Possession', 'H_Goal Attempts', 'A_Goal Attempts',
             'H_Shots on Goal', 'A_Shots on Goal', 'H_Shots off Goal', 'A_Shots off Goal', 'H_Free Kicks',"H_Red Cards", "A_Red Cards",
             'A_Free Kicks', 'H_Corner Kicks', 'A_Corner Kicks', 'H_Offsides', 'A_Offsides', 'H_Goalkeeper Saves', 'A_Goalkeeper Saves',
             'H_Fouls', 'A_Fouls', 'A_gameinfo', 'A_datetime', 'xG', 'GOALS', 'xPTS', 'A_xG', 'A_GOALS', 'A_xPTS'], keep='first')
    df = df.reset_index(drop=True)
    df["R"] = 'X'

    df['timing_chart_xg'] = df['timing_chart_xg'].astype('str') 

    def extract_xg_values(df):
        """ Extracts the xG values from the timing_chart_xg columnn.
            This contains a string formatted as a list of xG values for each minute.
        """
        for index, _ in df.iterrows():
            try:
                timing_chart = df["timing_chart_xg"].loc[index]
                
                # Away team XG at halftime
                away_xg = None
                for minute in ['46', '45', '47']:
                    try:
                        if minute == '46':
                            away_xg = timing_chart.split("46' Total xG:")[0].split("Total xG: ")[-1].split("\n")[0].replace(";", "")
                        else:
                            away_xg = timing_chart.split(f"{int(minute)}' Total xG:")[0].split(f"{minute}' Total xG: ")[-1].split("\n")[0].replace(";", "")
                        break
                    except Exception:
                        if minute == '46':
                            print(f"Could not find 46' Total xG for away team at index {index}, trying 45' or 47'")
                        continue
                
                if away_xg is None:
                    print(f"Failed to extract away XG for all minute variations at index {index}")
                    continue
                
                # Home team XG at halftime
                home_xg = None
                for minute in ['45', '44', '46']:
                    try:
                        if minute == '45':
                            home_xg = timing_chart.split("45' Total xG: ")[1].split(";45' Total xG:")[0].split("\n")[0].replace(";", "")
                        else:
                            home_xg = timing_chart.split(f"{minute}' Total xG: ")[1].split(f";{minute}' Total xG:")[0].split("\n")[0].replace(";", "")
                        break
                    except Exception:
                        if minute == '45':
                            print(f"Could not find 45' Total xG for home team at index {index}, trying 44' or 46'")
                        continue
                
                if home_xg is None:
                    print(f"Failed to extract home XG for all minute variations at index {index}")
                    continue
                    
                # Assign values to dataframe
                df.loc[index, 'xg_halftime'] = home_xg
                df.loc[index, 'Axg_halftime'] = away_xg
                
            except Exception as e:
                print(f"Error processing index {index}: {str(e)}")
                continue
        
        return df

    df = extract_xg_values(df)


    df.xg_halftime = df.xg_halftime.astype(float).fillna(0.0)
    df.Axg_halftime = df.Axg_halftime.astype(float).fillna(0.0) 

    for i in range(0, len(df)):
        try:

            if df["H_Goals"][i] > df["A_Goals"][i]:
                df["R"][i] = 'H'
            if df["H_Goals"][i] < df["A_Goals"][i]:
                df["R"][i] = 'A'
            else:
                df["R"][i] = 'D'
        except:
            print("error?")
    print("!!!", df.columns)
    df.columns = ['Home', 'Opponent', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A',
                'SoG-H', 'SoG-A', 'SoffG-H', 'SoffG-A', 'FK-H',"H_Red Cards", "A_Red Cards",
                'FK-A', 'C-H', 'C-A', 'Off-H', 'Off-A', 'GoKeSa-H', 'GoKeSa-A',
                'F-H', 'F-A', 'Round', 'Date', 'xG', 'GOALS', 'xPTS', 'A_xG', 'A_GOALS', 'A_xPTS', "timing_chart_xg", "homexg_complete_game", "awayxg_complete_game", 'R',  'xg_halftime', 'Axg_halftime']

    df = df[['Home', 'Opponent', 'R', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A',
             'SoG-H', 'SoG-A', 'SoffG-H', 'SoffG-A', 'FK-H',"H_Red Cards", "A_Red Cards",
             'FK-A', 'C-H', 'C-A', 'Off-H', 'Off-A', 'GoKeSa-H', 'GoKeSa-A', 'F-H',
             'F-A', 'Round', 'Date', 'xG', 'GOALS', 'xPTS', 'A_xG', 'A_GOALS', 'A_xPTS', 'xg_halftime', 'Axg_halftime', "homexg_complete_game", "awayxg_complete_game"]]

    df["IsHome"] = 0

    df = df[['Home', 'Opponent', 'R', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A',
             'SoG-H', 'SoG-A', 'SoffG-H', 'SoffG-A', 'FK-H',"H_Red Cards", "A_Red Cards",
             'FK-A', 'C-H', 'C-A', 'Off-H', 'Off-A', 'GoKeSa-H', 'GoKeSa-A', 'F-H',
             'F-A', 'Round', 'Date', 'IsHome', 'xG', 'GOALS', 'xPTS', 'A_xG', 'A_GOALS', 'A_xPTS','xg_halftime', 'Axg_halftime', "homexg_complete_game", "awayxg_complete_game", ]]
    return df


@st.cache_data
def df_specific_team(df, team):
    df4Home = df.loc[((df['Home'] == team))]

    df4Home["IsHome"] = 1

    # recalcualte the winner because of the columns switching to bring the selected team in the first column
    df4Home["1x2"] = 0

    df4Home["1x2"] = df4Home.apply(
        lambda row: calculate_1x2_home(row), axis=1, result_type='reduce')

    # Berechnung Opponentgames
    df4Opponent = df.loc[((df['Opponent'] == team))]

    # recalcualte the winner because of the columns switching to bring the selected team in the first column
    df4Opponent["1x2"] = 0

    df4Opponent["1x2"] = df4Opponent.apply(
        lambda row: calculate_1x2_Opponent(row), axis=1, result_type='reduce')

    # change the halftime-xg with halftime-Axg:
    # switched the two columns
    # Change the columns for the Opponentmatches of the specific team
    print("df4Opponent.columns before reassignment oppo", df4Opponent.columns)

    OpponentTeamReversedColumns = ['Opponent', 'Home',  '1x2', 'R',  'G-A', 'G-H', 'BP-A', 'BP-H', 'GA-A', 'GA-H',  
                                    'SoG-A', 'SoG-H', 'SoffG-A', 'SoffG-H',  'FK-A',  "A_Red Cards", "H_Red Cards",
                                    'FK-H','C-A', 'C-H',  'Off-A', 'Off-H', 'GoKeSa-A', 'GoKeSa-H', 'F-A', 
                                    'F-H', 'Round', 'Date', 'IsHome','A_xG', "A_xPTS", "A_GOALS", 'xG', "xPTS", "GOALS", 'Axg_halftime', 'xg_halftime',"awayxg_complete_game", "homexg_complete_game",   ]  # , 'IsHome'
    # Change the columns for the Opponentmatches of the specific team

    df4OpponentReversed = df4Opponent.reindex(
        columns=OpponentTeamReversedColumns)

    df4OpponentReversed.columns = ['Home', 'Opponent', '1x2', 'R', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A',
                                   'SoG-H', 'SoG-A', 'SoffG-H', 'SoffG-A', 'FK-H', "H_Red Cards", "A_Red Cards",
                                   'FK-A', 'C-H', 'C-A', 'Off-H', 'Off-A', 'GoKeSa-H', 'GoKeSa-A', 'F-H',
                                   'F-A', 'Round', 'Date', 'IsHome', 'xG', "xPTS", "GOALS", 'A_xG', "A_xPTS", "A_GOALS", 'xg_halftime', 'Axg_halftime', "homexg_complete_game", "awayxg_complete_game", ]

    print("df4OpponentReversed.columns after reassignment oppo", df4OpponentReversed.columns)

    return df4Home, df4OpponentReversed

@st.cache_data
def create_df4Complete(df4Home, df4OpponentReversed):
    # Alle Spiele werden als Heimspiel angezeigt, sind aber auch Auswärtsspiele dabei!
    df4Complete = pd.concat([df4Home, df4OpponentReversed], sort=False)

    df4Complete["G-H"] = df4Complete["G-H"].astype('float64')
    df4Complete["G-A"] = df4Complete["G-A"].astype('float64')
    df4Complete["BP-H"] = df4Complete["BP-H"].astype('float64')
    df4Complete["BP-A"] = df4Complete["BP-A"].astype('float64')

    # GoalDifference
    df4Complete["GoalDiff"] = df4Complete["G-H"] - df4Complete["G-A"]
    df4Complete = df4Complete.sort_values("Date",  ascending=False)

    # calculate column with 3 Ballposition types
    df4Complete["BPTypes"] = '0'

    df4Complete["BPTypes"] = df4Complete.apply(
        lambda row: calculate_1x2_BPTypes(row), axis=1, result_type='reduce')

    df4Complete['Date'] = pd.to_datetime(
        df4Complete['Date'], format="%d.%m.%Y %H:%M")

    # convert datetime to timestamp for scatter visualization
    df4Complete['timestamp'] = df4Complete.Date.astype('int64')//10**9

    df4Complete = df4Complete.sort_values("Date", ascending=False)

    # Create data for scatter graph
    df4Complete["SoG-H"] = df4Complete["SoG-H"].astype(int)
    df4Complete["SoG-A"] = df4Complete["SoG-A"].astype(int)

    return df4Complete


@st.cache_data
def calculate_1x2_home(row):
    if row['G-H'] > row['G-A']:
        return 'W'
    elif row['G-A'] > row['G-H']:
        return 'L'
    else:
        return 'D'

@st.cache_data
def calculate_1x2_Opponent(row):
    if row['G-A'] > row['G-H']:
        return 'W'
    elif row['G-H'] > row['G-A']:
        return 'L'
    else:
        return 'D'

@st.cache_data
def calculate_1x2_BPTypes(row):
    if row['BP-H'] > 55:
        return '>55'
    elif row['BP-H'] < 45:
        return '<45'
    else:
        return '45-55'

@st.cache_data
def calc_stats(df4Complete):
    BP_WPerc = df4Complete[['BPTypes', '1x2']
                           ].loc[df4Complete['BPTypes'] == '>55']
    BP_WAbs = BP_WPerc['1x2'].loc[df4Complete['1x2'] == 'W']
    BP_NWAbs = BP_WPerc['1x2'].loc[df4Complete['1x2'] != 'W']
    if len(BP_WAbs) + len(BP_NWAbs) > 0:
        BP_WPercText = len(BP_WAbs) / (len(BP_WAbs) + len(BP_NWAbs)) * 100
        BP_WPercText = round(BP_WPercText)
    else:
        BP_WPercText = 0

    # calculate winning % for 0.45 - 0.55
    N_WPerc = df4Complete[['BPTypes', '1x2']
                          ].loc[df4Complete['BPTypes'] == '45-55']
    N_WAbs = N_WPerc['1x2'].loc[df4Complete['1x2'] == 'W']
    N_NWAbs = N_WPerc['1x2'].loc[df4Complete['1x2'] != 'W']
    if len(N_WAbs) + len(N_NWAbs) > 0:
        N_WPercText = len(N_WAbs) / (len(N_WAbs) + len(N_NWAbs)) * 100
        N_WPercText = round(N_WPercText)
    else:
        N_WPercText = 5

    # calculate winning % for < 0.45
    C_WPerc = df4Complete[['BPTypes', '1x2']
                          ].loc[df4Complete['BPTypes'] == '<45']
    C_WAbs = C_WPerc['1x2'].loc[df4Complete['1x2'] == 'W']
    C_NWAbs = C_WPerc['1x2'].loc[df4Complete['1x2'] != 'W']
    if len(C_WAbs) + len(C_NWAbs) > 0:
        C_WPercText = len(C_WAbs) / (len(C_WAbs) + len(C_NWAbs)) * 100
        C_WPercText = round(C_WPercText)
    else:
        C_WPercText = 0

    return C_WPercText, N_WPercText, BP_WPercText


# get name of the selected team in dropdown
def load_xg_gamestats_sql(saison, team):

    if saison.split("_")[0] == 'b':
        xgprefix = 'bundesliga'
    elif saison.split("_")[0] == 'l1':
        xgprefix = 'ligue_1'
    elif saison.split("_")[0] == 'll':
        xgprefix = 'la_liga'
    elif saison.split("_")[0] == 'pl':
        xgprefix = 'epl'
    elif saison.split("_")[0] == 'sa':
        xgprefix = 'serie_a'

    xgtablename = "{}20{}".format(xgprefix, saison.split("_")[1][:2])

    df_complete_saison = pd.read_csv(
        "data/xg/"+xgtablename+".csv", index_col=0, encoding='utf-8')

    df_complete_saison = process_team_names_of_df(df_complete_saison)

    # execute the query and assign it to a pandas dataframe
    dfxg = df_complete_saison[(df_complete_saison.TEAMS == team) | (
        df_complete_saison.A_TEAMS == team)]

    return dfxg

# get all teams for the selected season in dropdown
def load_xg_season_stats_sql(saison):

    if saison.split("_")[0] == 'b':
        xgprefix = 'bundesliga'
    elif saison.split("_")[0] == 'l1':
        xgprefix = 'ligue_1'
    elif saison.split("_")[0] == 'll':
        xgprefix = 'la_liga'
    elif saison.split("_")[0] == 'pl':
        xgprefix = 'epl'
    elif saison.split("_")[0] == 'sa':
        xgprefix = 'serie_a'

    xgtablename = "{}20{}".format(xgprefix, saison.split("_")[1][:2])

    df_complete_saison = pd.read_csv(
        "data/xg/"+xgtablename+".csv", index_col=0, encoding='utf-8')

    df_complete_saison = process_team_names_of_df(df_complete_saison)

    return df_complete_saison


# Definiere die Funktionen für jede Seite
def page_teamx():
    # take only the 0 part of the every list entry
    teamsoptions = []
    for x in range(0, len(dfallteamnamesl)):
        teamsoptions.append(dfallteamnamesl[x])

    global xg_team
    xg_team = st.sidebar.selectbox("Team", list(np.sort(teamsoptions)), 0)

    # convert string to df to use process_team_names_of_df function
    df_teamname = pd.DataFrame([xg_team])

    # convert xg teamnames to correct ones that are used in htdatan
    team_df = df_teamname.replace(teamnamedict)
    # teamname corrected that it fits to htdatan teamnames
    team = team_df.iloc[0][0]

    df = df_complete_saison[(df_complete_saison.H_Teamnames == team) | (
        df_complete_saison.A_Teamnames == team)]

    
    df = process_team_names_of_df(df)
    dfxg = load_xg_gamestats_sql(saison, team)

    # rename columns for
    dfxg_rename = dfxg.rename(
        columns={'TEAMS': 'H_Teamnames', 'A_TEAMS': 'A_Teamnames'})
    # del dfxg

    dfxg_df_merged = pd.merge(
        df, dfxg_rename, on=["H_Teamnames", "A_Teamnames"])
    dfxg_df_merged = dfxg_df_merged.drop_duplicates()

    df = dfxg_df_merged

    df["homexg_complete_game"] = ""
    df["awayxg_complete_game"] = ""
    df["last_game_minute"] = -1
    df["start_min_game"] = -1

            
    for game_loc in df.index:

        homexg_complete_game = []
        awayxg_complete_game = []

        last_game_minute = df["timing_chart_xg"].loc[game_loc].rsplit("'")[-2].rsplit(";")[1]
        start_min_game = int( re.sub("[^0-9]", "", df["timing_chart_xg"].loc[game_loc][:2]) )

        # nehmen hier minute für minute und schauen nach dem xg wert für diese minute
        for x in range(start_min_game,int(last_game_minute)+1):
            try:
                homexgperminute = df["timing_chart_xg"].loc[game_loc].split("{}' Total xG: ".format(x))[1].split(";")[0][:4]  # [:4] - only last 4 digits so no goalscorer infos
                awayxgperminute = df["timing_chart_xg"].loc[game_loc].split("{}' Total xG: ".format(x), 2)[2].split(";")[0][:4]
            except:
                # falls die minute fehlt nehmen wir einfach den xg wert von der vorherigen minute!
                print("min {} is missing in xg".format(x))

            homexg_complete_game.append(homexgperminute)
            awayxg_complete_game.append(awayxgperminute)
            
        df["homexg_complete_game"].loc[game_loc] = homexg_complete_game
        df["awayxg_complete_game"].loc[game_loc] = awayxg_complete_game
        df["last_game_minute"].loc[game_loc] = last_game_minute
        df["start_min_game"].loc[game_loc] = start_min_game

    for x in range(len(df)):
        df.homexg_complete_game.iloc[x][0:0] = [None] * df.start_min_game.iloc[x]
        df.awayxg_complete_game.iloc[x][0:0] = [None] * df.start_min_game.iloc[x]



    df_homexg_complete_game = pd.DataFrame(df.homexg_complete_game.tolist(), index= df.index)
    df_awayxg_complete_game = pd.DataFrame(df.awayxg_complete_game.tolist(), index= df.index)

    df_homexg_complete_game = df_homexg_complete_game.apply(pd.to_numeric)
    df_awayxg_complete_game = df_awayxg_complete_game.apply(pd.to_numeric)

    dfxg_df_merged_cleaned = df_cleaning_converting(df)

    df4Home, df4OpponentReversed = df_specific_team(
        dfxg_df_merged_cleaned, team)

    df4Complete = create_df4Complete(df4Home, df4OpponentReversed)

    slidertext = 'Show last x halftimes'
    nrGames = st.sidebar.slider(slidertext, max_value=len(
        df4Complete), value=len(df4Complete), step=2)

    # change rows of df depending on userinput
    df4Complete = df4Complete[:nrGames]
    df4Complete = df4Complete.sort_values("Date", ascending=False)
    df4Complete = df4Complete.round(1)

    df4Complete[['xG', 'A_xG', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A', 'SoG-H', 'SoG-A',  'xPTS', 'A_xPTS',  "A_Red Cards", "H_Red Cards"]] = df4Complete[[
                 'xG', 'A_xG', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A', 'SoG-H', 'SoG-A',  'xPTS', 'A_xPTS',  "A_Red Cards", "H_Red Cards"]].apply(pd.to_numeric, errors='coerce', axis=1)

    df4Complete['A_Red Cards'] = df4Complete['A_Red Cards'].fillna(0)
    df4Complete['H_Red Cards'] = df4Complete['H_Red Cards'].fillna(0)

    df4Complete_show = df4Complete[['Home', 'Opponent', 'IsHome', 'R', 'xG', 'A_xG', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A',
                                    'SoG-H', 'SoG-A',  'xPTS', 'A_xPTS', 'Date', 'xg_halftime', 'Axg_halftime', "A_Red Cards", "H_Red Cards"]]

    # calc the xg per minute over all games to get the mean over all minutes from all games!

    values = st.sidebar.slider(
        'BP-Range for xG per minute',
        0.0, 100.0, (0.0, 100.0))

    smaller_bp = values[1]
    bigger_bp = values[0]
    dfxg_homexg_complete_game = pd.DataFrame(df4Complete[(df4Complete["BP-H"]>bigger_bp) & (df4Complete["BP-H"]<smaller_bp)].homexg_complete_game.tolist(), index= df4Complete[(df4Complete["BP-H"]>bigger_bp) & (df4Complete["BP-H"]<smaller_bp)].index)
    dfxg_awayxg_complete_game = pd.DataFrame(df4Complete[(df4Complete["BP-H"]>bigger_bp) & (df4Complete["BP-H"]<smaller_bp)].awayxg_complete_game.tolist(), index= df4Complete[(df4Complete["BP-H"]>bigger_bp) & (df4Complete["BP-H"]<smaller_bp)].index)
    dfxg_homexg_complete_game = dfxg_homexg_complete_game.apply(pd.to_numeric)
    dfxg_awayxg_complete_game = dfxg_awayxg_complete_game.apply(pd.to_numeric)
    dfxg_homexg_complete_game = dfxg_homexg_complete_game.fillna(0)
    dfxg_awayxg_complete_game = dfxg_awayxg_complete_game.fillna(0)
    if len(dfxg_homexg_complete_game) > 0 and len(dfxg_awayxg_complete_game) > 0:
        dfxg_homexg_complete_game = dfxg_homexg_complete_game.diff(axis=1)
        dfxg_awayxg_complete_game = dfxg_awayxg_complete_game.diff(axis=1)
        dfxg_homexg_complete_game[dfxg_homexg_complete_game < 0] = 0
        dfxg_awayxg_complete_game[dfxg_awayxg_complete_game < 0] = 0

    dfxg_homexg_complete_game_all_bps = pd.DataFrame(df4Complete.homexg_complete_game.tolist(), index= df4Complete.index)
    dfxg_awayxg_complete_game_all_bps = pd.DataFrame(df4Complete.awayxg_complete_game.tolist(), index= df4Complete.index)
    dfxg_homexg_complete_game_all_bps = dfxg_homexg_complete_game_all_bps.apply(pd.to_numeric)
    dfxg_awayxg_complete_game_all_bps = dfxg_awayxg_complete_game_all_bps.apply(pd.to_numeric)
    dfxg_homexg_complete_game_all_bps = dfxg_homexg_complete_game_all_bps.fillna(0)
    dfxg_awayxg_complete_game_all_bps = dfxg_awayxg_complete_game_all_bps.fillna(0)
    if len(dfxg_homexg_complete_game_all_bps) > 0 and len(dfxg_awayxg_complete_game_all_bps) > 0:
        dfxg_homexg_complete_game_all_bps = dfxg_homexg_complete_game_all_bps.diff(axis=1)
        dfxg_awayxg_complete_game_all_bps = dfxg_awayxg_complete_game_all_bps.diff(axis=1)
        dfxg_homexg_complete_game_all_bps[dfxg_homexg_complete_game_all_bps < 0] = 0
        dfxg_awayxg_complete_game_all_bps[dfxg_awayxg_complete_game_all_bps < 0] = 0

    dfxg_homexg_complete_game_bigger_55 = pd.DataFrame(df4Complete[(df4Complete["BP-H"]>55)].homexg_complete_game.tolist(), index= df4Complete[(df4Complete["BP-H"]>55)].index)
    dfxg_awayxg_complete_game_bigger_55 = pd.DataFrame(df4Complete[(df4Complete["BP-H"]>55)].awayxg_complete_game.tolist(), index= df4Complete[(df4Complete["BP-H"]>55)].index)
    dfxg_homexg_complete_game_bigger_55 = dfxg_homexg_complete_game_bigger_55.apply(pd.to_numeric)
    dfxg_awayxg_complete_game_bigger_55 = dfxg_awayxg_complete_game_bigger_55.apply(pd.to_numeric)
    dfxg_homexg_complete_game_bigger_55 = dfxg_homexg_complete_game_bigger_55.fillna(0)
    dfxg_awayxg_complete_game_bigger_55 = dfxg_awayxg_complete_game_bigger_55.fillna(0)
    if len(dfxg_homexg_complete_game_bigger_55) > 0 and len(dfxg_awayxg_complete_game_bigger_55) > 0:
        dfxg_homexg_complete_game_bigger_55 = dfxg_homexg_complete_game_bigger_55.diff(axis=1)
        dfxg_awayxg_complete_game_bigger_55 = dfxg_awayxg_complete_game_bigger_55.diff(axis=1)
        dfxg_homexg_complete_game_bigger_55[dfxg_homexg_complete_game_bigger_55 < 0] = 0
        dfxg_awayxg_complete_game_bigger_55[dfxg_awayxg_complete_game_bigger_55 < 0] = 0

    dfxg_homexg_complete_game_smaller_45 = pd.DataFrame(df4Complete[(df4Complete["BP-H"]<45)].homexg_complete_game.tolist(), index= df4Complete[(df4Complete["BP-H"]<45)].index)
    dfxg_awayxg_complete_game_smaller_45 = pd.DataFrame(df4Complete[(df4Complete["BP-H"]<45)].awayxg_complete_game.tolist(), index= df4Complete[(df4Complete["BP-H"]<45)].index)
    dfxg_homexg_complete_game_smaller_45 = dfxg_homexg_complete_game_smaller_45.apply(pd.to_numeric)
    dfxg_awayxg_complete_game_smaller_45 = dfxg_awayxg_complete_game_smaller_45.apply(pd.to_numeric)
    dfxg_homexg_complete_game_smaller_45 = dfxg_homexg_complete_game_smaller_45.fillna(0)
    dfxg_awayxg_complete_game_smaller_45 = dfxg_awayxg_complete_game_smaller_45.fillna(0)
    if len(dfxg_homexg_complete_game_smaller_45) > 0 and len(dfxg_awayxg_complete_game_smaller_45) > 0:
        dfxg_homexg_complete_game_smaller_45 = dfxg_homexg_complete_game_smaller_45.diff(axis=1)
        dfxg_awayxg_complete_game_smaller_45 = dfxg_awayxg_complete_game_smaller_45.diff(axis=1)
        dfxg_homexg_complete_game_smaller_45[dfxg_homexg_complete_game_smaller_45 < 0] = 0
        dfxg_awayxg_complete_game_smaller_45[dfxg_awayxg_complete_game_smaller_45 < 0] = 0


    print("df4Complete_show")
    print(df4Complete_show)
    # create df for visualizing
    df4CompleteGraph = df4Complete.copy()


    teamname_to_search = st.sidebar.text_input("Search for Opponent", )
    df4CompleteGraph = df4CompleteGraph[df4CompleteGraph["Opponent"].str.contains("{}".format(teamname_to_search), na=False, case=False)]


    df4CompleteGraph["SoG-H-SoG-A"] = df4CompleteGraph["SoG-H"] - \
        df4CompleteGraph["SoG-A"]
    df4CompleteGraph["SoG-H-SoG-A"] = df4CompleteGraph["SoG-H-SoG-A"].clip(
        lower=0)

    df4CompleteGraph["SoG-A-SoG-H"] = df4CompleteGraph["SoG-A"] - \
        df4CompleteGraph["SoG-H"]
    df4CompleteGraph["SoG-A-SoG-H"] = df4CompleteGraph["SoG-A-SoG-H"].clip(
        lower=0)

    df4CompleteGraph.sort_values("IsHome", ascending=False)


    # calculate the difference of team xg vs oppo xg
    dfxg_complete_game = dfxg_homexg_complete_game.clip(lower=0).mean() - dfxg_awayxg_complete_game.clip(lower=0).mean()
    fig_xg_perminute_home = go.Figure(data=[
        go.Bar(
            name='xG', y=dfxg_homexg_complete_game.clip(lower=0).mean(), marker_color='green'
            ),
        # -1 to show the bars to the below instead of above!
        go.Bar(
            name='Opponent xG', y=dfxg_awayxg_complete_game.clip(lower=0).mean()*-1, marker_color='red' 
            ),
    ])
    fig_xg_perminute_home.add_trace(
        go.Scatter(
            name='Diff xG',
            y=dfxg_complete_game,
            line=dict(color='gold', width=3)
            )
        )
    fig_xg_perminute_home.update_layout(
        title_text='Expectedgoals per minute: {} < bp < {}'.format(int(bigger_bp), int(smaller_bp)), title_x=title_x,
        yaxis=dict(
            title="xG"
        ),
        autosize=False,
        width=1400,
        height=400,
        )
    fig_xg_perminute_home.update_yaxes(range=[(dfxg_awayxg_complete_game.mean().max()+0.02)*-1, dfxg_homexg_complete_game.mean().max()+0.02])

    # calculate the difference of team xg vs oppo xg
    dfxg_complete_game_all_bps = dfxg_homexg_complete_game_all_bps.clip(lower=0).mean() - dfxg_awayxg_complete_game_all_bps.clip(lower=0).mean()
    fig_xg_homexg_complete_game_all_bpse = go.Figure(data=[
        go.Bar(
            name='xG', y=dfxg_homexg_complete_game_all_bps.clip(lower=0).mean(), marker_color='green'
            ),
        # -1 to show the bars to the below instead of above!
        go.Bar(
            name='Opponent xG', y=dfxg_awayxg_complete_game_all_bps.clip(lower=0).mean()*-1, marker_color='red' 
            ),
    ])
    fig_xg_homexg_complete_game_all_bpse.add_trace(
        go.Scatter(
            name='Diff xG',
            y=dfxg_complete_game_all_bps,
            line=dict(color='gold', width=3)
            )
        )
    fig_xg_homexg_complete_game_all_bpse.update_layout(
        title_text='Expectedgoals per minute', title_x=title_x,
        yaxis=dict(
            title="xG"
        ),
        autosize=False,
        width=1400,
        height=400,
        )
    fig_xg_homexg_complete_game_all_bpse.update_yaxes(range=[(dfxg_awayxg_complete_game.mean().max()+0.02)*-1, dfxg_homexg_complete_game.mean().max()+0.02])



    # calculate the difference of team xg vs oppo xg
    dfxg_complete_game_bigger_55 = dfxg_homexg_complete_game_bigger_55.clip(lower=0).mean() - dfxg_awayxg_complete_game_bigger_55.clip(lower=0).mean()
    fig_xg_perminute_home_bigger_55 = go.Figure(data=[
        go.Bar(
            name='xG', y=dfxg_homexg_complete_game_bigger_55.clip(lower=0).mean(), marker_color='green'
            ),
        # -1 to show the bars to the below instead of above!
        go.Bar(
            name='Opponent xG', y=dfxg_awayxg_complete_game_bigger_55.clip(lower=0).mean()*-1, marker_color='red' 
            ),
    ])
    fig_xg_perminute_home_bigger_55.add_trace(
        go.Scatter(
            name='Diff xG',
            y=dfxg_complete_game_bigger_55,
            line=dict(color='gold', width=3)
            )
        )
    fig_xg_perminute_home_bigger_55.update_layout(
        title_text='Expectedgoals per minute: bp > 55', title_x=title_x,
        yaxis=dict(
            title="xG"
        ),
        autosize=False,
        width=1400,
        height=400,
        )
    fig_xg_perminute_home_bigger_55.update_yaxes(range=[(dfxg_awayxg_complete_game.mean().max()+0.02)*-1, dfxg_homexg_complete_game.mean().max()+0.02])



    # calculate the difference of team xg vs oppo xg
    dfxg_complete_game_smaller_45 = dfxg_homexg_complete_game_smaller_45.clip(lower=0).mean() - dfxg_awayxg_complete_game_smaller_45.clip(lower=0).mean()
    fig_xg_perminute_home_smaller_45 = go.Figure(data=[
        go.Bar(
            name='xG', y=dfxg_homexg_complete_game_smaller_45.clip(lower=0).mean(), marker_color='green'
            ),
        # -1 to show the bars to the below instead of above!
        go.Bar(
            name='Opponent xG', y=dfxg_awayxg_complete_game_smaller_45.clip(lower=0).mean()*-1, marker_color='red' 
            ),
    ])
    fig_xg_perminute_home_smaller_45.add_trace(
        go.Scatter(
            name='Diff xG',
            y=dfxg_complete_game_smaller_45,
            line=dict(color='gold', width=3)
            )
        )
    fig_xg_perminute_home_smaller_45.update_layout(
        title_text='Expectedgoals per minute: bp < 45', title_x=title_x,
        yaxis=dict(
            title="xG"
        ),
        autosize=False,
        width=1400,
        height=400,
        )
    fig_xg_perminute_home_smaller_45.update_yaxes(range=[(dfxg_awayxg_complete_game.mean().max()+0.02)*-1, dfxg_homexg_complete_game.mean().max()+0.02])


    figScatter_SoG_SoGA = px.scatter(
        df4CompleteGraph.sort_values("IsHome", ascending=False),  # .query(f'Date.between{end_date}'),
        x='BP-H',
        y='GoalDiff',
        marginal_x="histogram",
        color="timestamp",
        hover_data=['H_Red Cards', 'A_Red Cards', 'Date'],
        symbol = 'IsHome',
        symbol_sequence= ['diamond-cross', 'diamond'],
        size="SoG-H-SoG-A",
        text="Opponent",
        width=widthfig,
    ).update_traces(textposition='top center', selector={'type': 'scatter'}, textfont_size=9, textfont_color="gray"
    ).update_traces(
        marker=dict(color='green'), selector={'type': 'histogram'}
    )
    figScatter_SoG_SoGA.update_xaxes(range=[5, 95])
    figScatter_SoG_SoGA.update_layout(
        title_text='All halftimes: Shots on Goal - Shots on Goal Opponent', title_x=title_x,
        yaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            title="Goal difference"
        ))
    figScatter_SoG_SoGA.update_layout(legend=dict(
        yanchor="top",
        y=1.2,
        xanchor="right",
        x=1.12
    ))

    figScatter_SoGA_soG = px.scatter(
        df4CompleteGraph.sort_values("IsHome", ascending=False),  # .query(f'Date.between{end_date}'),
        x='BP-H',
        y='GoalDiff',
        marginal_x="histogram",
        color="timestamp",
        hover_data=['H_Red Cards', 'A_Red Cards', 'Date'],
        size="SoG-A-SoG-H",
        symbol = 'IsHome',
        symbol_sequence= ['circle-x', 'circle'],
        text="Opponent",
        width=widthfig,
        # height=heightfig,
        title="SoGA-SoGH - Halftimes",
        # color_discrete_map={"W": "green", "D": "gray", "L": "red"}
        # facet_row="time", # makes seperate plot for value
        # marginal_x="histogram",
    ).update_traces(
        textposition='top center', selector={'type': 'scatter'}, textfont_size=9, textfont_color="gray").update_traces(
            marker=dict(color='red'), selector={'type': 'histogram'}
    )
    figScatter_SoGA_soG.update_xaxes(range=[5, 95])
    figScatter_SoGA_soG.update_layout(
        title_text='All halftimes: Shots on Goal Opponent - Shots on Goal', title_x=title_x,
        yaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            title="Goal difference"
        )
    )
    figScatter_SoGA_soG.update_layout(legend=dict(
        yanchor="top",
        y=1.2,
        xanchor="right",
        x=1.12
    ))


    # delete games where there is no two halftimes!
    df4CompleteGraph = df4CompleteGraph[df4CompleteGraph.groupby(
        'Opponent')['Opponent'].transform('size') >= 2]
    df4CompleteGraph = df4CompleteGraph.sort_index()
    # second half is second entry always!
    df4CompleteGraph["halftime"] = "0"
    df4CompleteGraph['halftime'] = np.where(df4CompleteGraph.index % 2, '1', '2')

    figScatter_h1_soG_SoGA = px.scatter(
        # .query(f'Date.between{end_date}'),
        df4CompleteGraph[df4CompleteGraph["halftime"] == "1"].sort_values("IsHome", ascending=False),
        x='BP-H',
        y='GoalDiff',
        size="SoG-H-SoG-A",
        text="Opponent",
        hover_data=['H_Red Cards', 'A_Red Cards', 'Date'],
        symbol = 'IsHome',
        symbol_sequence= ['diamond-cross', 'diamond'],
        width=widthfig,
        marginal_x="histogram",
    ).update_traces(textposition='top center',  marker=dict(
        color='green'), selector={'type': 'scatter'}, textfont_size=9, textfont_color="gray"
    ).update_traces(marker=dict(
        color='green'), selector={'type': 'histogram'}
    )
    figScatter_h1_soG_SoGA.update_xaxes(range=[5, 95])
    figScatter_h1_soG_SoGA.update_layout(
        title_text='Halftime 1: Shots on Goal - Shots on Goal Opponent', title_x=title_x,
        yaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            title="Goal difference"
        ))
    figScatter_h1_soG_SoGA.update_layout(legend=dict(
        yanchor="top",
        y=1.2,
        xanchor="right",
        x=1.12
    ))

    figScatter_h1_SoGA_soG = px.scatter(
        # .query(f'Date.between{end_date}'),
        df4CompleteGraph[df4CompleteGraph["halftime"] == "1"].sort_values("IsHome", ascending=False),
        x='BP-H',
        y='GoalDiff',
        size="SoG-A-SoG-H",
        text="Opponent",
        hover_data=['H_Red Cards', 'A_Red Cards', 'Date'],
        symbol = 'IsHome',
        symbol_sequence= ['circle-x', 'circle'],
        width=widthfig,
        marginal_x="histogram",
    ).update_traces(textposition='top center', marker=dict(
        color='red'), selector={'type': 'scatter'}, textfont_size=9, textfont_color="gray"
    ).update_traces(marker=dict(
        color='red'), selector={'type': 'histogram'}
    )
    figScatter_h1_SoGA_soG.update_xaxes(range=[5, 95])
    figScatter_h1_SoGA_soG.update_layout(
        title_text='Halftime 1: Shots on Goal Opponent - Shots on Goal', title_x=title_x,
        yaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            title="Goal difference"
        ))
    figScatter_h1_SoGA_soG.update_layout(legend=dict(
        yanchor="top",
        y=1.2,
        xanchor="right",
        x=1.12
    ))


    figScatter_h2_SoG_SoGA = px.scatter(
        # .query(f'Date.between{end_date}'),
        df4CompleteGraph[df4CompleteGraph["halftime"] == "2"].sort_values("IsHome", ascending=False),
        x='BP-H',
        y='GoalDiff',
        marginal_x="histogram",
        hover_data=['H_Red Cards', 'A_Red Cards', 'Date'],
        size="SoG-H-SoG-A",
        symbol = 'IsHome',
        symbol_sequence= ['diamond-cross', 'diamond'],
        text="Opponent",
        width=widthfig,
    ).update_traces(textposition='top center',  marker=dict(
        color='green'), selector={'type': 'scatter'}, textfont_size=9, textfont_color="gray"
    ).update_traces(marker=dict(
        color='green'), selector={'type': 'histogram'}
    )
    figScatter_h2_SoG_SoGA.update_xaxes(range=[5, 95])
    figScatter_h2_SoG_SoGA.update_layout(
        title_text='Halftime 2: Shots on Goal - Shots on Goal Opponent', title_x=title_x,
        yaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            title="Goal difference"
        )
    )
    figScatter_h2_SoG_SoGA.update_layout(legend=dict(
        yanchor="top",
        y=1.2,
        xanchor="right",
        x=1.12
    ))


    figScatter_h2_SoGA_SoG = px.scatter(
        # .query(f'Date.between{end_date}'),
        df4CompleteGraph[df4CompleteGraph["halftime"] == "2"].sort_values("IsHome", ascending=False),
        x='BP-H',
        y='GoalDiff',
        marginal_x="histogram",
        hover_data=['H_Red Cards', 'A_Red Cards', 'Date'],
        size="SoG-A-SoG-H",
        symbol = 'IsHome',
        symbol_sequence= ['circle-x', 'circle'],
        text="Opponent",
        width=widthfig,
    ).update_traces(textposition='top center', marker=dict(
        color='red'), selector={'type': 'scatter'}, textfont_size=9, textfont_color="gray"
    ).update_traces(marker=dict(
        color='red'), selector={'type': 'histogram'}
    )
    figScatter_h2_SoGA_SoG.update_xaxes(range=[5, 95])
    figScatter_h2_SoGA_SoG.update_layout(
        title_text='Halftime 2: Shots on Goal Opponent - Shots on Goal', title_x=title_x,
        yaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            title="Goal difference"
        ))
    figScatter_h2_SoGA_SoG.update_layout(legend=dict(
        yanchor="top",
        y=1.2,
        xanchor="right",
        x=1.12
    ))

    naming1x2 = {"W": "Win", "D": "Draw", "L": "Loss"}
    df4CompleteGraph['Halftime result'] = df4CompleteGraph['1x2'].replace(
        naming1x2)

    try:
        highest_count_yaxis = df4CompleteGraph.groupby(["BPTypes", "halftime"]).agg(
            'count').sort_values("Opponent", ascending=False).iloc[0].Home
    except:
        highest_count_yaxis = 0

    df4CompleteGraph['count'] = 1
    df4CompleteGraph_ht1 = df4CompleteGraph[df4CompleteGraph["halftime"] == "1"]
    df4CompleteGraph_ht2 = df4CompleteGraph[df4CompleteGraph["halftime"] == "2"]
    required_categories = ['<45', '45-55', '>55']
    for category in required_categories:
        if category not in df4CompleteGraph_ht1['BPTypes'].unique():
            # Append a row with the missing category and a count of 0
            df4CompleteGraph = df4CompleteGraph.append({
                'BPTypes': category, 
                'Halftime result': 'Draw',  
                'Home': '',  
                'Opponent': '',  
                'IsHome': 0, 
                'R': '',  
                'xG': 0.0,  
                'halftime': '1',
                'A_xG': 0.0,
                'count': 0
            }, ignore_index=True)
        if category not in df4CompleteGraph_ht2['BPTypes'].unique():
            # Append a row with the missing category and a count of 0
            df4CompleteGraph = df4CompleteGraph.append({
                'BPTypes': category, 
                'Halftime result': 'Draw',  
                'Home': '',  
                'Opponent': '',  
                'IsHome': 0, 
                'R': '',  
                'xG': 0.0,  
                'halftime': '2',
                'A_xG': 0.0,
                'count': 0
            }, ignore_index=True)


    # Create data for histogram 2
    BarBallpossesionstylesResultsHalftime1 = px.bar(
        df4CompleteGraph[df4CompleteGraph["halftime"] == "1"],
        x='BPTypes',
        y='count',
        # text=df4CompleteGraph.index,
        color='Halftime result',
        color_discrete_map={"Win": "green", "Draw": "gray", "Loss": "red"},
        width=widthfig,
        # height=heightfig,
        text="Opponent",
    )
    BarBallpossesionstylesResultsHalftime1.update_xaxes(categoryorder="array", categoryarray=['<45', '45-55', '>55'],).update_yaxes(
        range=[0, highest_count_yaxis])

    BarBallpossesionstylesResultsHalftime1.update_layout(
        title_text='Ballpossesionstyles - results halftime 1', title_x=title_x, xaxis=dict(
            tickmode='array', showticklabels=True,
        )
    )

    # Create data for histogram 2
    BarBallpossesionstylesResultsHalftime2 = px.bar(
        df4CompleteGraph[df4CompleteGraph["halftime"] == "2"],
        x='BPTypes',
        y='count',
        # text=df4CompleteGraph.index,
        color='Halftime result',
        color_discrete_map={"Win": "green", "Draw": "gray", "Loss": "red"},
        width=widthfig,
        # height=heightfig,
        text="Opponent",
    )
    BarBallpossesionstylesResultsHalftime2.update_xaxes(categoryorder="array", categoryarray=['<45', '45-55', '>55']).update_yaxes(
        range=[0, highest_count_yaxis])
    BarBallpossesionstylesResultsHalftime2.update_layout(
        title_text='Ballpossesionstyles - results halftime 2', title_x=title_x, xaxis=dict(
            tickmode='array', showticklabels=True,
        )
    )

    print(df4CompleteGraph.columns)
    print(df4CompleteGraph[["xG","A_xG", "xg_halftime", "Axg_halftime","halftime","Opponent",'Halftime result',"timestamp"]])

    df4CompleteGraph.xg_halftime = df4CompleteGraph.xg_halftime.astype(float).fillna(0.0)
    df4CompleteGraph.Axg_halftime = df4CompleteGraph.Axg_halftime.astype(float).fillna(0.0) 

    # all xg values for both halftimes!
    df4CompleteGraph["xG-A_xG"] = df4CompleteGraph["xG"] - df4CompleteGraph["A_xG"]
    df4CompleteGraph["A_xG-xG"] = df4CompleteGraph["A_xG"] - df4CompleteGraph["xG"]
    # all values for first half
    df4CompleteGraph["xg_halftime-Axg_halftime"] = df4CompleteGraph["xg_halftime"] - df4CompleteGraph["Axg_halftime"]
    df4CompleteGraph["Axg_halftime-xg_halftime"] = df4CompleteGraph["Axg_halftime"] - df4CompleteGraph["xg_halftime"]
   
    # all values for second half
    df4CompleteGraph["diff_xg_fulltime-diff_xg_halftime"] = df4CompleteGraph["xG-A_xG"] - df4CompleteGraph["xg_halftime-Axg_halftime"]
    df4CompleteGraph["diff_Axg_fulltime-diff_Axg_halftime"] = df4CompleteGraph["A_xG-xG"] - df4CompleteGraph["Axg_halftime-xg_halftime"]
 
    # formatting
    df4CompleteGraph["diff_xg_fulltime-diff_xg_halftime"] = df4CompleteGraph["diff_xg_fulltime-diff_xg_halftime"].clip(lower=0)
    df4CompleteGraph["diff_xg_fulltime-diff_xg_halftime"] = df4CompleteGraph["diff_xg_fulltime-diff_xg_halftime"].round(2)
    df4CompleteGraph["diff_Axg_fulltime-diff_Axg_halftime"] = df4CompleteGraph["diff_Axg_fulltime-diff_Axg_halftime"].clip(lower=0)
    df4CompleteGraph["diff_Axg_fulltime-diff_Axg_halftime"] = df4CompleteGraph["diff_Axg_fulltime-diff_Axg_halftime"].round(2)
    df4CompleteGraph["xg_halftime-Axg_halftime"] = df4CompleteGraph["xg_halftime-Axg_halftime"].clip(lower=0)
    df4CompleteGraph["xg_halftime-Axg_halftime"] = df4CompleteGraph["xg_halftime-Axg_halftime"].round(2)
    df4CompleteGraph["Axg_halftime-xg_halftime"] = df4CompleteGraph["Axg_halftime-xg_halftime"].clip(lower=0)
    df4CompleteGraph["Axg_halftime-xg_halftime"] = df4CompleteGraph["Axg_halftime-xg_halftime"].round(2)
    df4CompleteGraph["xG-A_xG"] = df4CompleteGraph["xG-A_xG"].clip(lower=0)
    df4CompleteGraph["xG-A_xG"] = df4CompleteGraph["xG-A_xG"].round(2)
    df4CompleteGraph["A_xG-xG"] = df4CompleteGraph["A_xG-xG"].clip(lower=0)
    df4CompleteGraph["A_xG-xG"] = df4CompleteGraph["A_xG-xG"].round(2)


    print(df4CompleteGraph[["xG","A_xG", "xg_halftime", "Axg_halftime","halftime","Opponent",'Halftime result',"timestamp"]])
    ht1 = df4CompleteGraph[df4CompleteGraph["halftime"] == "1"]
    print(ht1[["IsHome","xG","A_xG", "xg_halftime", "Axg_halftime","halftime","Opponent",'Halftime result',"timestamp"]])
    ht2 = df4CompleteGraph[df4CompleteGraph["halftime"] == "2"]
    print(ht2[["IsHome","xG","A_xG", "xg_halftime", "Axg_halftime","halftime","Opponent",'Halftime result',"timestamp"]])

    # Create barchart for xg per bptypes 1
    BarBallpossesionstylesXGHalftime1 = px.bar(
        df4CompleteGraph[df4CompleteGraph["halftime"] == "1"],
        x='BPTypes',
        y=['xg_halftime-Axg_halftime', 'Axg_halftime-xg_halftime'],
        barmode='group',
        # text=df4CompleteGraph.index,
        # title="BP-Styles - Halftimes",
        # color='Halftime result',
        color_discrete_map={"xg_halftime-Axg_halftime": "green", "Axg_halftime-xg_halftime": "red"},
        width=widthfig,
        # height=heightfig,
        # opacity=0.5,
        text="Opponent",
    ).update_xaxes(categoryorder="array", categoryarray=['<45', '45-55', '>55']).update_yaxes(
        range=[0, highest_count_yaxis])
    BarBallpossesionstylesXGHalftime1.update_layout(
        title_text='Ballpossesionstyles - xG halftime 1', title_x=title_x, xaxis=dict(
            tickmode='array', showticklabels=True,
        )
    )
    BarBallpossesionstylesXGHalftime1.update_layout(legend=dict(
        yanchor="top",
        y=1.2,
        xanchor="right",
        x=1.12
    ))


    # Create barchart for xg per bptypes 2
    BarBallpossesionstylesXGHalftime2 = px.bar(
        df4CompleteGraph[df4CompleteGraph["halftime"] == "2"],
        x='BPTypes',
        y=['diff_xg_fulltime-diff_xg_halftime', 'diff_Axg_fulltime-diff_Axg_halftime'],
        barmode='group',
        # text=df4CompleteGraph.index,
        # title="BP-Styles - Halftimes",
        # color='Halftime result',
        color_discrete_map={"diff_xg_fulltime-diff_xg_halftime": "green", "diff_Axg_fulltime-diff_Axg_halftime": "red"},
        width=widthfig,
        # height=heightfig,
        # opacity=0.5,
        text="Opponent",
    ).update_xaxes(categoryorder="array", categoryarray=['<45', '45-55', '>55']).update_yaxes(
        range=[0, highest_count_yaxis])
    BarBallpossesionstylesXGHalftime2.update_layout(
        title_text='Ballpossesionstyles - xG halftime 2', title_x=title_x, xaxis=dict(
            tickmode='array', showticklabels=True,
        )
    )
    BarBallpossesionstylesXGHalftime2.update_layout(legend=dict(
        yanchor="top",
        y=1.2,
        xanchor="right",
        x=1.12
    ))

    figHistogramxG_A_xG_1Ht = px.scatter(
        df4CompleteGraph[df4CompleteGraph["halftime"] == "1"],  # .query(f'Date.between{end_date}'),
        x='BP-H',
        y='GoalDiff',
        marginal_x="histogram",
        color="timestamp",
        hover_data=['H_Red Cards', 'A_Red Cards', 'Date'],
        size="xg_halftime-Axg_halftime",
        symbol = 'IsHome',
        symbol_sequence= ['diamond-cross', 'diamond'],
        text="Opponent",
        width=widthfig,
        # height=heightfig,
        # color_continuous_scale= 'Viridis',
        # facet_row="time", # makes seperate plot for value
        # marginal_x="histogram",
    ).update_traces(textposition='top center', selector={'type': 'scatter'}, textfont_size=9, textfont_color="gray" ).update_traces(
        marker=dict(color='green'), selector={'type': 'histogram'}
    )
    figHistogramxG_A_xG_1Ht.update_xaxes(range=[5, 95])
    figHistogramxG_A_xG_1Ht.update_layout(
        title_text='Ht1: Expectedgoals - Expectedgoals Opponent', title_x=title_x,
        yaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            title="Goal difference"
        ))
    figHistogramxG_A_xG_1Ht.update_layout(legend=dict(
        yanchor="top",
        y=1.2,
        xanchor="right",
        x=1.12
    ))

    figHistogramA_xG_xG_1Ht = px.scatter(
        df4CompleteGraph[df4CompleteGraph["halftime"] == "1"],  # .query(f'Date.between{end_date}'),
        x='BP-H',
        y='GoalDiff',
        marginal_x="histogram",
        color="timestamp",
        hover_data=['H_Red Cards', 'A_Red Cards', 'Date'],
        size="Axg_halftime-xg_halftime",
        symbol = 'IsHome',
        symbol_sequence= ['circle-x', 'circle'],
        text="Opponent",
        width=widthfig,
        # height=heightfig,
        # facet_row="time", # makes seperate plot for value
        # marginal_x="histogram",
    ).update_traces(textposition='top center', selector={'type': 'scatter'}, textfont_size=9, textfont_color="gray" ).update_traces(
        marker=dict(color='red'), selector={'type': 'histogram'}
    )
    figHistogramA_xG_xG_1Ht.update_xaxes(range=[5, 95])
    figHistogramA_xG_xG_1Ht.update_layout(
        title_text='Ht1: Expectedgoals Opponent - Expectedgoals', title_x=title_x,
        yaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            title="Goal difference"
        ))
    figHistogramA_xG_xG_1Ht.update_layout(legend=dict(
        yanchor="top",
        y=1.2,
        xanchor="right",
        x=1.12
    ))


    # Determine if there was a red card in a game
    df4CompleteGraph['red_card'] = df4CompleteGraph[['H_Red Cards', 'A_Red Cards']].apply(lambda x: x[0] > 0 or x[1] > 0, axis=1)
    # Define marker properties based on the presence of a red card
    df4CompleteGraph['marker_properties'] = df4CompleteGraph['red_card'].apply(
        lambda x: {'line_width': 2, 'line_color': 'red'} if x else {'line_width': 1, 'line_color': 'black'}
    )

    figHistogramxG_A_xG_2Ht = px.scatter(
        df4CompleteGraph[df4CompleteGraph["halftime"] == "2"],  # .query(f'Date.between{end_date}'),
        x='BP-H',
        y='GoalDiff',
        marginal_x="histogram",
        color="timestamp",
        hover_data=['H_Red Cards', 'A_Red Cards', 'Date'],
        size="diff_xg_fulltime-diff_xg_halftime",
        symbol = 'IsHome',
        symbol_sequence= ['diamond-cross', 'diamond'],
        text="Opponent",
        width=widthfig,
        # height=heightfig,
        # color_continuous_scale= 'Viridis',
        # facet_row="time", # makes seperate plot for value
        # marginal_x="histogram",
    ).update_traces(textposition='top center', selector={'type': 'scatter'},textfont_size=9, textfont_color="gray" ).update_traces(
        marker=dict(color='green'), selector={'type': 'histogram'}
    )
    figHistogramxG_A_xG_2Ht.update_xaxes(range=[5, 95])
    figHistogramxG_A_xG_2Ht.update_layout(
        title_text='Ht2: Expectedgoals - Expectedgoals Opponent', title_x=title_x,
        yaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            title="Goal difference"
        ))
    figHistogramxG_A_xG_2Ht.update_layout(legend=dict(
        yanchor="top",
        y=1.2,
        xanchor="right",
        x=1.12
    ))

    figHistogramA_xG_xG_2Ht = px.scatter(
        df4CompleteGraph[df4CompleteGraph["halftime"] == "2"],  # .query(f'Date.between{end_date}'),
        x='BP-H',
        y='GoalDiff',
        marginal_x="histogram",
        color="timestamp",
        hover_data=['H_Red Cards', 'A_Red Cards', 'Date'],
        size="diff_Axg_fulltime-diff_Axg_halftime",
        symbol = 'IsHome',
        symbol_sequence= ['circle-x', 'circle'],
        text="Opponent",
        width=widthfig,
        # height=heightfig,
        # facet_row="time", # makes seperate plot for value
        # marginal_x="histogram",
    ).update_traces(textposition='top center', selector={'type': 'scatter'},textfont_size=9, textfont_color="gray" ).update_traces(
        marker=dict(color='red'), selector={'type': 'histogram'}
    )
    figHistogramA_xG_xG_2Ht.update_xaxes(range=[5, 95])
    figHistogramA_xG_xG_2Ht.update_layout(
        title_text='Ht2: Expectedgoals Opponent - Expectedgoals', title_x=title_x,
        yaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            title="Goal difference"
        ))
    figHistogramA_xG_xG_2Ht.update_layout(legend=dict(
        yanchor="top",
        y=1.2,
        xanchor="right",
        x=1.12
    ))


    st.title("Football statistics - {}".format(team))
    st.markdown('The following two diagrams display the new metric Expected Goals (**xGoals**), which is a qualitative measurement on base of the shots on goal.  \nThe expected goal model shows how high the chance of the goal really was and calculates a value for each completion based on several factors.   \nF.I. a penalty has generally a probably of 75 % to result in a goal, which would increase the xGoal value for 0.75 regardless of the penalty-outcame in this case.', unsafe_allow_html=False)

    col1, col2 = st.columns(2)

    col1.plotly_chart(BarBallpossesionstylesResultsHalftime1)
    col2.plotly_chart(BarBallpossesionstylesResultsHalftime2)
    col1.plotly_chart(BarBallpossesionstylesXGHalftime1)
    col2.plotly_chart(BarBallpossesionstylesXGHalftime2)

    col1.plotly_chart(figHistogramxG_A_xG_1Ht)
    col1.plotly_chart(figHistogramA_xG_xG_1Ht)
    col2.plotly_chart(figHistogramxG_A_xG_2Ht)
    col2.plotly_chart(figHistogramA_xG_xG_2Ht)

    col1.plotly_chart(figScatter_SoG_SoGA)
    col2.plotly_chart(figScatter_SoGA_soG)
    col1.plotly_chart(figScatter_h1_soG_SoGA)
    col1.plotly_chart(figScatter_h1_SoGA_soG)
    col2.plotly_chart(figScatter_h2_SoG_SoGA)
    col2.plotly_chart(figScatter_h2_SoGA_SoG)

    st.plotly_chart(fig_xg_perminute_home)
    st.plotly_chart(fig_xg_homexg_complete_game_all_bpse)
    st.plotly_chart(fig_xg_perminute_home_bigger_55)
    st.plotly_chart(fig_xg_perminute_home_smaller_45)

    # C_WPercText, N_WPercText, BP_WPercText = calc_stats(df4Complete)
    # col2.write("% W < 0.45:   {}   \n % W 0.45 - 0.55:  {}   \n % W > 0.55:  {}".format(
    #     C_WPercText, N_WPercText, BP_WPercText))

    # show df
    st.dataframe(df4Complete_show.style.format({'xG': '{:.1f}', 'A_xG': '{:.1f}', 'SoG-H': '{:.0f}',
                                                'G-H': '{:.0f}', 'G-A': '{:.0f}', 'BP-H': '{:.0f}',
                                                'BP-A': '{:.0f}', 'GA-H': '{:.0f}', 'GA-A': '{:.0f}',
                                                'xPTS': '{:.1f}', 'A_xPTS': '{:.1f}', 'SoG-A': '{:.0f}',
                                                }))




# Erstelle ein Seitenleisten-Menü
page = st.sidebar.radio("Choose a page:", ( 'League Tables', 'Team Analyis'))

# Navigiere zur ausgewählten Seite
if page == 'Team Analyis':
    page_teamx()
elif page == 'League Tables':
    page_league_table()