# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
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
st.set_page_config(layout="wide")
pd.options.display.float_format = "{:,.1f}".format
warnings.filterwarnings('ignore')


# TEAMNAMES value in teamnamedict must match the htdatan teamname!

global teamnamedict
# C:\Users\mg\JupyterLabDir\Rest\Pr Winning\teamnamedict_streamlit.json
with open('teamnamedict_streamlit.json') as f:
    teamnamedict = json.load(f)

# def _max_width_():
#     max_width_str = f"max-width: 1300px;"
#     st.markdown(
#         f"""
#     <style>
#     .reportview-container .main .block-container{{
#         {max_width_str}
#     }}
#     </style>
#     """,
#         unsafe_allow_html=True,
#     )

# _max_width_()
global widthfig
widthfig = 700
heightfig = 600

# get all the gamestatistics from in dropdown specified league and season
# setup the database connection.  There's no need to setup cursors with pandas psql.
tables = list(glob.glob("htdatan/*"))

# take only the 0 part of the every list entry
global saissons
saissons = []

# ENV is BTCPRED
for x in range(0, len(tables)):    # CHANGE THIS - \\ - to - / - FOR DEPLOYMENT!
    saissons.append(tables[x].split("/")[1].split("_24102021.csv")[0])


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
# map names back for reading the correct csv name


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
        "htdatan/"+saison+".csv", index_col=0, encoding='utf-8')
except:
    df_complete_saison = pd.read_csv(
        "htdatan/"+saison+"_24102021.csv", index_col=0, encoding='utf-8')

df_complete_saison = df_complete_saison.replace(teamnamedict)
dfallteamnamesl = df_complete_saison.H_Teamnames.unique()

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

# return df, team, saison


def process_team_names_of_df(x_df):
    x_df = x_df.replace(teamnamedict)
    return x_df

#######################################################
###  calculate table with two halftimes to one game ###
#######################################################


# def convert_hts_to_complete_games(df):
#     # fill nane values of these not numeric columns
#     df[['FK-H', 'FK-A']].fillna(0)
#     # convert not numeric columns to numeric columns
#     df['FK-H'] = df['FK-H'].astype('float64')
#     df['FK-A'] = df['FK-A'].astype('float64')
#     df['C-H'] = df['C-H'].astype('float64')
#     df['F-H'] = df['F-H'].astype('float64')
#     df['GA-H'] = df['GA-H'].astype('float64')
#     df['SoffG-H'] = df['SoffG-H'].astype('float64')
#     df['SoG-H'] = df['SoG-H'].astype('float64')
#     df['C-A'] = df['C-A'].astype('float64')
#     df['F-A'] = df['F-A'].astype('float64')
#     df['GA-A'] = df['GA-A'].astype('float64')
#     df['SoffG-A'] = df['SoffG-A'].astype('float64')
#     df['SoG-A'] = df['SoG-A'].astype('float64')
#     df['G-H'] = df['G-H'].astype('float64')
#     df['G-A'] = df['G-A'].astype('float64')
#     df['BP-H'] = df['BP-H'].astype('float64')
#     df['BP-A'] = df['BP-A'].astype('float64')
#     # xGoals columns
#     if not set(['xG', 'xPTS', 'GOALS', 'A_xG', 'G-A', 'A_xPTS']).issubset(df.columns):
#         df['xG'] = -1.0
#         df['GOALS'] = -1.0
#         df['xPTS'] = -1.0
#         df['A_xG'] = -1.0
#         df['G-A'] = -1.0
#         df['A_xPTS'] = -1.0
#     else:
#         df['xG'] = df['xG'].astype('float64')
#         df['GOALS'] = df['GOALS'].astype('float64')
#         df['xPTS'] = df['xPTS'].astype('float64')
#         df['A_xG'] = df['A_xG'].astype('float64')
#         df['G-A'] = df['G-A'].astype('float64')
#         df['A_xPTS'] = df['A_xPTS'].astype('float64')

#     # calculate halftime table to fulltime table
#     df = df.groupby(['Home', 'Opponent', 'Date', 'Round']).agg({'BP-H': 'mean', 'C-H': 'sum',
#                                                                 'F-H': 'sum', 'FK-H': 'sum', 'GA-H': 'sum',
#                                                                 'GoKeSa-H': 'sum', 'G-H': 'sum', 'Off-H': 'sum',
#                                                                 'SoffG-H': 'sum', 'SoG-H': 'sum',
#                                                                            'BP-A': 'mean',
#                                                                            'C-A': 'sum',
#                                                                            'F-A': 'sum', 'FK-A': 'sum', 'GA-A': 'sum',
#                                                                            'GoKeSa-A': 'sum', 'G-A': 'sum', 'Off-A': 'sum',
#                                                                            'SoffG-A': 'sum', 'SoG-A': 'sum',
#                                                                            # xGoals stats are only from whole game - not halftime - so mean does not change anything
#                                                                            'xG': 'mean',
#                                                                            'GOALS': 'mean',
#                                                                            'xPTS': 'mean',
#                                                                            'A_xG': 'mean',
#                                                                            'A_GOALS': 'mean',
#                                                                            'A_xPTS': 'mean'
#                                                                 }).reset_index()

#     newcols = []

#     for x in df.columns:
#         if x.startswith('SUM') or x.startswith('MIN') or x.startswith('AVG'):
#             x = re.sub('SUM', '', x)
#             x = re.sub('MIN', '', x)
#             x = re.sub('AVG', '', x)
#             x = x.replace("`", "")
#             x = x.replace(")", "")
#             x = x.replace("(", "")
#         newcols.append(x)

#     df.columns = newcols

#     return df


def df_cleaning_converting(df):
    df = df[['H_Teamnames', 'A_Teamnames', 'H_Goals', 'A_Goals', 'H_Ball Possession', 'A_Ball Possession', 'A_Goal Attempts', 'H_Goal Attempts',
             'H_Shots on Goal', 'A_Shots on Goal', 'H_Shots off Goal', 'A_Shots off Goal', 'H_Free Kicks',"H_Red Cards", "A_Red Cards",
             'A_Free Kicks', 'H_Corner Kicks', 'A_Corner Kicks', 'H_Offsides', 'A_Offsides', 'H_Goalkeeper Saves', 'A_Goalkeeper Saves',
             'H_Fouls', 'A_Fouls', 'A_gameinfo', 'A_datetime', 'xG', 'GOALS', 'xPTS', 'A_xG', 'A_GOALS', 'A_xPTS', 'timing_chart_xg', "homexg_complete_game", "awayxg_complete_game"]]
    df = df.drop_duplicates(subset=['H_Teamnames', 'A_Teamnames', 'H_Goals', 'A_Goals', 'H_Ball Possession', 'A_Ball Possession', 'A_Goal Attempts', 'H_Goal Attempts',
             'H_Shots on Goal', 'A_Shots on Goal', 'H_Shots off Goal', 'A_Shots off Goal', 'H_Free Kicks',"H_Red Cards", "A_Red Cards",
             'A_Free Kicks', 'H_Corner Kicks', 'A_Corner Kicks', 'H_Offsides', 'A_Offsides', 'H_Goalkeeper Saves', 'A_Goalkeeper Saves',
             'H_Fouls', 'A_Fouls', 'A_gameinfo', 'A_datetime', 'xG', 'GOALS', 'xPTS', 'A_xG', 'A_GOALS', 'A_xPTS'], keep='first')
    df = df.reset_index(drop=True)
    df["R"] = 'X'

    df['timing_chart_xg'] = df['timing_chart_xg'].astype('str') 

    # calculate halftime xG for both teams!
    df['xg_halftime'] = -1
    df['Axg_halftime'] = -1
    for index, row in df.iterrows():
        print(index)
        try:    
            # away team xg at halfime!
            df['Axg_halftime'].loc[index] = df["timing_chart_xg"].loc[index].split("45' ")[0].split("Total xG: ")[-1].split("\n")[0].replace(";", "")  
            # home team xg at halfime!
            df['xg_halftime'].loc[index] = df["timing_chart_xg"].loc[index].split("45' ")[1].split("Total xG: ")[1].split("\n")[0].replace(";", "") 
        except Exception:
            print("WRONG! index: ", index)
            # print(row)
            # print(traceback.format_exc())

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
                'F-H', 'F-A', 'Round', 'Date', 'xG', 'GOALS', 'xPTS', 'A_xG', 'A_GOALS', 'A_xPTS', "timing_chart_xg", "homexg_complete_game", "awayxg_complete_game", 'R',  'xg_halftime', 'Axg_halftime',]

    df = df[['Home', 'Opponent', 'R', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A',
             'SoG-H', 'SoG-A', 'SoffG-H', 'SoffG-A', 'FK-H',"H_Red Cards", "A_Red Cards",
             'FK-A', 'C-H', 'C-A', 'Off-H', 'Off-A', 'GoKeSa-H', 'GoKeSa-A', 'F-H',
             'F-A', 'Round', 'Date', 'xG', 'GOALS', 'xPTS', 'A_xG', 'A_GOALS', 'A_xPTS', 'xg_halftime', 'Axg_halftime', "homexg_complete_game", "awayxg_complete_game", ]]

    df["IsHome"] = 0

    df = df[['Home', 'Opponent', 'R', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A',
             'SoG-H', 'SoG-A', 'SoffG-H', 'SoffG-A', 'FK-H',"H_Red Cards", "A_Red Cards",
             'FK-A', 'C-H', 'C-A', 'Off-H', 'Off-A', 'GoKeSa-H', 'GoKeSa-A', 'F-H',
             'F-A', 'Round', 'Date', 'IsHome', 'xG', 'GOALS', 'xPTS', 'A_xG', 'A_GOALS', 'A_xPTS','xg_halftime', 'Axg_halftime', "homexg_complete_game", "awayxg_complete_game", ]]
    return df


@st.cache(suppress_st_warning=True)
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

@st.cache(suppress_st_warning=True)
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


@st.cache
def calculate_1x2_home(row):
    if row['G-H'] > row['G-A']:
        return 'W'
    elif row['G-A'] > row['G-H']:
        return 'L'
    else:
        return 'D'


@st.cache
def calculate_1x2_Opponent(row):
    if row['G-A'] > row['G-H']:
        return 'W'
    elif row['G-H'] > row['G-A']:
        return 'L'
    else:
        return 'D'


@st.cache
def calculate_1x2_BPTypes(row):
    if row['BP-H'] > 55:
        return '>55'
    elif row['BP-H'] < 45:
        return '<45'
    else:
        return '45-55'


@st.cache
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
        "xg/"+xgtablename+".csv", index_col=0, encoding='utf-8')

    df_complete_saison = process_team_names_of_df(df_complete_saison)

    # execute the query and assign it to a pandas dataframe
    dfxg = df_complete_saison[(df_complete_saison.TEAMS == team) | (
        df_complete_saison.A_TEAMS == team)]

    return dfxg


df = process_team_names_of_df(df)

dfxg = load_xg_gamestats_sql(saison, team)

# rename columns for
dfxg_rename = dfxg.rename(
    columns={'TEAMS': 'H_Teamnames', 'A_TEAMS': 'A_Teamnames'})
# del dfxg

dfxg_df_merged = pd.merge(
    df, dfxg_rename, on=["H_Teamnames", "A_Teamnames"])
dfxg_df_merged = dfxg_df_merged.drop_duplicates()

# %%
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

    for x in range(start_min_game,int(last_game_minute)+1):
        homexgperminute = df["timing_chart_xg"].loc[game_loc].split("{}' Total xG: ".format(x))[1].split(";")[0][:4]  # [:4] - only last 4 digits so no goalscorer infos
        awayxgperminute = df["timing_chart_xg"].loc[game_loc].split("{}' Total xG: ".format(x), 2)[2].split(";")[0][:4]
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

df4Complete[['xG', 'A_xG', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A', 'SoG-H', 'SoG-A',  'xPTS', 'A_xPTS',  "A_Red Cards", "H_Red Cards"]] = df4Complete[['xG',
                                                                                                                              'A_xG', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A', 'SoG-H', 'SoG-A',  'xPTS', 'A_xPTS',  "A_Red Cards", "H_Red Cards"]].apply(pd.to_numeric, errors='coerce', axis=1)

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

# calculate the y axis to display for the xg per minute per ball position
if dfxg_homexg_complete_game.mean().max() > dfxg_awayxg_complete_game.mean().max():
    dfxg_y_axis_max = dfxg_homexg_complete_game.mean().max()
else:
    dfxg_y_axis_max = dfxg_awayxg_complete_game.mean().max()

fig_xg_perminute_home = px.line(
    dfxg_homexg_complete_game.mean(),
    width=widthfig,
).update_traces(textposition='top center', selector={'type': 'scatter'}).update_traces(
    marker=dict(color='green'), selector={'type': 'histogram'}
)
fig_xg_perminute_home.add_scatter(y=dfxg_awayxg_complete_game.mean(), mode='lines', name='Opponent xG')
fig_xg_perminute_home.update_layout(
    title_text='Expectedgoals per minute: {} < bp < {}'.format(int(bigger_bp), int(smaller_bp)), title_x=0.5,
    yaxis=dict(
        title="xG"
    ))
fig_xg_perminute_home.update_yaxes(range=[0, dfxg_y_axis_max+0.02])
# Only thing I figured is - I could do this 

fig_xg_homexg_complete_game_all_bpse = px.line(
    dfxg_homexg_complete_game_all_bps.mean(),
    width=widthfig,
).update_traces(textposition='top center', selector={'type': 'scatter'}).update_traces(
    marker=dict(color='green'), selector={'type': 'histogram'}
)
fig_xg_homexg_complete_game_all_bpse.add_scatter(y=dfxg_awayxg_complete_game_all_bps.mean(), mode='lines', name='Opponent xG')
fig_xg_homexg_complete_game_all_bpse.update_layout(
    title_text='Expectedgoals per minute', title_x=0.5,
    yaxis=dict(
        title="xG"
    ))
fig_xg_homexg_complete_game_all_bpse.update_yaxes(range=[0, dfxg_awayxg_complete_game_all_bps.mean()+0.02])


fig_xg_perminute_home_bigger_55 = px.line(
    dfxg_homexg_complete_game_bigger_55.mean(),
    width=widthfig,
).update_traces(textposition='top center', selector={'type': 'scatter'}).update_traces(
    marker=dict(color='green'), selector={'type': 'histogram'}
)
fig_xg_perminute_home_bigger_55.add_scatter(y=dfxg_awayxg_complete_game_bigger_55.mean(), mode='lines', name='Opponent xG')
fig_xg_perminute_home_bigger_55.update_layout(
    title_text='Expectedgoals per minute: bp > 55', title_x=0.5,
    yaxis=dict(
        title="xG"
    ))
fig_xg_perminute_home_bigger_55.update_yaxes(range=[0, dfxg_awayxg_complete_game_bigger_55.mean()+0.02])

fig_xg_perminute_home_smaller_45 = px.line(
    dfxg_homexg_complete_game_smaller_45.mean(),
    width=widthfig,
).update_traces(textposition='top center', selector={'type': 'scatter'}).update_traces(
    marker=dict(color='green'), selector={'type': 'histogram'}
)
fig_xg_perminute_home_smaller_45.add_scatter(y=dfxg_awayxg_complete_game_smaller_45.mean(), mode='lines', name='Opponent xG')
fig_xg_perminute_home_smaller_45.update_layout(
    title_text='Expectedgoals per minute: bp < 45', title_x=0.5,
    yaxis=dict(
        title="xG"
    ))
fig_xg_perminute_home_smaller_45.update_yaxes(range=[0, dfxg_awayxg_complete_game_smaller_45.mean()+0.02])

figScatter = px.scatter(
    df4CompleteGraph.sort_values("IsHome", ascending=False),  # .query(f'Date.between{end_date}'),
    x='BP-H',
    y='GoalDiff',
    marginal_x="histogram",
    color="timestamp",
    hover_data=['H_Red Cards', 'A_Red Cards'],
    symbol = 'IsHome',
    symbol_sequence= ['diamond-cross', 'diamond'],
    size="SoG-H-SoG-A",
    text="Opponent",
    width=widthfig,
    # height=heightfig,
    # title="SoGH-SoGA - Halftimes",
    # color_continuous_scale= 'Viridis',
    # color_discrete_map={"W": "green", "D": "gray", "L": "red"}

    # facet_row="time", # makes seperate plot for value
    # marginal_x="histogram",
).update_traces(textposition='top center', selector={'type': 'scatter'}).update_traces(
    marker=dict(color='green'), selector={'type': 'histogram'}
)
figScatter.update_xaxes(range=[5, 95])
figScatter.update_layout(
    title_text='All halftimes: Shots on Goal - Shots on Goal Opponent', title_x=0.5,
    yaxis=dict(
        tickmode='linear',
        tick0=1,
        dtick=1,
        title="Goal difference"
    ))
figScatter.update_layout(legend=dict(
    yanchor="top",
    y=1.2,
    xanchor="right",
    x=1.12
))

figScatter1 = px.scatter(
    df4CompleteGraph.sort_values("IsHome", ascending=False),  # .query(f'Date.between{end_date}'),
    x='BP-H',
    y='GoalDiff',
    marginal_x="histogram",
    color="timestamp",
    hover_data=['H_Red Cards', 'A_Red Cards'],
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
    textposition='top center', selector={'type': 'scatter'}).update_traces(
        marker=dict(color='red'), selector={'type': 'histogram'}
)
figScatter1.update_xaxes(range=[5, 95])
figScatter1.update_layout(
    title_text='All halftimes: Shots on Goal Opponent - Shots on Goal', title_x=0.5,
    yaxis=dict(
        tickmode='linear',
        tick0=1,
        dtick=1,
        title="Goal difference"
    )
)
figScatter1.update_layout(legend=dict(
    yanchor="top",
    y=1.2,
    xanchor="right",
    x=1.12
))


# wie oft passierts das team hinten ist und noch gewinnt / nicht verliert:´
# sortier nach sieg x loss
# mach diagramm halftimes für 1te und 2te hz

# delete games where there is no two halftimes!
df4CompleteGraph = df4CompleteGraph[df4CompleteGraph.groupby(
    'Opponent')['Opponent'].transform('size') >= 2]
df4CompleteGraph = df4CompleteGraph.sort_index()
# second half is second entry always!
df4CompleteGraph["halftime"] = "0"
df4CompleteGraph.iloc[::2]["halftime"] = "2"
df4CompleteGraph.iloc[1::2]["halftime"] = "1"

figScatter5 = px.scatter(
    # .query(f'Date.between{end_date}'),
    df4CompleteGraph[df4CompleteGraph["halftime"] == "1"].sort_values("IsHome", ascending=False),
    x='BP-H',
    y='GoalDiff',
    size="SoG-H-SoG-A",
    text="Opponent",
    hover_data=['H_Red Cards', 'A_Red Cards'],
    symbol = 'IsHome',
    symbol_sequence= ['diamond-cross', 'diamond'],
    width=widthfig,
    # height=heightfig,
    # color_continuous_scale= 'Viridis',
    # facet_row="time", # makes seperate plot for value
    marginal_x="histogram",
).update_traces(textposition='top center',  marker=dict(
    color='green'), selector={'type': 'scatter'}
).update_traces(marker=dict(
    color='green'), selector={'type': 'histogram'}
)
figScatter5.update_xaxes(range=[5, 95])
figScatter5.update_layout(
    title_text='Halftime 1: Shots on Goal - Shots on Goal Opponent', title_x=0.5,
    yaxis=dict(
        tickmode='linear',
        tick0=1,
        dtick=1,
        title="Goal difference"
    ))
figScatter5.update_layout(legend=dict(
    yanchor="top",
    y=1.2,
    xanchor="right",
    x=1.12
))


figScatter6 = px.scatter(
    # .query(f'Date.between{end_date}'),
    df4CompleteGraph[df4CompleteGraph["halftime"] == "1"].sort_values("IsHome", ascending=False),
    x='BP-H',
    y='GoalDiff',
    size="SoG-A-SoG-H",
    text="Opponent",
    hover_data=['H_Red Cards', 'A_Red Cards'],
    symbol = 'IsHome',
    symbol_sequence= ['circle-x', 'circle'],
    width=widthfig,
    # height=heightfig,
    # color_continuous_scale= 'Viridis',
    # facet_row="time", # makes seperate plot for value
    marginal_x="histogram",
).update_traces(textposition='top center', marker=dict(
    color='red'), selector={'type': 'scatter'}
).update_traces(marker=dict(
    color='red'), selector={'type': 'histogram'}
)
figScatter6.update_xaxes(range=[5, 95])
figScatter6.update_layout(
    title_text='Halftime 1: Shots on Goal Opponent - Shots on Goal', title_x=0.5,
    yaxis=dict(
        tickmode='linear',
        tick0=1,
        dtick=1,
        title="Goal difference"
    ))
figScatter6.update_layout(legend=dict(
    yanchor="top",
    y=1.2,
    xanchor="right",
    x=1.12
))


figScatter7 = px.scatter(
    # .query(f'Date.between{end_date}'),
    df4CompleteGraph[df4CompleteGraph["halftime"] == "2"].sort_values("IsHome", ascending=False),
    x='BP-H',
    y='GoalDiff',
    marginal_x="histogram",
    hover_data=['H_Red Cards', 'A_Red Cards'],
    size="SoG-H-SoG-A",
    symbol = 'IsHome',
    symbol_sequence= ['diamond-cross', 'diamond'],
    text="Opponent",
    width=widthfig,
    # height=heightfig,
    # color_continuous_scale= 'Viridis',
    # facet_row="time", # makes seperate plot for value
).update_traces(textposition='top center',  marker=dict(
    color='green'), selector={'type': 'scatter'}
).update_traces(marker=dict(
    color='green'), selector={'type': 'histogram'}
)
figScatter7.update_xaxes(range=[5, 95])
figScatter7.update_layout(
    title_text='Halftime 2: Shots on Goal - Shots on Goal Opponent', title_x=0.5,
    yaxis=dict(
        tickmode='linear',
        tick0=1,
        dtick=1,
        title="Goal difference"
    )
)
figScatter7.update_layout(legend=dict(
    yanchor="top",
    y=1.2,
    xanchor="right",
    x=1.12
))


figScatter8 = px.scatter(
    # .query(f'Date.between{end_date}'),
    df4CompleteGraph[df4CompleteGraph["halftime"] == "2"].sort_values("IsHome", ascending=False),
    x='BP-H',
    y='GoalDiff',
    marginal_x="histogram",
    hover_data=['H_Red Cards', 'A_Red Cards'],
    size="SoG-A-SoG-H",
    symbol = 'IsHome',
    symbol_sequence= ['circle-x', 'circle'],
    text="Opponent",
    width=widthfig,
    # height=heightfig,
    # color_continuous_scale= 'Viridis',
    # facet_row="time", # makes seperate plot for value
    # marginal_x="histogram",
).update_traces(textposition='top center', marker=dict(
    color='red'), selector={'type': 'scatter'}
).update_traces(marker=dict(
    color='red'), selector={'type': 'histogram'}
)
figScatter8.update_xaxes(range=[5, 95])
figScatter8.update_layout(
    title_text='Halftime 2: Shots on Goal Opponent - Shots on Goal', title_x=0.5,
    yaxis=dict(
        tickmode='linear',
        tick0=1,
        dtick=1,
        title="Goal difference"
    ))
figScatter8.update_layout(legend=dict(
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

# Create data for histogram 2
BarBallpossesionstylesResultsHalftime1 = px.bar(
    df4CompleteGraph[df4CompleteGraph["halftime"] == "1"],
    x='BPTypes',
    # text=df4CompleteGraph.index,
    # title="BP-Styles - Halftimes",
    color='Halftime result',
    color_discrete_map={"Win": "green", "Draw": "gray", "Loss": "red"},
    width=widthfig,
    # height=heightfig,
    # opacity=0.5,
    text="Opponent",
).update_xaxes(categoryorder="array", categoryarray=['<45', '45-55', '>55'],).update_yaxes(
    range=[0, highest_count_yaxis])

BarBallpossesionstylesResultsHalftime1.update_layout(
    title_text='Ballpossesionstyles - results halftime 1', title_x=0.5, xaxis=dict(
        tickmode='array', showticklabels=True,
    )
)

# Create data for histogram 2
BarBallpossesionstylesResultsHalftime2 = px.bar(
    df4CompleteGraph[df4CompleteGraph["halftime"] == "2"],
    x='BPTypes',
    # text=df4CompleteGraph.index,
    # title="BP-Styles - Halftimes",
    color='Halftime result',
    color_discrete_map={"Win": "green", "Draw": "gray", "Loss": "red"},
    width=widthfig,
    # height=heightfig,
    # opacity=0.5,
    text="Opponent",
).update_xaxes(categoryorder="array", categoryarray=['<45', '45-55', '>55']).update_yaxes(
    range=[0, highest_count_yaxis])
BarBallpossesionstylesResultsHalftime2.update_layout(
    title_text='Ballpossesionstyles - results halftime 2', title_x=0.5, xaxis=dict(
        tickmode='array', showticklabels=True,
    )
)


# # create scatterplot with XG - bubble size
# df4CompleteGraph = df4Complete.copy()

# df4CompleteGraph = convert_hts_to_complete_games(df4CompleteGraph)

# # Calculate again the stuff like for the single halftimes before!
# # GoalDifference
# df4CompleteGraph["GoalDiff"] = df4CompleteGraph["G-H"] - \
#     df4CompleteGraph["G-A"]
# df4CompleteGraph = df4CompleteGraph.sort_values("Date",  ascending=False)
# # calculate column with 3 Ballposition types
# df4CompleteGraph["BPTypes"] = '0'
# df4CompleteGraph["BPTypes"] = df4CompleteGraph.apply(
#     lambda row: calculate_1x2_BPTypes(row), axis=1, result_type='reduce')
# df4CompleteGraph['Date'] = pd.to_datetime(
#     df4CompleteGraph['Date'], format="%d.%m.%Y %H:%M")
# # convert datetime to timestamp for scatter visualization
# df4CompleteGraph['timestamp'] = df4CompleteGraph.Date.astype('int64')//10**9
# df4CompleteGraph = df4CompleteGraph.sort_values("Date", ascending=False)
# Create data for scatter graph
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
df4CompleteGraph["xg_halftime2-Axg_halftime2"] = df4CompleteGraph["xG-A_xG"] - df4CompleteGraph["xg_halftime-Axg_halftime"]
df4CompleteGraph["xg_halftime2-Axg_halftime2"] = df4CompleteGraph["xg_halftime2-Axg_halftime2"].clip(lower=0)
df4CompleteGraph["xg_halftime2-Axg_halftime2"] = df4CompleteGraph["xg_halftime2-Axg_halftime2"].round(2)
df4CompleteGraph["Axg_halftime2-xg_halftime2"] = df4CompleteGraph["xg_halftime-Axg_halftime"] - df4CompleteGraph["xG-A_xG"]
df4CompleteGraph["Axg_halftime2-xg_halftime2"] = df4CompleteGraph["Axg_halftime2-xg_halftime2"].clip(lower=0)
df4CompleteGraph["Axg_halftime2-xg_halftime2"] = df4CompleteGraph["Axg_halftime2-xg_halftime2"].round(2)
df4CompleteGraph["xG-A_xG"] = df4CompleteGraph["xG-A_xG"].clip(lower=0)
df4CompleteGraph["xG-A_xG"] = df4CompleteGraph["xG-A_xG"].round(2)
df4CompleteGraph["A_xG-xG"] = df4CompleteGraph["A_xG-xG"].clip(lower=0)
df4CompleteGraph["A_xG-xG"] = df4CompleteGraph["A_xG-xG"].round(2)
df4CompleteGraph["xg_halftime-Axg_halftime"] = df4CompleteGraph["xg_halftime-Axg_halftime"].clip(lower=0)
df4CompleteGraph["xg_halftime-Axg_halftime"] = df4CompleteGraph["xg_halftime-Axg_halftime"].round(2)
df4CompleteGraph["Axg_halftime-xg_halftime"] = df4CompleteGraph["Axg_halftime-xg_halftime"].clip(lower=0)
df4CompleteGraph["Axg_halftime-xg_halftime"] = df4CompleteGraph["Axg_halftime-xg_halftime"].round(2)


print(df4CompleteGraph[["xG","A_xG", "xg_halftime", "Axg_halftime","halftime","Opponent",'Halftime result',"timestamp"]])
ht1 = df4CompleteGraph[df4CompleteGraph["halftime"] == "1"]
print(ht1[["IsHome","xG","A_xG", "xg_halftime", "Axg_halftime","halftime","Opponent",'Halftime result',"timestamp"]])
ht2 = df4CompleteGraph[df4CompleteGraph["halftime"] == "2"]
print(ht2[["IsHome","xG","A_xG", "xg_halftime", "Axg_halftime","halftime","Opponent",'Halftime result',"timestamp"]])

figHistogramxG_A_xG_1Ht = px.scatter(
    df4CompleteGraph[df4CompleteGraph["halftime"] == "1"],  # .query(f'Date.between{end_date}'),
    x='BP-H',
    y='GoalDiff',
    marginal_x="histogram",
    color="timestamp",
    hover_data=['H_Red Cards', 'A_Red Cards'],
    size="xg_halftime-Axg_halftime",
    symbol = 'IsHome',
    symbol_sequence= ['diamond-cross', 'diamond'],
    text="Opponent",
    width=widthfig,
    # height=heightfig,
    # color_continuous_scale= 'Viridis',
    # facet_row="time", # makes seperate plot for value
    # marginal_x="histogram",
).update_traces(textposition='top center', selector={'type': 'scatter'} ).update_traces(
    marker=dict(color='green'), selector={'type': 'histogram'}
)
figHistogramxG_A_xG_1Ht.update_xaxes(range=[5, 95])
figHistogramxG_A_xG_1Ht.update_layout(
    title_text='Ht1: Expectedgoals - Expectedgoals Opponent', title_x=0.5,
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
    hover_data=['H_Red Cards', 'A_Red Cards'],
    size="Axg_halftime-xg_halftime",
    symbol = 'IsHome',
    symbol_sequence= ['circle-x', 'circle'],
    text="Opponent",
    width=widthfig,
    # height=heightfig,
    # facet_row="time", # makes seperate plot for value
    # marginal_x="histogram",
).update_traces(textposition='top center', selector={'type': 'scatter'} ).update_traces(
    marker=dict(color='red'), selector={'type': 'histogram'}
)
figHistogramA_xG_xG_1Ht.update_xaxes(range=[5, 95])
figHistogramA_xG_xG_1Ht.update_layout(
    title_text='Ht1: Expectedgoals Opponent - Expectedgoals', title_x=0.5,
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


figHistogramxG_A_xG_2Ht = px.scatter(
    df4CompleteGraph[df4CompleteGraph["halftime"] == "2"],  # .query(f'Date.between{end_date}'),
    x='BP-H',
    y='GoalDiff',
    marginal_x="histogram",
    color="timestamp",
    hover_data=['H_Red Cards', 'A_Red Cards'],
    size="xg_halftime2-Axg_halftime2",
    symbol = 'IsHome',
    symbol_sequence= ['diamond-cross', 'diamond'],
    text="Opponent",
    width=widthfig,
    # height=heightfig,
    # color_continuous_scale= 'Viridis',
    # facet_row="time", # makes seperate plot for value
    # marginal_x="histogram",
).update_traces(textposition='top center', selector={'type': 'scatter'} ).update_traces(
    marker=dict(color='green'), selector={'type': 'histogram'}
)
figHistogramxG_A_xG_2Ht.update_xaxes(range=[5, 95])
figHistogramxG_A_xG_2Ht.update_layout(
    title_text='Ht2: Expectedgoals - Expectedgoals Opponent', title_x=0.5,
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
    hover_data=['H_Red Cards', 'A_Red Cards'],
    size="Axg_halftime2-xg_halftime2",
    symbol = 'IsHome',
    symbol_sequence= ['circle-x', 'circle'],
    text="Opponent",
    width=widthfig,
    # height=heightfig,
    # facet_row="time", # makes seperate plot for value
    # marginal_x="histogram",
).update_traces(textposition='top center', selector={'type': 'scatter'} ).update_traces(
    marker=dict(color='red'), selector={'type': 'histogram'}
)
figHistogramA_xG_xG_2Ht.update_xaxes(range=[5, 95])
figHistogramA_xG_xG_2Ht.update_layout(
    title_text='Ht2: Expectedgoals Opponent - Expectedgoals', title_x=0.5,
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



# Streamlit encourages well-structured code, like starting execution in a main() function.
st.title("Football statistics - {}".format(team))
st.markdown('The following two diagrams display the new metric Expected Goals (**xGoals**), which is a qualitative measurement on base of the shots on goal.  \nThe expected goal model shows how high the chance of the goal really was and calculates a value for each completion based on several factors.   \nF.I. a penalty has generally a probably of 75 % to result in a goal, which would increase the xGoal value for 0.75 regardless of the penalty-outcame in this case.', unsafe_allow_html=False)

col1, col2 = st.columns(2)

col1.plotly_chart(figHistogramxG_A_xG_1Ht)

col2.plotly_chart(figHistogramA_xG_xG_1Ht)

col1.plotly_chart(figHistogramxG_A_xG_2Ht)

col2.plotly_chart(figHistogramA_xG_xG_2Ht)

col1.plotly_chart(fig_xg_perminute_home)

col2.plotly_chart(fig_xg_homexg_complete_game_all_bpse)

col1.plotly_chart(fig_xg_perminute_home_bigger_55)

col2.plotly_chart(fig_xg_perminute_home_smaller_45)

col1.plotly_chart(figScatter)

col2.plotly_chart(figScatter1)

col1.plotly_chart(figScatter5)

col2.plotly_chart(figScatter6)

col1.plotly_chart(figScatter7)

col2.plotly_chart(figScatter8)

col1.plotly_chart(BarBallpossesionstylesResultsHalftime1)

col2.plotly_chart(BarBallpossesionstylesResultsHalftime2)

# C_WPercText, N_WPercText, BP_WPercText = calc_stats(df4Complete)
# col2.write("% W < 0.45:   {}   \n % W 0.45 - 0.55:  {}   \n % W > 0.55:  {}".format(
#     C_WPercText, N_WPercText, BP_WPercText))

# show df
st.dataframe(df4Complete_show.style.format({'xG': '{:.1f}', 'A_xG': '{:.1f}', 'SoG-H': '{:.0f}',
                                            'G-H': '{:.0f}', 'G-A': '{:.0f}', 'BP-H': '{:.0f}',
                                            'BP-A': '{:.0f}', 'GA-H': '{:.0f}', 'GA-A': '{:.0f}',
                                            'xPTS': '{:.1f}', 'A_xPTS': '{:.1f}', 'SoG-A': '{:.0f}',
                                            }))
