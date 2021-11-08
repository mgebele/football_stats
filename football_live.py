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
import streamlit as st
st.set_page_config(layout="wide")

pd.options.display.float_format = "{:,.1f}".format

warnings.filterwarnings('ignore')

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
heightfig = 500

# get all the gamestatistics from in dropdown specified league and season
# setup the database connection.  There's no need to setup cursors with pandas psql.
tables = list(glob.glob("htdatan/*"))

# take only the 0 part of the every list entry
global saissons
saissons = []
for x in range(0, len(tables)):
    saissons.append(tables[x].split("\\")[1])

print(saissons)
global saison
saison = st.sidebar.selectbox("Saison", list(saissons), 8)

print(saison)
df_complete_saison = pd.read_csv(
    "htdatan/"+saison, index_col=0, encoding='utf-8')

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


def convert_hts_to_complete_games(df):

    # fill nane values of these not numeric columns
    df[['FK-H', 'FK-A']].fillna(0)
    # convert not numeric columns to numeric columns
    df['FK-H'] = df['FK-H'].astype('float64')
    df['FK-A'] = df['FK-A'].astype('float64')
    df['C-H'] = df['C-H'].astype('float64')
    df['F-H'] = df['F-H'].astype('float64')
    df['GA-H'] = df['GA-H'].astype('float64')
    df['SoffG-H'] = df['SoffG-H'].astype('float64')
    df['SoG-H'] = df['SoG-H'].astype('float64')
    df['C-A'] = df['C-A'].astype('float64')
    df['F-A'] = df['F-A'].astype('float64')
    df['GA-A'] = df['GA-A'].astype('float64')
    df['SoffG-A'] = df['SoffG-A'].astype('float64')
    df['SoG-A'] = df['SoG-A'].astype('float64')
    df['G-H'] = df['G-H'].astype('float64')
    df['G-A'] = df['G-A'].astype('float64')
    df['BP-H'] = df['BP-H'].astype('float64')
    df['BP-A'] = df['BP-A'].astype('float64')
    # xGoals columns
    if not set(['xG', 'xPTS', 'GOALS', 'A_xG', 'G-A', 'A_xPTS']).issubset(df.columns):
        df['xG'] = -1.0
        df['GOALS'] = -1.0
        df['xPTS'] = -1.0
        df['A_xG'] = -1.0
        df['G-A'] = -1.0
        df['A_xPTS'] = -1.0
    else:
        df['xG'] = df['xG'].astype('float64')
        df['GOALS'] = df['GOALS'].astype('float64')
        df['xPTS'] = df['xPTS'].astype('float64')
        df['A_xG'] = df['A_xG'].astype('float64')
        df['G-A'] = df['G-A'].astype('float64')
        df['A_xPTS'] = df['A_xPTS'].astype('float64')

    # calculate halftime table to fulltime table
    df = df.groupby(['Home', 'Opponent', 'Date', 'Round']).agg({'BP-H': 'mean', 'C-H': 'sum',
                                                                'F-H': 'sum', 'FK-H': 'sum', 'GA-H': 'sum',
                                                                'GoKeSa-H': 'sum', 'G-H': 'sum', 'Off-H': 'sum',
                                                                'SoffG-H': 'sum', 'SoG-H': 'sum',
                                                                           'BP-A': 'mean',
                                                                           'C-A': 'sum',
                                                                           'F-A': 'sum', 'FK-A': 'sum', 'GA-A': 'sum',
                                                                           'GoKeSa-A': 'sum', 'G-A': 'sum', 'Off-A': 'sum',
                                                                           'SoffG-A': 'sum', 'SoG-A': 'sum',
                                                                           # xGoals stats are only from whole game - not halftime - so mean does not change anything
                                                                           'xG': 'mean',
                                                                           'GOALS': 'mean',
                                                                           'xPTS': 'mean',
                                                                           'A_xG': 'mean',
                                                                           'A_GOALS': 'mean',
                                                                           'A_xPTS': 'mean'
                                                                }).reset_index()

    newcols = []

    for x in df.columns:
        if x.startswith('SUM') or x.startswith('MIN') or x.startswith('AVG'):
            x = re.sub('SUM', '', x)
            x = re.sub('MIN', '', x)
            x = re.sub('AVG', '', x)
            x = x.replace("`", "")
            x = x.replace(")", "")
            x = x.replace("(", "")
        newcols.append(x)

    df.columns = newcols

    return df


def df_cleaning_converting(df):
    df = df[['H_Teamnames', 'A_Teamnames', 'H_Goals', 'A_Goals', 'H_Ball Possession', 'A_Ball Possession', 'A_Goal Attempts', 'H_Goal Attempts',
             'H_Shots on Goal', 'A_Shots on Goal', 'H_Shots off Goal', 'A_Shots off Goal', 'H_Free Kicks',
             'A_Free Kicks', 'H_Corner Kicks', 'A_Corner Kicks', 'H_Offsides', 'A_Offsides', 'H_Goalkeeper Saves', 'A_Goalkeeper Saves',
             'H_Fouls', 'A_Fouls', 'A_gameinfo', 'A_datetime', 'xG', 'GOALS', 'xPTS', 'A_xG', 'A_GOALS', 'A_xPTS']]

    df["R"] = 'X'

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

    df.columns = ['Home', 'Opponent', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A',
                  'SoG-H', 'SoG-A', 'SoffG-H', 'SoffG-A', 'FK-H',
                  'FK-A', 'C-H', 'C-A', 'Off-H', 'Off-A', 'GoKeSa-H', 'GoKeSa-A',
                  'F-H', 'F-A', 'Round', 'Date', 'xG', 'GOALS', 'xPTS', 'A_xG', 'A_GOALS', 'A_xPTS', 'R']

    df = df[['Home', 'Opponent', 'R', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A',
             'SoG-H', 'SoG-A', 'SoffG-H', 'SoffG-A', 'FK-H',
             'FK-A', 'C-H', 'C-A', 'Off-H', 'Off-A', 'GoKeSa-H', 'GoKeSa-A', 'F-H',
             'F-A', 'Round', 'Date', 'xG', 'GOALS', 'xPTS', 'A_xG', 'A_GOALS', 'A_xPTS']]

    df["IsHome"] = 0

    df = df[['Home', 'Opponent', 'R', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A',
             'SoG-H', 'SoG-A', 'SoffG-H', 'SoffG-A', 'FK-H',
             'FK-A', 'C-H', 'C-A', 'Off-H', 'Off-A', 'GoKeSa-H', 'GoKeSa-A', 'F-H',
             'F-A', 'Round', 'Date', 'IsHome', 'xG', 'GOALS', 'xPTS', 'A_xG', 'A_GOALS', 'A_xPTS']]
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

    # Change the columns for the Opponentmatches of the specific team
    OpponentTeamReversedColumns = ['Opponent', 'Home',  '1x2', 'R',  'G-A', 'G-H',
                                   'BP-A', 'BP-H', 'GA-A', 'GA-H',  'SoG-A', 'SoG-H', 'SoffG-A', 'SoffG-H',  'FK-A', 'FK-H',
                                   'C-A', 'C-H',  'Off-A', 'Off-H', 'GoKeSa-A', 'GoKeSa-H', 'F-A', 'F-H',
                                   'Round', 'Date', 'IsHome',
                                   'A_xG', "A_xPTS", "A_GOALS", 'xG', "xPTS", "GOALS", ]  # , 'IsHome'

    df4OpponentReversed = df4Opponent.reindex(
        columns=OpponentTeamReversedColumns)

    df4OpponentReversed.columns = ['Home', 'Opponent', '1x2', 'R', 'G-H',
                                   'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A',
                                   'SoG-H', 'SoG-A', 'SoffG-H', 'SoffG-A', 'FK-H',
                                   'FK-A', 'C-H', 'C-A', 'Off-H', 'Off-A', 'GoKeSa-H', 'GoKeSa-A', 'F-H',
                                   'F-A', 'Round', 'Date', 'IsHome',
                                   'xG', "xPTS", "GOALS", 'A_xG', "A_xPTS", "A_GOALS"]
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

    # execute the query and assign it to a pandas dataframe
    dfxg = df_complete_saison[(df_complete_saison.TEAMS == team) | (
        df_complete_saison.A_TEAMS == team)]

    dfxg = process_team_names_of_df(dfxg)

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

dfxg_df_merged_cleaned = df_cleaning_converting(df)

df4Home, df4OpponentReversed = df_specific_team(
    dfxg_df_merged_cleaned, team)

df4Complete = create_df4Complete(df4Home, df4OpponentReversed)

slidertext = 'Show last x halftimes'
nrGames = st.sidebar.slider(slidertext, max_value=len(
    df4Complete), value=len(df4Complete))

# change rows of df depending on userinput
df4Complete = df4Complete[:nrGames]
df4Complete = df4Complete.sort_values("Date", ascending=False)
df4Complete = df4Complete.round(1)

df4Complete[['xG', 'A_xG', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A', 'SoG-H', 'SoG-A',  'xPTS', 'A_xPTS']] = df4Complete[['xG',
                                                                                                                              'A_xG', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A', 'SoG-H', 'SoG-A',  'xPTS', 'A_xPTS']].apply(pd.to_numeric, errors='coerce', axis=1)

df4Complete_show = df4Complete[['Home', 'Opponent', 'IsHome', 'R', 'xG', 'A_xG', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A',
                                'SoG-H', 'SoG-A',  'xPTS', 'A_xPTS', 'Date']]


# create df for visualizing
df4CompleteGraph = df4Complete.copy()

df4CompleteGraph["SoG-H-SoG-A"] = df4CompleteGraph["SoG-H"] - \
    df4CompleteGraph["SoG-A"]
df4CompleteGraph["SoG-H-SoG-A"] = df4CompleteGraph["SoG-H-SoG-A"].clip(
    lower=0)

df4CompleteGraph["SoG-A-SoG-H"] = df4CompleteGraph["SoG-A"] - \
    df4CompleteGraph["SoG-H"]
df4CompleteGraph["SoG-A-SoG-H"] = df4CompleteGraph["SoG-A-SoG-H"].clip(
    lower=0)


figScatter = px.scatter(
    df4CompleteGraph,  # .query(f'Date.between{end_date}'),
    x='BP-H',
    y='GoalDiff',
    color="timestamp",
    size="SoG-H-SoG-A",
    text="Opponent",
    width=widthfig,
    # height=heightfig,
    title="SoGH-SoGA - Halftimes",
    # color_continuous_scale= 'Viridis',
    # color_discrete_map={"W": "green", "D": "gray", "L": "red"}

    # facet_row="time", # makes seperate plot for value
    # marginal_x="histogram",
).update_traces(textposition='top center')


figScatter1 = px.scatter(
    df4CompleteGraph,  # .query(f'Date.between{end_date}'),
    x='BP-H',
    y='GoalDiff',
    color="timestamp",
    size="SoG-A-SoG-H",
    text="Opponent",
    width=widthfig,
    # height=heightfig,
    title="SoGA-SoGH - Halftimes",
    # color_discrete_map={"W": "green", "D": "gray", "L": "red"}

    # facet_row="time", # makes seperate plot for value
    # marginal_x="histogram",
).update_traces(textposition='top center')


# Create data for histogram 2
figHist2 = px.bar(
    df4CompleteGraph,
    x='BPTypes',
    # y='1x2',
    text=df4CompleteGraph.index,
    title="BP-Styles - Halftimes",
    color='1x2',
    color_discrete_map={"W": "green", "D": "gray", "L": "red"},
    width=widthfig,
    # height=heightfig,


    # opacity=0.5,
    # text="Opponent",
).update_xaxes(categoryorder="array",  categoryarray=['<45', '45-55', '>55'])


# create scatterplot with XG - bubble size
df4CompleteGraph = df4Complete.copy()

df4CompleteGraph = convert_hts_to_complete_games(df4CompleteGraph)

# Calculate again the stuff like for the single halftimes before!
# GoalDifference
df4CompleteGraph["GoalDiff"] = df4CompleteGraph["G-H"] - \
    df4CompleteGraph["G-A"]
df4CompleteGraph = df4CompleteGraph.sort_values("Date",  ascending=False)
# calculate column with 3 Ballposition types
df4CompleteGraph["BPTypes"] = '0'
df4CompleteGraph["BPTypes"] = df4CompleteGraph.apply(
    lambda row: calculate_1x2_BPTypes(row), axis=1, result_type='reduce')
df4CompleteGraph['Date'] = pd.to_datetime(
    df4CompleteGraph['Date'], format="%d.%m.%Y %H:%M")
# convert datetime to timestamp for scatter visualization
df4CompleteGraph['timestamp'] = df4CompleteGraph.Date.astype('int64')//10**9
df4CompleteGraph = df4CompleteGraph.sort_values("Date", ascending=False)
# Create data for scatter graph
df4CompleteGraph["xG-A_xG"] = df4CompleteGraph["xG"] - df4CompleteGraph["A_xG"]
df4CompleteGraph["xG-A_xG"] = df4CompleteGraph["xG-A_xG"].clip(lower=0)
df4CompleteGraph["xG-A_xG"] = df4CompleteGraph["xG-A_xG"].round(1)
df4CompleteGraph["A_xG-xG"] = df4CompleteGraph["A_xG"] - df4CompleteGraph["xG"]
df4CompleteGraph["A_xG-xG"] = df4CompleteGraph["A_xG-xG"].clip(lower=0)
df4CompleteGraph["A_xG-xG"] = df4CompleteGraph["A_xG-xG"].round(1)


figScatter3 = px.scatter(
    df4CompleteGraph,  # .query(f'Date.between{end_date}'),
    x='BP-H',
    y='GoalDiff',
    color="timestamp",
    size="xG-A_xG",
    text="Opponent",
    width=widthfig,
    # height=heightfig,
    title="xG-A_xG - Complete games",
    # color_continuous_scale= 'Viridis',
    # facet_row="time", # makes seperate plot for value
    # marginal_x="histogram",
).update_traces(textposition='top center', marker_symbol="cross")

figScatter4 = px.scatter(
    df4CompleteGraph,  # .query(f'Date.between{end_date}'),
    x='BP-H',
    y='GoalDiff',
    color="timestamp",
    size="A_xG-xG",
    text="Opponent",
    width=widthfig,
    # height=heightfig,
    title="A_xG-xG - Complete games",
    # facet_row="time", # makes seperate plot for value
    # marginal_x="histogram",
).update_traces(textposition='top center')


# Streamlit encourages well-structured code, like starting execution in a main() function.
st.title("Football statistics - {}".format(team))

col1, col2 = st.columns(2)

col1.plotly_chart(figScatter3)

col2.plotly_chart(figScatter4)

col1.plotly_chart(figScatter)

col2.plotly_chart(figScatter1)

st.plotly_chart(figHist2)

# C_WPercText, N_WPercText, BP_WPercText = calc_stats(df4Complete)
# col2.write("% W < 0.45:   {}   \n % W 0.45 - 0.55:  {}   \n % W > 0.55:  {}".format(
#     C_WPercText, N_WPercText, BP_WPercText))

# show df
st.dataframe(df4Complete_show.style.format({'xG': '{:.1f}', 'A_xG': '{:.1f}', 'SoG-H': '{:.0f}',
                                            'G-H': '{:.0f}', 'G-A': '{:.0f}', 'BP-H': '{:.0f}',
                                            'BP-A': '{:.0f}', 'GA-H': '{:.0f}', 'GA-A': '{:.0f}',
                                            'xPTS': '{:.1f}', 'A_xPTS': '{:.1f}', 'SoG-A': '{:.0f}',
                                            }))
