import plotly.graph_objects as go
import plotly_express as px
import pandas as pd
import numpy as np
import streamlit as st
import re

from scripts.data_processing.pre_processing_loading import process_team_names_of_df, df_cleaning_converting, load_xg_gamestats_sql
from scripts.data_processing.processing_augmentation import df_specific_team, create_df4Complete

global widthfig
widthfig = 700
heightfig = 600

custom_green_scale = [
        [0.0, 'rgb(0, 50, 0)'],         # Sehr dunkelgrün (nahezu schwarz)
        [0.2, 'rgb(0, 100, 0)'],        # Dunkelgrün
        [0.4, 'rgb(34, 139, 34)'],      # Forest Green
        [0.6, 'rgb(50, 205, 50)'],      # Lime Green
        [0.8, 'rgb(144, 238, 144)'],    # Light Green
        [1.0, 'rgb(193, 255, 193)']     # Hellgrün (PaleGreen statt fast weiß)
    ]

custom_red_scale = [
    [0.0, 'rgb(50, 0, 0)'],         # Sehr dunkelrot (nahezu schwarz)
    [0.2, 'rgb(139, 0, 0)'],        # Dunkelrot (Firebrick)
    [0.4, 'rgb(178, 34, 34)'],       # Crimson
    [0.6, 'rgb(220, 20, 60)'],       # Orangerot (Crimson)
    [0.8, 'rgb(255, 99, 71)'],       # Tomate (Tomato)
    [1.0, 'rgb(255, 160, 160)']     # Hellrot (LightCoral statt fast weiß)
]
title_x=.2 # alignment of title of plotly diagrams. 0 = left, 1 = right


def team_x_data_processing(df_complete_saison, teamnamedict, saison, team):
    df = df_complete_saison[(df_complete_saison.H_TEAMNAMES == team) | (
        df_complete_saison.A_TEAMNAMES == team)]
    
    df = process_team_names_of_df(df, teamnamedict)
    dfxg = load_xg_gamestats_sql(saison, team, teamnamedict)

    # rename columns for
    dfxg_rename = dfxg.rename(
        columns={'TEAMS': 'H_TEAMNAMES', 'A_TEAMS': 'A_TEAMNAMES'})

    df = pd.merge(
        df, dfxg_rename, on=["H_TEAMNAMES", "A_TEAMNAMES"])
    df = df.drop_duplicates()

    df["homexg_complete_game"] = ""
    df["awayxg_complete_game"] = ""
    df["last_game_minute"] = -1
    df["start_min_game"] = -1

    for game_loc in df.index:

        homexg_complete_game = []
        awayxg_complete_game = []

        last_game_minute = df["TIMING_CHART_XG"].loc[game_loc].rsplit("'")[-2].rsplit(";")[1]
        start_min_game = int( re.sub("[^0-9]", "", df["TIMING_CHART_XG"].loc[game_loc][:2]) )

        # nehmen hier minute für minute und schauen nach dem xg wert für diese minute
        for x in range(start_min_game,int(last_game_minute)+1):
            try:
                homexgperminute = df["TIMING_CHART_XG"].loc[game_loc].split("{}' Total xG: ".format(x))[1].split(";")[0][:4]  # [:4] - only last 4 digits so no goalscorer infos
                awayxgperminute = df["TIMING_CHART_XG"].loc[game_loc].split("{}' Total xG: ".format(x), 2)[2].split(";")[0][:4]
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
    # rename xg table columns
    df.rename(columns={'A_GOALS_x': 'A_GOALS'}, inplace=True)
    df.rename(columns={'A_GOALS_y': 'A_GOALS_COMPLETE_GAME'}, inplace=True)
    df.rename(columns={'homexg_complete_game': 'HOMEXG_COMPLETE_GAME'}, inplace=True)
    df.rename(columns={'awayxg_complete_game': 'AWAYXG_COMPLETE_GAME'}, inplace=True)

    dfxg_df_merged_cleaned = df_cleaning_converting(df)

    df4Home, df4OpponentReversed = df_specific_team(dfxg_df_merged_cleaned, team)

    df4Complete = create_df4Complete(df4Home, df4OpponentReversed)

    slidertext = 'Show last x halftimes'
    nrGames = len(df4Complete) #st.sidebar.slider(slidertext, max_value=len( df4Complete), value=len(df4Complete), step=2)

    # change rows of df depending on userinput
    df4Complete = df4Complete[:nrGames]
    df4Complete = df4Complete.sort_values("Date", ascending=False)
    df4Complete = df4Complete.round(1)

    df4Complete[['xG', 'A_xG', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A', 'SoG-H', 'SoG-A',  'xPTS', 'A_xPTS',  "A_Red Cards", "H_Red Cards", "halftime"]] = df4Complete[[
                'xG', 'A_xG', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A', 'SoG-H', 'SoG-A',  'xPTS', 'A_xPTS',  "A_Red Cards", "H_Red Cards", "halftime"]].apply(pd.to_numeric, errors='coerce', axis=1)

    df4Complete['A_Red Cards'] = df4Complete['A_Red Cards'].fillna(0)
    df4Complete['H_Red Cards'] = df4Complete['H_Red Cards'].fillna(0)

    return df4Complete

# Function to create bar charts for a given halftime
def create_bar_chart(df4CompleteGraph, halftime, title, highest_count_yaxis, color_map, category_orders, category_order):
    filtered_df = df4CompleteGraph[df4CompleteGraph["halftime"] == halftime]
    fig = px.bar(
        filtered_df,
        x='BPTypes',
        y='AdjustedCount',
        color='Halftime result',
        color_discrete_map=color_map,
        width=widthfig,
        text="Opponent",
        category_orders=category_orders,
        barmode='group'  # Ensures bars are grouped side by side
    )
    fig.update_xaxes(categoryorder="array", categoryarray=category_order)
    fig.update_yaxes(range=[0, highest_count_yaxis], title='Adjusted Count')
    fig.update_layout(
        title_text=title,
        title_x=title_x,  # Center alignment
        xaxis=dict(tickmode='array', showticklabels=True),
        bargap=0.2,        # Adjust this value (0 to 1) for spacing between bars
        bargroupgap=0.1    # Adjust this value for spacing within grouped bars
    )
    return fig

def team_x_visualization(df4Complete):
    # calc the xg per minute over all games to get the mean over all minutes from all games!
    values = (0.0, 100.0) #st.sidebar.slider('B-PRange for xG per minute', 0.0, 100.0, (0.0, 100.0))

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


    # create df for visualizing
    df4CompleteGraph = df4Complete.copy()

    # teamname_to_search =  st.sidebar.text_input("Search for Opponent", )
    # df4CompleteGraph = df4CompleteGraph[df4CompleteGraph["Opponent"].str.contains("{}".format(teamname_to_search), na=False, case=False)]

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
        title_text='xG per minute: {} < bp < {}'.format(int(bigger_bp), int(smaller_bp)), title_x=title_x,
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
        title_text='xG per minute', title_x=title_x,
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
        title_text='xG per minute: bp > 55', title_x=title_x,
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
        title_text='xG per minute: bp < 45', title_x=title_x,
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
        color_continuous_scale= custom_green_scale,
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
        color_continuous_scale= custom_red_scale,
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


    # second half is second entry always for all rows where halftime column has nan values
    # from 2025 on we have the halftime values correctly filled with 1 or 2.
    # Remove games that do not have exactly two rows

    df4CompleteGraph = df4CompleteGraph[df4CompleteGraph.groupby('Date')['Date'].transform('size') >= 2]
    df4CompleteGraph = df4CompleteGraph.sort_index()
    def fill_halftime(group):
        halftime_values = group['halftime'].copy()
        # This assumes there are exactly two rows per group
        halftime_values[pd.isna(halftime_values)] = [1, 2]
        return halftime_values

    df4CompleteGraph['halftime'] = df4CompleteGraph.groupby('Date').apply(fill_halftime).reset_index(level=0, drop=True)
    df4CompleteGraph['halftime'] = df4CompleteGraph['halftime'].astype(int)
    df4CompleteGraph = df4CompleteGraph.sort_index()


    naming1x2 = {"W": "Win", "D": "Draw", "L": "Loss"}
    df4CompleteGraph['Halftime result'] = df4CompleteGraph['1x2'].replace(
        naming1x2)

    try:
        highest_count_yaxis = df4CompleteGraph.groupby(["BPTypes", "halftime"]).agg(
            'count').sort_values("Opponent", ascending=False).iloc[0].Home
    except:
        highest_count_yaxis = 0

    df4CompleteGraph['count'] = 1
    df4CompleteGraph_ht1 = df4CompleteGraph[df4CompleteGraph["halftime"] == 1]
    df4CompleteGraph_ht2 = df4CompleteGraph[df4CompleteGraph["halftime"] == 2]
    required_categories = ['<45', '45-55', '>55']
    for category in required_categories:
        if category not in df4CompleteGraph_ht1['BPTypes'].unique():
            print("category", category)
            # Create a DataFrame for the new row
            new_row_ht1 = pd.DataFrame([{
                'BPTypes': category, 
                'Halftime result': 'Draw',  
                'Home': '',  
                'Opponent': '',  
                'IsHome': 0, 
                'R': '',  
                'xG': 0.0,  
                'halftime': 1,
                'A_xG': 0.0,
                'count': 0
            }])
            # Use pd.concat to add the new row
            df4CompleteGraph = pd.concat([df4CompleteGraph, new_row_ht1], ignore_index=True)
        
        if category not in df4CompleteGraph_ht2['BPTypes'].unique():
            print("category", category)
            # Create a DataFrame for the new row
            new_row_ht2 = pd.DataFrame([{
                'BPTypes': category, 
                'Halftime result': 'Draw',  
                'Home': '',  
                'Opponent': '',  
                'IsHome': 0, 
                'R': '',  
                'xG': 0.0,  
                'halftime': 2,
                'A_xG': 0.0,
                'count': 0
            }])
            # Use pd.concat to add the new row
            df4CompleteGraph = pd.concat([df4CompleteGraph, new_row_ht2], ignore_index=True)


    # add goal differences for halftime charts, with draws consistently represented by a single unit in height
    df4CompleteGraph['GoalDifference_abs'] = abs(df4CompleteGraph['G-H'] - df4CompleteGraph['G-A'])

    # Create the 'AdjustedCount' column based on the halftime result
    df4CompleteGraph['AdjustedCount'] = df4CompleteGraph.apply(
        lambda row: 1 if row['Halftime result'] == 'Draw' else row['GoalDifference_abs'],
        axis=1
    )
    
    # Define color mapping and category order
    color_map = {"Win": "green", "Draw": "gray", "Loss": "red"}
    category_order = ['<45', '45-55', '>55']
    category_orders = {'BPTypes': category_order}

    # Create and display the bar charts
    BarBPstylesResultsHalftime1 = create_bar_chart(df4CompleteGraph=df4CompleteGraph, halftime=1, title='BP Styles - Ht1 Results',
        highest_count_yaxis=highest_count_yaxis, color_map=color_map, category_orders=category_orders, category_order=category_order)
    
    BarBPstylesResultsHalftime2 = create_bar_chart(df4CompleteGraph=df4CompleteGraph, halftime=2, title='BP Styles - Ht2 Results',
        highest_count_yaxis=highest_count_yaxis, color_map=color_map, category_orders=category_orders, category_order=category_order)

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
    ht1 = df4CompleteGraph[df4CompleteGraph["halftime"] == 1]
    print(ht1[["IsHome","xG","A_xG", "xg_halftime", "Axg_halftime","halftime","Opponent",'Halftime result',"timestamp"]])
    ht2 = df4CompleteGraph[df4CompleteGraph["halftime"] == 2]
    print(ht2[["IsHome","xG","A_xG", "xg_halftime", "Axg_halftime","halftime","Opponent",'Halftime result',"timestamp"]])

    # Determine if there was a red card in a game
    df4CompleteGraph['red_card'] = df4CompleteGraph[['H_Red Cards', 'A_Red Cards']].apply(lambda x: x[0] > 0 or x[1] > 0, axis=1)
    # Define marker properties based on the presence of a red card
    df4CompleteGraph['marker_properties'] = df4CompleteGraph['red_card'].apply(
        lambda x: {'line_width': 2, 'line_color': 'red'} if x else {'line_width': 1, 'line_color': 'black'}
    )

    #  Define the clusters
    clusters = [
        'diff_xg_fulltime-diff_xg_halftime', 
        'diff_Axg_fulltime-diff_Axg_halftime',
        'xg_halftime-Axg_halftime', 
        'Axg_halftime-xg_halftime'
    ]
    grouped_sum = df4CompleteGraph.groupby(["BPTypes", "halftime"])[clusters].sum()
    try:
        # Find the maximum value across all clusters and groups
        highest_xg_count_yaxis = grouped_sum.values.max()
    except Exception as e:
        print(f"An error occurred while calculating the y-axis range: {e}")
        highest_xg_count_yaxis = 0

    # Create barchart for xg per bptypes 1
    BarBPstylesXGHalftime1 = px.bar(
        df4CompleteGraph[df4CompleteGraph["halftime"] == 1],
        x='BPTypes',
        y=['xg_halftime-Axg_halftime', 'Axg_halftime-xg_halftime'],
        barmode='group',
        # text=df4CompleteGraph.index,
        # title="BP-Styles - Halftimes",
        # color='Halftime result',
        color_discrete_map={"xg_halftime-Axg_halftime": "green", "Axg_halftime-xg_halftime": "red"},
        width=widthfig,
        # opacity=0.5,
        text="Opponent",
    ).update_xaxes(categoryorder="array", categoryarray=['<45', '45-55', '>55']).update_yaxes(
        range=[0, highest_xg_count_yaxis])
    BarBPstylesXGHalftime1.update_layout(
        title_text='BP Styles - xG Ht1', title_x=title_x, xaxis=dict(
            tickmode='array', showticklabels=True,
        )
    )
    BarBPstylesXGHalftime1.update_layout(legend=dict(
        yanchor="top",
        y=1.2,
        xanchor="right",
        x=1.12
    ))

    # Create barchart for xg per bptypes 2
    BarBPstylesXGHalftime2 = px.bar(
        df4CompleteGraph[df4CompleteGraph["halftime"] == 2],
        x='BPTypes',
        y=['diff_xg_fulltime-diff_xg_halftime', 'diff_Axg_fulltime-diff_Axg_halftime'],
        barmode='group',
        # text=df4CompleteGraph.index,
        # title="BP-Styles - Halftimes",
        # color='Halftime result',
        color_discrete_map={"diff_xg_fulltime-diff_xg_halftime": "green", "diff_Axg_fulltime-diff_Axg_halftime": "red"},
        width=widthfig,
        # opacity=0.5,
        text="Opponent",
    ).update_xaxes(categoryorder="array", categoryarray=['<45', '45-55', '>55']).update_yaxes(
        range=[0, highest_xg_count_yaxis])
    BarBPstylesXGHalftime2.update_layout(
        title_text='BP Styles - xG Ht2', title_x=title_x, xaxis=dict(
            tickmode='array', showticklabels=True,
        )
    )
    BarBPstylesXGHalftime2.update_layout(legend=dict(
        yanchor="top",
        y=1.2,
        xanchor="right",
        x=1.12
    ))
    
    return BarBPstylesResultsHalftime1, BarBPstylesResultsHalftime2, BarBPstylesXGHalftime1, BarBPstylesXGHalftime2, \
            figScatter_SoG_SoGA, figScatter_SoGA_soG, fig_xg_perminute_home, fig_xg_homexg_complete_game_all_bpse, \
            fig_xg_perminute_home_bigger_55, fig_xg_perminute_home_smaller_45, df4CompleteGraph


def plot_scatters(df4CompleteGraph, max_loss_goals_diff, max_win_goals_diff):
    plot_data = df4CompleteGraph[df4CompleteGraph["halftime"] == 1].copy()
    # Create figure
    figHistogramHt1xG_Combined = go.Figure()

    # Add traces with conditional symbol based on size
    for color_scale, color in [(custom_green_scale, 'green'), (custom_red_scale, 'red')]:
        size_col = 'xg_halftime-Axg_halftime' if color == 'green' else 'Axg_halftime-xg_halftime'
        
        # Filter data where size > 0
        data_subset = plot_data[plot_data[size_col] > 0]
        
        trace_name = 'xG - xGO' if color == 'green' else 'xGO - xG'

        for is_home in [0, 1]:
            home_subset = data_subset[data_subset['IsHome'] == is_home]
            
            # Logarithmic size scaling with minimum visible size
            # log(x+1) ensures small values are visible while preventing extreme large sizes
            marker_sizes = 10 + 20 * np.log(home_subset[size_col] + 1)

            # Determine symbol based on IsHome
            symbol = 'diamond' if is_home == 1 else 'circle'
            
            trace = go.Scatter(
                x=home_subset['BP-H'],
                y=home_subset['GoalDiff'],
                mode='markers+text',
                marker=dict(
                    size=marker_sizes,
                    color=home_subset['timestamp'],
                    colorscale=color_scale,
                    symbol=symbol
                ),
                name=f'{trace_name} {"(Home)" if is_home == 1 else "(Away)"}',
                text=home_subset['Opponent'],
                hovertemplate=
                '<b>Opponent</b>: %{text}<br>' +
                '<b>BP-H</b>: %{x:.2f}<br>' +
                '<b>Goal Difference</b>: %{y}<br>' +
                '<b>xG-dif</b>: %{customdata[3]:.2f}<br>' +
                '<b>H Red Cards</b>: %{customdata[0]}<br>' +
                '<b>A Red Cards</b>: %{customdata[1]}<br>' +
                '<b>Date</b>: %{customdata[2]}<br>' +
                '<b>Is Home</b>: %{customdata[4]}<extra></extra>',
                customdata=home_subset[['H_Red Cards', 'A_Red Cards', 'Date', size_col, 'IsHome']],
                textposition='top center',
                textfont=dict(size=9, color='gray')
            )
            
            figHistogramHt1xG_Combined.add_trace(trace)

    # Update layout
    figHistogramHt1xG_Combined.update_layout(
        title_text='HT1: xG-xGO and xGO-xG',
        title_x=title_x,
        yaxis=dict(
            tickmode='linear', 
            tick0=1, 
            dtick=1, 
            title="Goal difference",
            range=[-1*(max_loss_goals_diff+1), max_win_goals_diff+1]
        ),
        legend=dict(
            yanchor="top", 
            y=1.2, 
            xanchor="right", 
            x=1.12
        )
    )
    figHistogramHt1xG_Combined.update_xaxes(range=[15, 90])


    plot_data_Ht2 = df4CompleteGraph[df4CompleteGraph["halftime"] == 2].copy()
    # Create figure
    figHistogramHt2xG_Combined = go.Figure()

    # Add traces with conditional symbol based on size
    for color_scale, color in [(custom_green_scale, 'green'), (custom_red_scale, 'red')]:
        size_col = 'diff_xg_fulltime-diff_xg_halftime' if color == 'green' else 'diff_Axg_fulltime-diff_Axg_halftime'
        
        # Filter data where size > 0
        data_subset = plot_data_Ht2[plot_data_Ht2[size_col] > 0]
        trace_name = 'xG - xGO' if color == 'green' else 'xGO - xG'

        for is_home in [0, 1]:
            home_subset = data_subset[data_subset['IsHome'] == is_home]
            
            # Determine symbol based on IsHome
            symbol = 'diamond' if is_home == 1 else 'circle'

            # Logarithmic size scaling with minimum visible size
            # log(x+1) ensures small values are visible while preventing extreme large sizes
            marker_sizes = 10 + 20 * np.log(home_subset[size_col] + 1)

            trace = go.Scatter(
                x=home_subset['BP-H'],
                y=home_subset['GoalDiff'],
                mode='markers+text',
                marker=dict(
                    size=marker_sizes,
                    color=home_subset['timestamp'],
                    colorscale=color_scale,
                    symbol=symbol
                ),
                name=f'{trace_name} {"(Home)" if is_home == 1 else "(Away)"}',
                text=home_subset['Opponent'],
                hovertemplate=
                '<b>Opponent</b>: %{text}<br>' +
                '<b>BP-H</b>: %{x:.2f}<br>' +
                '<b>Goal Difference</b>: %{y}<br>' +
                '<b>xG-dif</b>: %{customdata[3]:.2f}<br>' +
                '<b>H Red Cards</b>: %{customdata[0]}<br>' +
                '<b>A Red Cards</b>: %{customdata[1]}<br>' +
                '<b>Date</b>: %{customdata[2]}<br>' +
                '<b>Is Home</b>: %{customdata[4]}<extra></extra>',
                customdata=home_subset[['H_Red Cards', 'A_Red Cards', 'Date', size_col, 'IsHome']],
                textposition='top center',
                textfont=dict(size=9, color='gray')
            )
            
            figHistogramHt2xG_Combined.add_trace(trace)

    # Update layout
    figHistogramHt2xG_Combined.update_layout(
        title_text='HT2: xG-xGO and xGO-xG',
        title_x=title_x,
        yaxis=dict(
            tickmode='linear', 
            tick0=1, 
            dtick=1, 
            title="Goal difference",
            range=[-1*(max_loss_goals_diff+1), max_win_goals_diff+1]
        ),
        legend=dict(
            yanchor="top", 
            y=1.2, 
            xanchor="right", 
            x=1.12
        )
    )
    figHistogramHt2xG_Combined.update_xaxes(range=[15, 90])

    return figHistogramHt1xG_Combined, figHistogramHt2xG_Combined


def get_max_win_loss_goals_diff(df):
    # Filter the DataFrame for 'Loss' and 'Win' results
    loss_df = df[df["Halftime result"] == "Loss"]
    win_df = df[df["Halftime result"] == "Win"]

    # Calculate the maximum AdjustedCount for 'Loss' and 'Win'
    max_loss = loss_df["AdjustedCount"].max()
    max_win = win_df["AdjustedCount"].max()

    return max_loss, max_win

def page_h2h_comparison(df_complete_saison : pd.DataFrame, teamnamedict : dict, saison : str):

    dfallteamnamesl = df_complete_saison.H_TEAMNAMES.unique()

    # take only the 0 part of the every list entry
    teamsoptions = []
    for x in range(0, len(dfallteamnamesl)):
        teamsoptions.append(dfallteamnamesl[x])

    global xg_team
    xg_team = st.sidebar.selectbox("Team", list(np.sort(teamsoptions)), 0,  key="slider_team_1")
    # convert string to df to use process_team_names_of_df function
    df_teamname = pd.DataFrame([xg_team])
    # convert xg teamnames to correct ones that are used in htdatan
    team_df = df_teamname.replace(teamnamedict)
    # teamname corrected that it fits to htdatan teamnames
    team = team_df.iloc[0][0]

    # Now second Team:
    xg_team_2 = st.sidebar.selectbox("Team2", list(np.sort(teamsoptions)), 1, key="slider_team_2")
    # convert string to df to use process_team_names_of_df function
    df_teamname_2 = pd.DataFrame([xg_team_2])
    # convert xg teamnames to correct ones that are used in htdatan
    team_2_df = df_teamname_2.replace(teamnamedict)
    # teamname corrected that it fits to htdatan teamnames
    team_2 = team_2_df.iloc[0][0]

    if st.sidebar.button("Run Comparison"):

        df4Complete = team_x_data_processing(df_complete_saison, teamnamedict, saison, team)

        BarBPstylesResultsHalftime1, BarBPstylesResultsHalftime2, BarBPstylesXGHalftime1, \
        BarBPstylesXGHalftime2, figScatter_SoG_SoGA, figScatter_SoGA_soG, fig_xg_perminute_home, \
        fig_xg_homexg_complete_game_all_bpse, fig_xg_perminute_home_bigger_55, \
        fig_xg_perminute_home_smaller_45, df4CompleteGraph = team_x_visualization(df4Complete)

        max_loss_goals_diff, max_win_goals_diff = get_max_win_loss_goals_diff(df4CompleteGraph)
        print("team max_loss_goals_diff, max_win_goals_diff")
        print(max_loss_goals_diff, max_win_goals_diff)

        # need to get the max_loss_goals_diff, max_win_goals_diff for team 2 to set an equal y-axis for both teams
        df4Complete2 = team_x_data_processing(df_complete_saison, teamnamedict, saison, team_2)
        BarBPstylesResultsHalftime1_Team2, BarBPstylesResultsHalftime2_Team2, BarBPstylesXGHalftime1_Team2, \
        BarBPstylesXGHalftime2_Team2, figScatter_SoG_SoGA_Team2, figScatter_SoGA_soG_Team2, fig_xg_perminute_home_Team2, fig_xg_homexg_complete_game_all_bpse_Team2, \
        fig_xg_perminute_home_bigger_55_Team2, fig_xg_perminute_home_smaller_45_Team2, df4Complete2Graph_Team2 = team_x_visualization(df4Complete2)
        max_loss_goals_diff_Team2, max_win_goals_diff_Team2 = get_max_win_loss_goals_diff(df4Complete2Graph_Team2)

        # getting the max win and loss values for the y-axis for both teams to allign the scatterplots
        comp_max_loss_goals_diff = max(max_loss_goals_diff, max_loss_goals_diff_Team2)
        comp_max_win_goals_diff = max(max_win_goals_diff, max_win_goals_diff_Team2)

        # plots for team 1
        figHistogramHt1xG_Combined, figHistogramHt2xG_Combined = plot_scatters(df4CompleteGraph, comp_max_loss_goals_diff, comp_max_win_goals_diff)

        col1, col2 = st.columns(2)
        col1.title("{}".format(team))
        col2.title("{}".format(team_2))

        col1.plotly_chart(BarBPstylesResultsHalftime1)
        col1.plotly_chart(BarBPstylesResultsHalftime2)
        col1.plotly_chart(BarBPstylesXGHalftime1)
        col1.plotly_chart(BarBPstylesXGHalftime2)
        col1.plotly_chart(figHistogramHt1xG_Combined)
        col1.plotly_chart(figHistogramHt2xG_Combined)
        col1.plotly_chart(figScatter_SoG_SoGA)
        col1.plotly_chart(figScatter_SoGA_soG)
        col1.plotly_chart(fig_xg_perminute_home)
        col1.plotly_chart(fig_xg_homexg_complete_game_all_bpse)
        col1.plotly_chart(fig_xg_perminute_home_bigger_55)
        col1.plotly_chart(fig_xg_perminute_home_smaller_45)
        
        col1.dataframe(df4Complete[['Home', 'Opponent', 'IsHome', 'R', 'xG', 'A_xG', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A',
                                    'SoG-H', 'SoG-A',  'xPTS', 'A_xPTS', 'Date', 'xg_halftime', 'Axg_halftime', "A_Red Cards", "H_Red Cards", "halftime"]].style.format({'xG': '{:.1f}', 'A_xG': '{:.1f}', 'SoG-H': '{:.0f}',
                                                    'G-H': '{:.0f}', 'G-A': '{:.0f}', 'BP-H': '{:.0f}',
                                                    'BP-A': '{:.0f}', 'GA-H': '{:.0f}', 'GA-A': '{:.0f}',
                                                    'xPTS': '{:.1f}', 'A_xPTS': '{:.1f}', 'SoG-A': '{:.0f}',
                                                    }))

        
        # plots for team 2
        figHistogramHt1xG_Combined_Team2, figHistogramHt2xG_Combined_Team2 = plot_scatters(df4Complete2Graph_Team2, comp_max_loss_goals_diff, comp_max_win_goals_diff)
        col2.plotly_chart(BarBPstylesResultsHalftime1_Team2)
        col2.plotly_chart(BarBPstylesResultsHalftime2_Team2)
        col2.plotly_chart(BarBPstylesXGHalftime1_Team2)
        col2.plotly_chart(BarBPstylesXGHalftime2_Team2)
        col2.plotly_chart(figHistogramHt1xG_Combined_Team2)
        col2.plotly_chart(figHistogramHt2xG_Combined_Team2)
        col2.plotly_chart(figScatter_SoG_SoGA_Team2)
        col2.plotly_chart(figScatter_SoGA_soG_Team2)
        col2.plotly_chart(fig_xg_perminute_home_Team2)
        col2.plotly_chart(fig_xg_homexg_complete_game_all_bpse_Team2)
        col2.plotly_chart(fig_xg_perminute_home_bigger_55_Team2)
        col2.plotly_chart(fig_xg_perminute_home_smaller_45_Team2)
        col2.dataframe(df4Complete2[['Home', 'Opponent', 'IsHome', 'R', 'xG', 'A_xG', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A',
                                'SoG-H', 'SoG-A',  'xPTS', 'A_xPTS', 'Date', 'xg_halftime', 'Axg_halftime', "A_Red Cards", "H_Red Cards", "halftime"]].style.format({'xG': '{:.1f}', 'A_xG': '{:.1f}', 'SoG-H': '{:.0f}',
                                                'G-H': '{:.0f}', 'G-A': '{:.0f}', 'BP-H': '{:.0f}',
                                                'BP-A': '{:.0f}', 'GA-H': '{:.0f}', 'GA-A': '{:.0f}',
                                                'xPTS': '{:.1f}', 'A_xPTS': '{:.1f}', 'SoG-A': '{:.0f}',
                                                }))