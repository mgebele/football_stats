
import pandas as pd
import numpy as np
import streamlit as st
import traceback

def extract_xg_values(df):
        """ Extracts the xG values from the TIMING_CHART_XG columnn.
            This contains a string formatted as a list of xG values for each minute.
        """
        for index, _ in df.iterrows():
            try:
                timing_chart = df["TIMING_CHART_XG"].loc[index]
                
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
                        # if minute == '45':
                            # print(f"Could not find 45' Total xG for home team at index {index}, trying 44' or 46'")
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

def process_team_names_of_df(x_df, teamnamedict):
    x_df = x_df.replace(teamnamedict)
    return x_df

#######################################################
###  calculate table with two halftimes to one game ###
#######################################################
def df_cleaning_converting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and converts a DataFrame by selecting specific columns, removing duplicates,
    and resetting the index. Adds a new column 'R' initialized with 'X' and converts
    'TIMING_CHART_XG' to a string type. Extracts expected goals (xG) values for
    halftime and assigns them to the DataFrame. Updates the result column 'R' based on
    the goal comparison between home and away teams. Finally, renames the columns for
    consistency and reorders them.
    """
    df = df[['H_TEAMNAMES', 'A_TEAMNAMES', 'H_GOALS', 'A_GOALS', 'H_BALL_POSSESSION', 'A_BALL_POSSESSION', 'H_GOAL_ATTEMPTS', 'A_GOAL_ATTEMPTS',
            'H_SHOTS_ON_GOAL', 'A_SHOTS_ON_GOAL', 'H_SHOTS_OFF_GOAL', 'A_SHOTS_OFF_GOAL', 'H_FREE_KICKS', "H_RED_CARDS", "A_RED_CARDS",
            'A_FREE_KICKS', 'H_CORNER_KICKS', 'A_CORNER_KICKS', 'H_OFFSIDES', 'A_OFFSIDES', 'H_GOALKEEPER_SAVES', 'A_GOALKEEPER_SAVES',
            'H_FOULS', 'A_FOULS', 'A_GAMEINFO', 'A_DATETIME', 'XG', 'GOALS', 'XPTS', 'A_XG', 'A_GOALS', 'A_XPTS', 'TIMING_CHART_XG', 
            "HOMEXG_COMPLETE_GAME", "AWAYXG_COMPLETE_GAME", 'HALFTIME']]
    df = df.drop_duplicates(subset=['H_TEAMNAMES', 'A_TEAMNAMES', 'H_GOALS', 'A_GOALS', 'H_BALL_POSSESSION', 'A_BALL_POSSESSION', 'H_GOAL_ATTEMPTS', 'A_GOAL_ATTEMPTS',
            'H_SHOTS_ON_GOAL', 'A_SHOTS_ON_GOAL', 'H_SHOTS_OFF_GOAL', 'A_SHOTS_OFF_GOAL', 'H_FREE_KICKS', "H_RED_CARDS", "A_RED_CARDS",
            'A_FREE_KICKS', 'H_CORNER_KICKS', 'A_CORNER_KICKS', 'H_OFFSIDES', 'A_OFFSIDES', 'H_GOALKEEPER_SAVES', 'A_GOALKEEPER_SAVES',
            'H_FOULS', 'A_FOULS', 'A_GAMEINFO', 'A_DATETIME', 'XG', 'GOALS', 'XPTS', 'A_XG', 'A_GOALS', 'A_XPTS'], keep='first')
    df = df.reset_index(drop=True)
    df["R"] = 'X'

    df['TIMING_CHART_XG'] = df['TIMING_CHART_XG'].astype('str') 

    df = extract_xg_values(df)

    df.xg_halftime = df.xg_halftime.astype(float).fillna(0.0)
    df.Axg_halftime = df.Axg_halftime.astype(float).fillna(0.0) 

    df.columns = ['Home', 'Opponent', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A',
                'SoG-H', 'SoG-A', 'SoffG-H', 'SoffG-A', 'FK-H',"H_Red Cards", "A_Red Cards",
                'FK-A', 'C-H', 'C-A', 'Off-H', 'Off-A', 'GoKeSa-H', 'GoKeSa-A',
                'F-H', 'F-A', 'Round', 'Date', 'xG', 'GOALS', 'xPTS', 'A_xG', 'A_GOALS', 'A_xPTS',
                "timing_chart_xg", "homexg_complete_game", "awayxg_complete_game", 'halftime', 'R',  'xg_halftime',
                'Axg_halftime']

    df = df[['Home', 'Opponent', 'R', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A',
             'SoG-H', 'SoG-A', 'SoffG-H', 'SoffG-A', 'FK-H',"H_Red Cards", "A_Red Cards",
             'FK-A', 'C-H', 'C-A', 'Off-H', 'Off-A', 'GoKeSa-H', 'GoKeSa-A', 'F-H',
             'F-A', 'Round', 'Date', 'xG', 'GOALS', 'xPTS', 'A_xG', 'A_GOALS', 'A_xPTS',
             'xg_halftime', 'Axg_halftime', "homexg_complete_game", "awayxg_complete_game", 'halftime']]

    df["IsHome"] = 0

    df = df[['Home', 'Opponent', 'R', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A',
             'SoG-H', 'SoG-A', 'SoffG-H', 'SoffG-A', 'FK-H',"H_Red Cards", "A_Red Cards",
             'FK-A', 'C-H', 'C-A', 'Off-H', 'Off-A', 'GoKeSa-H', 'GoKeSa-A', 'F-H',
             'F-A', 'Round', 'Date', 'IsHome', 'xG', 'GOALS', 'xPTS', 'A_xG', 'A_GOALS', 'A_xPTS',
             'xg_halftime', 'Axg_halftime', "homexg_complete_game", "awayxg_complete_game", 'halftime']]

    for i in range(len(df)):
        try:
            if df["G-H"].iloc[i] > df["G-A"].iloc[i]:
                df.loc[i, "R"] = 'H'
            elif df["G-H"].iloc[i] < df["G-A"].iloc[i]:
                df.loc[i, "R"] = 'A'
            else:
                df.loc[i, "R"] = 'D'
        except Exception as e:
            # pass
            print(f"Error at row index {i}: {e}")
            traceback.print_exc()
    
    return df


# get name of the selected team in dropdown
def load_xg_gamestats_sql(saison, team, teamnamedict):

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

    df_complete_saison.columns = df_complete_saison.columns.str.upper()
    df_complete_saison.columns = df_complete_saison.columns.str.replace(' ', '_')

    df_complete_saison = process_team_names_of_df(df_complete_saison, teamnamedict)

    # execute the query and assign it to a pandas dataframe
    dfxg = df_complete_saison[(df_complete_saison.TEAMS == team) | (
        df_complete_saison.A_TEAMS == team)]
    return dfxg


# get all teams for the selected season in dropdown
def load_xg_season_stats_sql(saison, teamnamedict):
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
    else:
        raise ValueError("Unknown league")
    xgtablename = "{}20{}".format(xgprefix, saison.split("_")[1][:2])

    df_complete_saison = pd.read_csv(
        "data/xg/"+xgtablename+".csv", index_col=0, encoding='utf-8')

    df_complete_saison = process_team_names_of_df(df_complete_saison, teamnamedict)
    return df_complete_saison