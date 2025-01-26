import streamlit as st
import pandas as pd

# Zuweisen von Punkten basierend auf dem Halbzeit-Ergebnis
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
def df_specific_team(df, team):
    df4Home = df.loc[((df['Home'] == team))]

    df4Home["IsHome"] = 1

    # recalcualte the winner because of the columns switching to bring the selected team in the first column
    df4Home["1x2"] = 0

    df4Home["1x2"] = df4Home.apply(lambda row: calculate_1x2_home(row), axis=1, result_type='reduce')

    # Berechnung Opponentgames
    df4Opponent = df.loc[((df['Opponent'] == team))]

    # recalcualte the winner because of the columns switching to bring the selected team in the first column
    df4Opponent["1x2"] = 0

    df4Opponent["1x2"] = df4Opponent.apply(lambda row: calculate_1x2_Opponent(row), axis=1, result_type='reduce')

    # change the halftime-xg with halftime-Axg:
    # switched the two columns
    # Change the columns for the Opponentmatches of the specific team
    # print("df4Opponent.columns before reassignment oppo", df4Opponent.columns)

    OpponentTeamReversedColumns = ['Opponent', 'Home',  '1x2', 'R',  'G-A', 'G-H', 'BP-A', 'BP-H', 'GA-A', 'GA-H',  
                                    'SoG-A', 'SoG-H', 'SoffG-A', 'SoffG-H',  'FK-A',  "A_Red Cards", "H_Red Cards",
                                    'FK-H','C-A', 'C-H',  'Off-A', 'Off-H', 'GoKeSa-A', 'GoKeSa-H', 'F-A', 
                                    'F-H', 'Round', 'Date', 'IsHome','A_xG', "A_xPTS", "A_GOALS", 'xG', "xPTS", "GOALS", 
                                    'Axg_halftime', 'xg_halftime',"awayxg_complete_game", "homexg_complete_game", "halftime"] 
    
    # Change the columns for the Opponentmatches of the specific team
    df4OpponentReversed = df4Opponent.reindex(
        columns=OpponentTeamReversedColumns)

    df4OpponentReversed.columns = ['Home', 'Opponent', '1x2', 'R', 'G-H', 'G-A', 'BP-H', 'BP-A', 'GA-H', 'GA-A',
                                   'SoG-H', 'SoG-A', 'SoffG-H', 'SoffG-A', 'FK-H', "H_Red Cards", "A_Red Cards",
                                   'FK-A', 'C-H', 'C-A', 'Off-H', 'Off-A', 'GoKeSa-H', 'GoKeSa-A', 'F-H',
                                   'F-A', 'Round', 'Date', 'IsHome', 'xG', "xPTS", "GOALS", 'A_xG', "A_xPTS", "A_GOALS",
                                   'xg_halftime', 'Axg_halftime', "homexg_complete_game", "awayxg_complete_game", "halftime"]

    # print("df4OpponentReversed.columns after reassignment oppo", df4OpponentReversed.columns)
    return df4Home, df4OpponentReversed



@st.cache_data
def calculate_1x2_BPTypes(row):
    if row['BP-H'] > 55:
        return '>55'
    elif row['BP-H'] < 45:
        return '<45'
    else:
        return '45-55'
    
@st.cache_data
def create_df4Complete(df4Home, df4OpponentReversed):
    # Alle Spiele werden als Heimspiel angezeigt, sind aber auch Auswärtsspiele dabei!
    df4Complete = pd.concat([df4Home, df4OpponentReversed], sort=False)

    df4Complete["G-H"] = df4Complete["G-H"].astype('float64')
    df4Complete["G-A"] = df4Complete["G-A"].astype('float64')
    df4Complete["BP-H"] = df4Complete["BP-H"].astype('float64')
    df4Complete["BP-A"] = df4Complete["BP-A"].astype('float64')

    df4Complete["GoalDiff"] = df4Complete["G-H"] - df4Complete["G-A"]
    df4Complete = df4Complete.sort_values("Date",  ascending=False)

    # calculate column with 3 Ballposition types
    df4Complete["BPTypes"] = '0'

    df4Complete["BPTypes"] = df4Complete.apply(lambda row: calculate_1x2_BPTypes(row), 
                                               axis=1, result_type='reduce')

    df4Complete['Date'] = pd.to_datetime(df4Complete['Date'], format="%d.%m.%Y %H:%M")

    # convert datetime to timestamp for scatter visualization
    df4Complete['timestamp'] = df4Complete.Date.astype('int64')//10**9

    df4Complete = df4Complete.sort_values("Date", ascending=False)

    # Create data for scatter graph
    df4Complete["SoG-H"] = df4Complete["SoG-H"].astype(int)
    df4Complete["SoG-A"] = df4Complete["SoG-A"].astype(int)

    return df4Complete