
import pandas as pd
import streamlit as st
from scripts.data_processing.pre_processing_loading import process_team_names_of_df, load_xg_season_stats_sql, df_cleaning_converting
from scripts.data_processing.processing_augmentation import df_specific_team, create_df4Complete, calc_halftime_result_a, calc_halftime_result_h, get_table_ballpositionstyle, get_table_counterstyle, get_table_even  # , post_process_table

def page_league_table(df_complete_saison : pd.DataFrame, saison : str, teamnamedict : dict):
    
    # TODO: iteriere über alle teams und packe alle ergebnisse in eine seperate liste und die kommt
    # dann in eine neue seperate ergebnis df!
    result_table_ballpositionstyle_list = []
    result_table_counterstyle_list = []
    result_table_evenstyle_list = []
    error_list = []

    for team in df_complete_saison["H_TEAMNAMES"].unique():
        try:
            # teamname corrected that it fits to htdatan teamnames
            # team = df_complete_saison["H_TEAMNAMES"].unique()[0]
            dfxg = load_xg_season_stats_sql(saison, teamnamedict)

            # convert xg teamnames to correct ones that are used in htdatan
            dfxg = process_team_names_of_df(dfxg, teamnamedict)

            # rename columns for
            dfxg_rename = dfxg.rename(
                columns={'TEAMS': 'H_TEAMNAMES', 'A_TEAMS': 'A_TEAMNAMES'})

            dfxg_df_merged = pd.merge(
                df_complete_saison, dfxg_rename, on=["H_TEAMNAMES", "A_TEAMNAMES"])
            df = dfxg_df_merged.drop_duplicates()

            # the following is only so the functions have the expecting columns
            df["HOMEXG_COMPLETE_GAME"] = ""
            df["AWAYXG_COMPLETE_GAME"] = ""
            df["last_game_minute"] = -1
            df["start_min_game"] = -1

            # rename xg table columns
            df.rename(columns={'A_GOALS_x': 'A_GOALS'}, inplace=True)
            df.rename(columns={'A_GOALS_y': 'A_GOALS_COMPLETE_GAME'}, inplace=True)

            dfxg_df_merged_cleaned = df_cleaning_converting(df)

            df4Home, df4OpponentReversed = df_specific_team(dfxg_df_merged_cleaned, team)
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

        except Exception as e:
            print(e)

    # rename xg table columns
    df.rename(columns={'A_GOALS_x': 'A_GOALS'}, inplace=True)
    df.rename(columns={'A_GOALS_y': 'A_GOALS_COMPLETE_GAME'}, inplace=True)
    df.rename(columns={'homexg_complete_game': 'HOMEXG_COMPLETE_GAME'}, inplace=True)
    df.rename(columns={'awayxg_complete_game': 'AWAYXG_COMPLETE_GAME'}, inplace=True)

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