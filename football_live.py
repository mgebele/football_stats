import glob
import pandas as pd
import numpy as np
import warnings
import json
import streamlit as st
from pathlib import Path

from scripts.subpages.single_team_page import page_teamx
from scripts.subpages.league_table_overview_page import page_league_table

# C:\Users\gebel\github\football_stats>activate football_stats

st.set_page_config(layout="wide")
pd.options.display.float_format = "{:,.1f}".format
warnings.filterwarnings('ignore')

# TEAMNAMES value in teamnamedict must match the htdatan teamname!
global teamnamedict
# C:\Users\mg\JupyterLabDir\Rest\Pr Winning\teamnamedict_streamlit.json
with open('scripts/streamlit_app/teamnamedict_streamlit.json') as f:
    teamnamedict = json.load(f)

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
                        saison.split(" ")[1])

df_complete_saison = pd.read_csv("data/htdatan/"+saison+".csv", index_col=0, encoding='utf-8')
df_complete_saison = df_complete_saison.replace(teamnamedict)

# adapt column names, because they are not the same for all seasons
df_complete_saison.columns = df_complete_saison.columns.str.upper()
df_complete_saison.columns = df_complete_saison.columns.str.replace(' ', '_')
if 'HALFTIME' not in df_complete_saison.columns:
    df_complete_saison['HALFTIME'] = np.nan

print("df_complete_saison.columns")
print(df_complete_saison.columns)

# Erstelle ein Seitenleisten-Menü
page = st.sidebar.radio("Choose a page:", ( 'League Tables', 'Team Analyis'))

# Navigiere zur ausgewählten Seite
if page == 'Team Analyis':
    page_teamx(df_complete_saison, teamnamedict, saison)
elif page == 'League Tables':
    page_league_table(df_complete_saison, saison, teamnamedict)