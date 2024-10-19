# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# exceute with pip automate_385 lib

# %%
import time
import datetime
from git import Repo
import re
import pandas.io.sql as psql
import mysql.connector
import MySQLdb
import pandas as pd
import os
import subprocess

os.chdir("C:\\Users\\mg\\github\\football_stats\\data")

# # # start - do xg games # # #
# get all the gamestatistics from in dropdown specified league and season
# setup the database connection.  
db = MySQLdb.connect(host='127.0.0.1',
                     database='xg',
                     user='root',
                     password='root')

cursor = db.cursor()     # get the cursor

cursor.execute("USE xg")  # select the database
cursor.execute("SHOW TABLES")

# %%
# TODO: New saison 2025 2026, change here the names to xx_2025
saissons = [
    'bundesliga2024', 'epl2024', 'la_liga2024', 'ligue_12024', 'serie_a2024',
]

for saison in saissons:
    # get all teamnames
    queryall = "SELECT * FROM {}".format(saison)
    print(queryall)
    # execute the query and assign it to a pandas dataframe
    dfall = pd.read_sql(queryall, con=db)
    # convert string to df to use process_team_names_of_df function
    df_teamname = pd.DataFrame(dfall)
    df_teamname.to_csv("xg\\{}.csv".format(saison),
                       encoding='utf-8', index=True)

# make sure .git folder is properly configured
PATH_OF_GIT_REPO = r'C:\Users\mg\github\football_stats\.git'
now = datetime.datetime.now()
COMMIT_MESSAGE = 'new game update xg {}'.format(now.date())
time.sleep(1)
repo = Repo(PATH_OF_GIT_REPO)
time.sleep(1)
repo.git.add(update=True)
time.sleep(1)
subprocess.call(['git', 'add', '.'])
time.sleep(1)
repo.index.commit(COMMIT_MESSAGE)
time.sleep(1)
origin = repo.remote(name='origin')
time.sleep(1)
origin.push()

db.close()

# # # end - do xg games # # #

time.sleep(3)

# # # start - do htdatan games # # #

# get all the gamestatistics from in dropdown specified league and season
# setup the database connection.
db = MySQLdb.connect(host='127.0.0.1',
                     database='htdatan',
                     user='root',
                     password='root')

cursor = db.cursor() 
cursor.execute("USE htdatan")  # select the database
cursor.execute("SHOW TABLES")

# %%
# TODO: New saison 2025 2026, change here the names!
saissons_ht_2021 = [    
                    'b_2425',
                    'll_2425',
                    'l1_2425',
                    'pl_2425',
                    'sa_2425',
                    ]

for saison in saissons_ht_2021:
    # get all teamnames
    queryall = "SELECT * FROM {}".format(saison)
    print(queryall)
    # execute the query and assign it to a pandas dataframe
    dfall = pd.read_sql(queryall, con=db)
    # convert string to df to use process_team_names_of_df functions
    df_teamname = pd.DataFrame(dfall)
    df_teamname.to_csv("htdatan\\{}.csv".format(saison),
                       encoding='utf-8', index=True)

# make sure .git folder is properly configured
PATH_OF_GIT_REPO = r'C:\Users\mg\github\football_stats\.git'
now = datetime.datetime.now()
COMMIT_MESSAGE = 'new game update htdatan {}'.format(now.date())

time.sleep(1)
repo = Repo(PATH_OF_GIT_REPO)
time.sleep(1)
repo.git.add(update=True)
time.sleep(1)
subprocess.call(['git', 'add', '.'])
time.sleep(1)
repo.index.commit(COMMIT_MESSAGE)
time.sleep(1)
origin = repo.remote(name='origin')
time.sleep(1)
origin.push()

db.close()

print("all run without Exception")

# # # end - do htdatan games # # #