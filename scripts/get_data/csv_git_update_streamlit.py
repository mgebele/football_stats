# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import time
import datetime
from git import Repo
# import re
# import pandas.io.sql as psql
# import mysql.connector
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
saissons_2021 = [
    'bundesliga2021', 'epl2021', 'la_liga2021', 'ligue_12021', 'serie_a2021',
    'bundesliga2022', 'epl2022', 'la_liga2022', 'ligue_12022', 'serie_a2022',
    'bundesliga2023', 'epl2023', 'la_liga2023', 'ligue_12023', 'serie_a2023',
]

for saison in saissons_2021:
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
saissons_ht_2021 = ['b_2122',
                    'l1_2122',
                    'll_2122',
                    'pl_2122',
                    'sa_2122',

                    'b_2223',
                    'll_2223',
                    'l1_2223',
                    'pl_2223',
                    'sa_2223',

                    'b_2324',
                    'll_2324',
                    'l1_2324',
                    'pl_2324',
                    'sa_2324',
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