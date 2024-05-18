# football_stats
# activate pip env "winning"

Webscraping of football data top leagues from xg and flashscore and store in local database
C:\Users\gebel\github\JupyterLabDir\Rest\Pr Winning\scrapeData\2324_startscripts.bat
    call activate winning

    chdir /d C:\Users\mg\JupyterLabDir\Rest\Pr Winning\scrapeData\scraping_flashscore_scripts_2324\

    python 2324_Working_update_sa.py
    python 2324_Working_update_pl.py
    python 2324_Working_update_ligue1.py
    python 2324_Working_update_bundesliga.py
    python 2324_Working_update_laliga.py

    chdir /d C:\Users\mg\JupyterLabDir\Rest\Pr Winning\scrapeData\xgoals\

    python check_db_xg_update_missing_games_all_leagues.py


(Optional) Backup xg flashscore data in drive:
C:\Users\mg\JupyterLabDir\Rest\Pr Winning\creating_backups\database_dump.py
os.chdir("G:\\My Drive\\footballStatsSQLBackup\\sql data")

(Optional) Check flashscore data for correctness: 
C:\Users\mg\JupyterLabDir\Rest\Pr Winning\clean_database\flashscore\mail_alert_odd_halftime_numbers.py


Update streamlit repo football xg and flashscore data by getting them from local database from hp:
C:\Users\mg\JupyterLabDir\Rest\streamlit\rest\football\csv_git_update_streamlit.bat
    call activate automate_385

    chdir /d C:\Users\mg\JupyterLabDir\Rest\streamlit\rest\football\
    python csv_git_update_streamlit.py # get xg games local mysql db, commit push repo. Get flashscore games from local db and commit push git repo. 

    
<!-- 
    chdir /d C:\Users\mg\github\football_stats\
    python csv_git_update_streamlit.py # git add all and login git and push to streamlit repo  -->