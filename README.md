```markdown
# Football Stats

## Overview
This repository contains the code for updating a Streamlit application with football statistics from various top leagues, using data scraped from xG and Flashscore websites. The data is first stored as CSV files and then fetched, processed, and visualized using Streamlit.

## This Repository
### Purpose
Update Streamlit repository with xG and Flashscore data by retrieving them from a local database.

### Scripts
- **JupyterLabDir\Rest\streamlit\rest\football\csv_git_update_streamlit.py**:
  - Retrieves xG and Flashscore games from the local MySQL database, commits, and pushes to the repository.


## Other Repository: Web Scraping
The data used in this repository comes from another repository responsible for web scraping football data from xG and Flashscore.

### Overview
Web scraping of football data from top leagues using xG and Flashscore, and storing the data in a local MySQL database.

### Scripts
- **JupyterLabDir\Rest\Pr Winning\scrapeData\2324_startscripts.bat**:
  - Activates the Python environment:
    ```sh
    call activate winning
    ```
  - Changes directory to the scraping scripts and runs them:
    ```sh
    chdir /d JupyterLabDir\Rest\Pr Winning\scrapeData\scraping_flashscore_scripts_2324\
    python 2324_Working_update_sa.py
    python 2324_Working_update_pl.py
    python 2324_Working_update_ligue1.py
    python 2324_Working_update_bundesliga.py
    python 2324_Working_update_laliga.py
    ```
  - Changes directory to the xGoals scripts and runs the update script:
    ```sh
    chdir /d JupyterLabDir\Rest\Pr Winning\scrapeData\xgoals\
    python check_db_xg_update_missing_games_all_leagues.py
    ```

- **Optional Backup**:
  - **JupyterLabDir\Rest\Pr Winning\creating_backups\database_dump.py**:
    - Backs up xG and Flashscore data to Google Drive:
      ```sh
      os.chdir("G:\\My Drive\\footballStatsSQLBackup\\sql data")
      ```

- **Optional Data Verification**:
  - **JupyterLabDir\Rest\Pr Winning\clean_database\flashscore\mail_alert_odd_halftime_numbers.py**:
    - Checks Flashscore data for correctness and alerts on odd halftime numbers.


## Setup

### Environment Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/mgebele/football_stats.git
   cd football_stats
   ```
2. Create and activate the virtual environment:
   ```sh
   python -m venv env
   source env/bin/activate # On Windows, use `env\Scripts\activate`
   ```
3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

### Running the Streamlit Application
1. Ensure you are in the root directory of the repository.
2. Run the Streamlit application:
   ```sh
   streamlit run streamlit_app.py
   ```

### Fetching and Updating Data
- The data processing scripts are located in `scripts/get_data/`.
- These scripts fetch the latest football data from the local MySQL database, process it, and save it as CSV files in the `data/` directory.

### Streamlit Application
- The main Streamlit application script is located in `scripts/streamlit_app/football_live.py`.
- This script loads the processed data, applies necessary transformations, and provides an interactive UI for data visualization.

## Contributing
Contributions are welcome! Please create a pull request or open an issue to discuss your changes.

## License
This project is licensed under the [MIT License](LICENSE).