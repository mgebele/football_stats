# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import time
from git import Repo
import datetime
import re
import numpy as np
import pandas as pd
import os
import subprocess

# make sure .git folder is properly configured
PATH_OF_GIT_REPO = r'C:\Users\mg\github\football_stats\.git'
now = datetime.datetime.now()
COMMIT_MESSAGE = 'new game update {}'.format(now.date())
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
