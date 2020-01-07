import os

"""
Setting up Environment with 'requirements.txt':
"""

# 1. cd to the directory where requirements.txt is located.
# 2. activate your virtualenv.
# 3. run: pip install -r requirements.txt in your shell.

"""
Setting up Environment with 'environment.yml' file:
"""

# 1. conda env create -f environment.yml (creates a new environment based on the yml file)
# 2. conda env list (lists all environments that exists on computer - check whether new environment is installed)
# 3. activate environment in terminal (conda activate environment)

"""
Path Architecture
"""
path_main = '/Users/nickzumbuhl/Desktop/master_thesis/masterthesis'
path_data_input = '/Users/nickzumbuhl/Desktop/master_thesis/masterthesis/Data'
output_folder = 'Output'

os.chdir(path_main)
os.mkdir(path_main + '/' + output_folder)
