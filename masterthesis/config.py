import os

"""
Setting up Environment with 'requirements.txt':
"""

# 1. cd to the directory where requirements.txt is located.
# 2. activate your virtualenv.
# 3. run: pip install -r requirements.txt in your shell.
# alternatively: conda install --file requirements.txt

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

if os.path.exists(path_main + '/' + output_folder) is not True:
    os.mkdir(path_main + '/' + output_folder)


class PathArchitecture:
    def __init__(self,
                 path_main,
                 path_output,
                 path_input):
        self.path_main = path_main
        self.path_output = path_output
        self.path_input = path_input
        self.output_path = None

    def make_folder(self):
        if os.path.exists(self.path_main + '/' + self.path_output) is not True:
            os.mkdir(self.path_main + '/' + self.path_output)
        self.output_path = self.path_main + '/' + self.path_output


instance_path = PathArchitecture(path_main,
                                 output_folder,
                                 path_data_input)
instance_path.make_folder()

