import os
import time

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

"main_path: path where all the python files are located and where the output is placed later on"
main_path = "/Users/nickzumbuhl/Desktop/master_thesis/masterthesis"

"data_input_path: folder path where the data is located on your local machine - the data consists of"
"the engineered features. As the raw data is too large (~50GB of data) it was not possible to place on the local"
"machine"
data_input_path = "/Users/nickzumbuhl/Desktop/master_thesis/masterthesis/data"

data_output_dashboard = "/Users/nickzumbuhl/Desktop/dashboard_deployment"


class PathArchitecture:
    def __init__(self, path_main, path_input, path_dashboard_deployment):
        self.path_main = path_main
        self.path_dashboard_deployment = path_dashboard_deployment
        self.output_folder_name = "output"
        self.path_input = path_input
        self.HARModel = "HARModel"
        self.NN = "NeuralNet"
        self.output_path = None
        self.output_HAR = None
        self.output_LSTM = None
        self.output_Tables = None
        self.output_Predictions = None
        self.output_Graphs = None
        self.output_AR = None
        self.output_GridSearch_GA = None

    def make_folder(self):
        if os.path.exists(self.path_main + "/" + self.output_folder_name) is not True:
            os.mkdir(self.path_main + "/" + self.output_folder_name)
        self.output_path = self.path_main + "/" + self.output_folder_name

    def make_sub_folder(self):
        if os.path.exists(self.output_path + "/" + "HARModel") is not True:
            os.mkdir(self.output_path + "/" + "HARModel")

        if os.path.exists(self.output_path + "/" + "NeuralNet") is not True:
            os.mkdir(self.output_path + "/" + "NeuralNet")

        if os.path.exists(self.output_path + "/" + "AutoRegression") is not True:
            os.mkdir(self.output_path + "/" + "AutoRegression")

        if os.path.exists(self.output_path + "/" + "Tables") is not True:
            os.mkdir(self.output_path + "/" + "Tables")

        if os.path.exists(self.output_path + "/" + "Graphs") is not True:
            os.mkdir(self.output_path + "/" + "Graphs")

        if os.path.exists(self.path_dashboard_deployment) is not True:
            os.mkdir(self.path_dashboard_deployment)

        if os.path.exists(self.output_path + "/" + "Predictions") is not True:
            os.mkdir(self.output_path + "/" + "Predictions")

        if os.path.exists(self.output_path + "/" + "GridSearch_GA") is not True:
            os.mkdir(self.output_path + "/" + "GridSearch_GA")

        self.output_HAR = self.output_path + "/" + "HARModel"
        self.output_LSTM = self.output_path + "/" + "NeuralNet"
        self.output_Tables = self.output_path + "/" + "Tables"
        self.output_Graphs = self.output_path + "/" + "Graphs"
        self.output_AR = self.output_path + "/" + "AutoRegression"
        self.output_Predictions = self.output_path + "/" + "Predictions"
        self.output_GridSearch_GA = self.output_path + "/" + "GridSearch_GA"

    def config_folder_structure(self):
        self.make_folder()
        self.make_sub_folder()


folder_structure = PathArchitecture(main_path, data_input_path, data_output_dashboard)
folder_structure.config_folder_structure()
