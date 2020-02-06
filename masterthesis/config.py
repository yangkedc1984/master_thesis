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
main_path = (
    "/Users/nickzumbuhl/Desktop/master_thesis/masterthesis"  # what is the ubuntu path?
)
data_input_path = "/Users/nickzumbuhl/Desktop/master_thesis/masterthesis/data"  # what is the ubuntu path?
folder_output = "output"


# change path architecture such that it works on the VM as well !c


class PathArchitecture:

    """
    This seems to be a weird way to handle folder paths
    (especially for the input data --> every user has the data located in a different place)
    """

    def __init__(self, path_main, path_output, path_input):
        self.path_main = path_main
        self.path_output = path_output
        self.path_input = path_input
        self.HARModel = "HARModel"
        self.NN = "NeuralNet"
        self.output_path = None
        self.output_HAR = None
        self.output_LSTM = None

    def make_folder(self):
        if os.path.exists(self.path_main + "/" + self.path_output) is not True:
            os.mkdir(self.path_main + "/" + self.path_output)
        self.output_path = self.path_main + "/" + self.path_output

    def make_subfolder(self):
        if os.path.exists(self.output_path + "/" + "HARModel") is not True:
            os.mkdir(self.output_path + "/" + "HARModel")

        if os.path.exists(self.output_path + "/" + "NeuralNet") is not True:
            os.mkdir(self.output_path + "/" + "NeuralNet")

        self.output_HAR = self.output_path + "/" + "HARModel"
        self.output_LSTM = self.output_path + "/" + "NeuralNet"

    def config_folder_structure(self):
        self.make_folder()
        self.make_subfolder()


instance_path = PathArchitecture(main_path, folder_output, data_input_path)
instance_path.config_folder_structure()
instance_path.output_LSTM
instance_path.output_HAR
