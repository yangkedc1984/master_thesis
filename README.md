## ReadMe: LSTM Neural Networks and HAR Models for Realized Volatility - An Application to Financial Volatility Forecasting

The following document aims to provide an overview of the whole code base used throughout the thesis. Generally speaking, this repository contains code for all the data loading, preprocessing, estimation, training, and prediction. For reporting the results an individual repository was set up containing a dashboard with all the relevant information. The dashboard repository can be found here: https://github.com/nickzumbuehl/dashboard_deployment and runs under the url: http://nick-vola.herokuapp.com.

This document outlines the following points:
1. Setting up the Environment
2. Pipeline Overview
   - Input Files
   - Process Files
   - Output Files
3. Python-Files in the Repo
   - feature_engineering.py
   - AutoRegression_Model.py & run_AutoRegression_Model.py
   - HAR_Model.py & run_HAR_model.py
   - LSTM.py, run_LSTM.py
     - GeneticAlgorithm.py
     - run_GeneticAlgorithm.py
   - dashboard_data_prep.py
   

### 1. Setting up Environment
The ```requirements.txt``` contains information on all the relevant packages and versions used in the code base. In order to set up the environment, the please follow the subsequent process:
1. cd to the directory where your ```requirements.txt```is located
2. activate your virtual environment
3. run: ``` pip install -r requirements.txt``` in your shell. Alternatively, when working with conda run: ```conda install --file requirements.txt```.

## Pipeline 
![](pipeline_advance.png)
## Description of each file

#write a brief text of how the whole thesis is structured. (Data Preprocessing, Feature #Engineering, HAR Model (Data Preprocessing & Estimation), LSTM Model (Data #Preprocessing), Genetic Algorithm (Hyperparameter Optimization), Results (Accuracy #Measures, Graphs, Export of all results)

https://help.github.com/en/github/writing-on-github/basic-writing-and-formatting-syntax
