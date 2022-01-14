# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 15:42:38 2022
@author: Noemi
"""
import numpy as np
import pandas as pd
import utils.feature_engineering as fe
import utils.MLP_model as MLP
import utils.LGBM_model as LGBM
import sys 

if __name__ == "__main__":
    
    # Get working directory path 
    workdir = sys.argv[1] 
    output_path = sys.argv[2]

    # Load train files
    X_train_station = workdir + 'DATA_RAINFALL/Train/Train/full_X_train.csv'
    Y_train_station = pd.read_csv(workdir+'DATA_RAINFALL/Train/Train/full_Y_train.csv',sep=",",header=0)
    
    # Load test files
    path_baseline_test = workdir+'DATA_RAINFALL/Test/Test/Baselines/'
    X_station_test = workdir+"DATA_RAINFALL/Test/Test/full_X_test.csv"

    ########################################################################
    #                                                                      #
    #                         Feature Engineering                          #
    #                                                                      #
    ########################################################################
    
    # Train set
    X_station_train_mean=fe.train_feature(X_train_station)
    # Test set 
    X_station_test_mean = fe.test_feature(X_station_test)


    MLP.modele(X_station_train_mean,Y_train_station,X_station_test_mean)
    LGBM.modele(X_station_train_mean,Y_train_station,X_station_test_mean)
    print("predictions .csv files were saved in the git folder.")


