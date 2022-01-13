# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 15:42:38 2022

@author: Noemi
"""
import numpy as np
import pandas as pd
import feature_engineering as fe
import MLP_model as MLP
import LGBM_model as LGBM

path_utils = workdir+'utils/'
# DATASET
            # Train
X_train_station = "./DATA_RAINFALL/Train/Train/full_X_train.csv"
Y= "./DATA_RAINFALL/Train/Train/full_Y_train.csv"
Y_train_station = pd.read_csv(Y,sep=",",header=0)
            # Test
path_baseline_test = './DATA_RAINFALL_/Test/Test/Baselines/'
X_station_test = "./DATA_RAINFALL_/Test/Test/full_X_test.csv"

# Feature Engineering
            # Train
X_station_train_mean=fe.train_feature(X_train_station)
            #Test_mean
X_station_test_mean = fe.test_feature(X_station_test)



MLP.modele(X_station_train_mean,Y_train_station,X_station_test_mean)
LGBM.modele(X_station_train_mean,Y_train_station,X_station_test_mean)


