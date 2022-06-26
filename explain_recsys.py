#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 17:29:54 2022

@author: padela
"""

import catboost
from catboost import *
import shap
import matplotlib.pyplot as plt
shap.initjs()

import argparse
import json
import logging

from load_dataset import prepare_dataset
import pandas as pd
import os
import shutil
import numpy as np
from write_results import compare_best_validation_curves, histogram_median_events_per_dataset

#REFACTOR
from os.path import join
from IO import read, write, folders, create_folders

#Converter
import pm4py
import os

import pandas as pd
import utils

def evaluate_shap_vals(trace, model, X_test, case_id_name):
    trace = trace.iloc[-1]
    X_test = X_test[trace.index]
    df = X_test.append(trace).reset_index(drop=True)
    df = df[[i for i in X_test.columns if i!=case_id_name]]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)
    return shap_values[-1]
   



