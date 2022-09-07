#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:49:47 2022

@author: padela
"""

import argparse
import json
import os
import shutil
import warnings

# REFACTOR
from os.path import join

import numpy as np
import pandas as pd

# Converter
import pm4py

import hash_maps
import next_act
import utils
from IO import read, folders, create_folders
from load_dataset import prepare_dataset

# Create backup folde
if not os.path.exists('explanations'):
    os.mkdir('explanations')
if not os.path.exists('experiments'):
    os.mkdir('experiments')


def convert_to_csv(filename=str):
    if '.csv' in filename:
        return None
    if '.xes.gz' in filename:
        pm4py.convert_to_dataframe(pm4py.read_xes(filename)).to_csv(path_or_buf=(filename[:-7] + '.csv'), index=None)
        print('Conversion ok')
        return None
    if '.xes' in filename:
        pm4py.convert_to_dataframe(pm4py.read_xes(filename)).to_csv(path_or_buf=(filename[:-4] + '.csv'), index=None)
        print('Conversion ok')
    else:
        raise TypeError('Check the path or the log type, admitted formats : csv, xes, xes.gz')


def modify_filename(filename):
    if '.csv' in filename: return filename
    if '.xes.gz' in filename: return filename[:-7] + '.csv'
    if '.xes' in filename:
        return filename[:-4] + '.csv'
    else:
        None


def read_data(filename, start_time_col, date_format="%Y-%m-%d %H:%M:%S"):
    if '.csv' in filename:
        try:
            df = pd.read_csv(filename, header=0, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(filename, header=0, encoding="cp1252", low_memory=False)
    elif '.parquet' in filename:
        df = pd.read_parquet(filename, engine='pyarrow')
    # if a datetime cast it to seconds
    if not np.issubdtype(df[start_time_col], np.number):
        df[start_time_col] = pd.to_datetime(df[start_time_col], format=date_format)
        df[start_time_col] = df[start_time_col].astype(np.int64) / int(1e9)
    return df


parser = argparse.ArgumentParser(
    description='Main script for Catboost training')

parser.add_argument('--filename_completed', default='data/completed.csv')
parser.add_argument('--filename_running', default=None)
parser.add_argument('--case_id_name', required=True, type=str)
parser.add_argument('--activity_name', required=True, type=str)
parser.add_argument('--start_date_name', required=True, type=str)
parser.add_argument('--end_date_name', type=str, default=None)
parser.add_argument('--resource_name', type=str, default=None)
parser.add_argument('--role_name', type=str, default=None)
parser.add_argument('--predict_activities', type=str, nargs='*', default=None)
parser.add_argument('--retained_activities', type=str, nargs='*', default=None)
parser.add_argument('--lost_activities', type=str, nargs='*', default=None)

parser.add_argument('--date_format', default="%Y-%m-%d %H:%M:%S")
parser.add_argument('--pred_column', default=None)
parser.add_argument('--experiment_name', default='')
parser.add_argument('--pred_attributes', type=str, nargs='*', default=None)
parser.add_argument('--costs', default=None)
parser.add_argument('--working_time', default=None)
parser.add_argument('--mode', default="train")

parser.add_argument('--shap', default=False)
parser.add_argument('--explain', default=False)
parser.add_argument('--override', default=True)  # if True retrains model and overrides previous one
parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--outlier_thrs', default=0, type=int)

args = parser.parse_args()

# mandatory parameters
filename = args.filename_completed
filename_running = args.filename_running
case_id_name = args.case_id_name
activity_name = args.activity_name
start_date_name = args.start_date_name
date_format = args.date_format
pred_column = args.pred_column  # remaining_time if you want to predict that
experiment_name = args.experiment_name
# experiment_name = create_experiment_name(filename, pred_column)


if args.costs is not None:
    costs = json.loads(args.costs)
    working_time = json.loads(args.working_time)
else:
    costs = None
    working_time = None

# optional parameters
end_date_name = args.end_date_name
resource_column_name = args.resource_name
role_column_name = args.role_name
predict_activities = args.predict_activities
retained_activities = args.retained_activities
lost_activities = args.lost_activities
shap_calculation = args.shap
pred_attributes = args.pred_attributes
override = args.override
num_epochs = args.num_epochs
shap = args.shap
explain = args.explain
explain = bool(explain)
outlier_thrs = args.outlier_thrs

convert_to_csv(filename)
filename = modify_filename(filename)
df = read_data(filename, args.start_date_name, args.date_format)
print(df.shape)
if filename_running is not None:
    df_running = read_data(filename_running, args.start_date_name, args.date_format)
    print(df_running.shape)

create_folders(folders, safe=override)

use_remaining_for_num_targets = None
custom_attribute_column_name = None
if pred_column == "total_cost":
    pred_column = "case_cost"
    use_remaining_for_num_targets = False
elif pred_column == "remaining_cost":
    pred_column = "case_cost"
    use_remaining_for_num_targets = True
elif pred_column == "remaining_time":
    use_remaining_for_num_targets = True
elif pred_column == "lead_time":
    pred_column = "remaining_time"
    use_remaining_for_num_targets = False
elif pred_column != "independent_activity" and pred_column != "churn_activity":
    # we can also predict a custom attribute
    if pred_column in df.columns:
        custom_attribute_column_name = pred_column
        pred_column = "custom_attribute"
    else:
        raise NotImplementedError

np.random.seed(1618)  # 6415
prepare_dataset(df=df, case_id_name=case_id_name, activity_column_name=activity_name, start_date_name=start_date_name,
                date_format=date_format,
                end_date_name=end_date_name, pred_column=pred_column, mode="train", experiment_name=experiment_name,
                override=override,
                pred_attributes=pred_attributes, costs=costs,
                working_times=working_time, resource_column_name=resource_column_name,
                role_column_name=role_column_name,
                use_remaining_for_num_targets=use_remaining_for_num_targets,
                predict_activities=predict_activities, lost_activities=lost_activities,
                retained_activities=retained_activities, custom_attribute_column_name=custom_attribute_column_name,
                shap=shap_calculation)

# copy results as a backup
fromDirectory = join(os.getcwd(), 'experiment_files')
toDirectory = join(os.getcwd(), 'experiments', experiment_name)

if os.path.exists(toDirectory):
    answer = None
    while answer not in {'y', 'n'}:
        print('An experiment with this name already exists, do you want to replace the folder storing the data ? [y/n]')
        answer = input()
    if answer == 'y':
        shutil.rmtree(toDirectory)
        shutil.copytree(fromDirectory, toDirectory)
    else:
        print('Backup folder not created')
else:
    shutil.copytree(fromDirectory, toDirectory)
    print('Data and results saved')

print('Starting import model and data..')
if not os.path.exists(f'expls_{experiment_name}'):
    os.mkdir(f'expls_{experiment_name}')
info = read(folders['model']['data_info'])
X_train, X_test, y_train, y_test = utils.import_vars(experiment_name=experiment_name, case_id_name=case_id_name)
model = utils.import_predictor(experiment_name=experiment_name, pred_column=pred_column)
print('Importing completed...')

print('Analyze variables..')
quantitative_vars, qualitative_trace_vars, qualitative_vars = utils.variable_type_analysis(X_train, case_id_name,
                                                                                           activity_name)
warnings.filterwarnings("ignore")
print('Variable analysis done')
print('Variable analysis done')

print('Creating hash-map of possible next activities')
traces_hash = hash_maps.fill_hashmap(X_train=X_train, case_id_name=case_id_name, activity_name=activity_name,
                                     thrs=outlier_thrs)
print('Hash-map created')

# %% Generate and test recommendations
print('Starting generating, evaluating and explaining recommendations')
df_rec = utils.get_test(X_test, case_id_name).reset_index(drop=True)
df_score = utils.create_eval_set(X_test, y_test).values
columns = X_test.columns
next_act.generate_recommendations(df_rec, df_score, columns, case_id_name, pred_column, activity_name,
                                  traces_hash, model, quantitative_vars, qualitative_vars, X_test,
                                  experiment_name, explain=explain, predict_activities=predict_activities)
