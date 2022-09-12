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

#Internal packages
import hash_maps
import next_act
import utils
from IO import read, folders, create_folders
from load_dataset import prepare_dataset





parser = argparse.ArgumentParser(
    description='Main script for Catboost training and creating recommendations')

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
parser.add_argument('--outlier_thrs', default=0.01, type=int)

args = parser.parse_args()

# mandatory parameters
filename = args.filename_completed
filename_running = args.filename_running
case_id_name = args.case_id_name
activity_name = args.activity_name
start_date_name = args.start_date_name
date_format = args.date_format
pred_column = args.pred_column
experiment_name = args.experiment_name

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

import dill
#Dump your session
filename = 'session_save.pkl'
# dill.dump_session(filename)
#Load the session again:
dill.load_session(filename)

class RecSys:
    def __init__(self, filename_completed, filename_running, case_id_name, activity_name, start_date_name, date_format,
                 pred_column, experiment_name, end_date_name, resource_column_name, role_column_name,
                 predict_activities, retained_activities, lost_activities, shap_calculation, pred_attributes,
                 override, num_epochs, shap, explain, outlier_thrs):

        self.outlier_thrs = outlier_thrs
        self.explain = explain
        self.shap = shap
        self.num_epochs = num_epochs
        self.override = override
        self.pred_attributes = pred_attributes
        self.shap_calculation = shap_calculation
        self.lost_activities = lost_activities
        self.retained_activities = retained_activities
        self.predict_activities = predict_activities
        self.role_column_name = role_column_name
        self.end_date_name = end_date_name
        self.filename_completed = filename_completed
        self.filename_running = filename_running
        self.case_id_name = case_id_name
        self.activity_name = activity_name
        self.start_date_name = start_date_name
        self.date_format = date_format
        self.pred_column = pred_column
        self.experiment_name = experiment_name
        self.resource_column_name = resource_column_name

    def generate_train_and_test_logs(self):
        print('Importing train and test logs')
        self.X_train, self.X_test, self.y_train, self.y_test = utils.import_vars(experiment_name=self.experiment_name,
                                                                                 case_id_name=self.case_id_name)
        print('Importing completed')

    def import_model(self):
        print('Importing model')
        self.model = utils.import_predictor(experiment_name=self.experiment_name, pred_column=self.pred_column)
        print('Model importing completed')

    def import_vartypes(self):
        print('Analyze variables..')
        if 'X_train' not in vars():
            self.generate_train_and_test_logs()
        self.quantitative_vars, self.qualitative_trace_vars, self.qualitative_vars = utils.variable_type_analysis(X_train=self.X_train,
                                                                                                                  case_id_name=self.case_id_name,
                                                                                                                  activity_name=self.activity_name)
        print('Variable analysis done')

    def create_transtition_system(self):
        print(f'Creating transition system with a threshold for outliers {outlier_thrs}')
        if 'X_train' not in vars():
            self.generate_train_and_test_logs()
        self.traces_hash = hash_maps.fill_hashmap(X_train=self.X_train, case_id_name=self.case_id_name, activity_name=self.activity_name,
                                             thrs=self.outlier_thrs)
        print('Transition system created')

    def create_running_log(self):
        print('Create running log for validation procedure')
        if 'X_test' not in vars():
            self.generate_train_and_test_logs()
        self.df_rec = utils.get_test(self.X_test, self.case_id_name).reset_index(drop=True)
        print('Creation completed')

    def create_log_for_running_cases(self):

        if set('X_test','y_test') not in vars():
            print('train and test not generated, starting automatic generation')
            self.generate_train_and_test_logs()

        self.df_score = utils.create_eval_set(self.X_test, self.y_test).values

prova = RecSys(filename_completed=filename, filename_running=filename_running, case_id_name=case_id_name, activity_name=activity_name,
               start_date_name=start_date_name, date_format=date_format,pred_column=pred_column, experiment_name=experiment_name,
               end_date_name=end_date_name, resource_column_name=resource_column_name, role_column_name=role_column_name,
               predict_activities=predict_activities, retained_activities=retained_activities, lost_activities=lost_activities,
               shap_calculation=shap, pred_attributes=pred_attributes, override=override, num_epochs=num_epochs, shap=shap,
               explain=explain, outlier_thrs=outlier_thrs)
print('Before is'+4*'\n')
# print(prova.y_test)

print('Later is')
prova.generate_train_and_test_logs()
print(prova.y_test)


if __name__ == '__main__':
    print('Execution_completed')
