# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 15:37:20 2021

@author: padel
"""

# %%
import numpy as np
import os
import pickle
import pandas as pd
import random
import time
import math
from collections import Counter
import scipy
import catboost
from IO import read, write, folders, create_folders
import glob
import itertools
import pm4py

import hash_maps
np.random.seed(1618)


# %% Import test and valid
def import_vars(experiment_name=str, case_id_name=str):
    info = read(folders['model']['data_info'])
    column_types = info["column_types"]
    dfTrain = read(folders['model']['dfTrain'], dtype=column_types)
    dfTest = read(folders['model']['dfTest'], dtype=column_types)
    return dfTrain.iloc[:, :-1].reset_index(drop=True), dfTest.iloc[:, :-1].reset_index(drop=True), dfTrain.iloc[:,-1].reset_index(drop=True), dfTest.iloc[:, -1].reset_index(drop=True)


# %% Def import predictor function
def import_predictor(experiment_name=str, pred_column=str):
    if pred_column == 'independent_activity':
        model = catboost.CatBoostClassifier()
        model = read('experiments/' + experiment_name + '/model/model.pkl')

    if pred_column == 'remaining_time':
        model = catboost.CatBoostRegressor()
        model = read('experiments/' + experiment_name + '/model/model.pkl')

    return model


# %% Def import variable type analysiis function
def variable_type_analysis(X_train, case_id_name, activity_name):
    quantitative_vars = list()
    qualitative_trace_vars = list()
    qualitative_vars = list()

    for col in X_train.columns:

        if (col not in [case_id_name, activity_name]) and (col[0] != '#'):
            if type(X_train[col][0]) != str:
                quantitative_vars.append(col)
            else:
                trace = True
                for idx in X_train[case_id_name].unique():  # 150 has been set to be big enough
                    df = X_train[X_train[case_id_name] == idx]
                    if len(set(df[col].unique())) != 1:
                        trace = False
                if trace == True:
                    qualitative_trace_vars.append(col)
                else:
                    qualitative_vars.append(col)

    return quantitative_vars, qualitative_trace_vars, qualitative_vars


# %% Filler for categorical variables
def categorical_var_filler(X_train, activity_name, var_analyzed, q_val=.95):
    # Remove missings
    X_train = X_train[X_train[var_analyzed] != 'missing']

    # Create an empty dict with activities as keys
    act_dict = dict()
    for act in X_train[activity_name].unique():
        act_dict[act] = dict()

    # Fill it considering frequences
    for act, ce_uo in zip(X_train[activity_name], X_train[var_analyzed]):
        if ce_uo in act_dict[act].keys():
            act_dict[act][ce_uo] += 1
        else:
            act_dict[act][ce_uo] = 0
    for act in act_dict.keys():
        for ce_uo in act_dict[act].keys():
            if act_dict[act][ce_uo] < np.quantile(list(act_dict[act].values()), q_val):
                act_dict[act][ce_uo] = 0

    for act in act_dict.keys():
        act_dict[act] = {i: act_dict[act][i] for i in act_dict[act] if act_dict[act][i] != 0}
        act_dict[act] = {key: (val / np.sum(list(act_dict[act].values()))) for key, val in act_dict[act].items()}
        # act_dict[act] = {key:((val)/(np.sum(val))) for key,val in act_dict[act].items()} 

    return act_dict


# %% It's used to understand if consider all the variables and perform a cut on weighted average of having a complete filler
def accept_variability(X_train, activity_name, var, thrs=5):
    accept = True
    for act in X_train[activity_name].unique():
        if len(set(X_train[X_train[activity_name] == act][var])) > thrs:
            accept = False
    return accept


# %% Create a dictionary for filling the other entries
def create_duration_dict(X_train, activity_name):
    duration_dict = dict()
    for act_name in X_train['ACTIVITY'].unique():
        duration_dict[act_name] = int(X_train[X_train['ACTIVITY'] == act_name]['activity_duration'].median())
    return duration_dict


# %% Get a dictionary with for each couple of activities there is the average value of service time
def numerical_var_filler(X_train, case_id_name, activity_name, var=str):
    # Define the final dictionary to fill and the list of idxs
    time_freq_dict = dict()
    case_id_list = X_train[case_id_name].unique()
    # X_train = X_train[X_train[activity_name]!='Pending Request for acquittance of heirs']

    for id_user in case_id_list:
        trace = X_train[X_train[case_id_name] == id_user].reset_index(drop=True)

        for idx in range(len(trace) - 1):
            if hash_maps.str_list(list(trace[activity_name])[idx:idx + 2]) not in time_freq_dict.keys():
                time_freq_dict[hash_maps.str_list(list(trace[activity_name])[idx:idx + 2])] = [trace[var][idx + 1]]
            else:
                time_freq_dict[hash_maps.str_list(list(trace[activity_name])[idx:idx + 2])].append(trace[var][idx + 1])

    return {key: np.median(val) * (np.median(val) >= 0) for key, val in time_freq_dict.items()}


# %% Create act df
def create_act_df(qualitative_vars, act):
    for el in glob.glob('filling_variables/*'):
        globals()[el[18:-4]] = pickle.load(open(el, 'rb'))
    prob_df = pd.DataFrame(columns=qualitative_vars)
    l = [list(globals()[var + '_filler'][act].keys()) for var in qualitative_vars]
    prob_df = pd.DataFrame(list(itertools.product(*l)), columns=qualitative_vars)
    weights = list()
    for idx in prob_df.index:
        l = prob_df.loc[idx]
        weights.append(np.prod([globals()[i + '_filler'][act][l[i]] for i in l.index]))
    prob_df['weights'] = weights
    return prob_df


# %% Get next daytime and weekday

def days_euclidean_division(time_passed=int, act_day=str):
    # Initialize a counter for days
    day_dict = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    curr_day = day_dict[act_day]

    passed_days = time_passed // (24 * 60 * 60)
    daytime2 = time_passed % (24 * 60 * 60)

    curr_day = [day for day in list(day_dict.keys()) if day_dict[day] == ((curr_day + passed_days) % 7)][0]
    if curr_day in ['Saturday', 'Sunday']:
        curr_day = 'Monday'

    return curr_day, daytime2


# %% Evaluation

def generate_dict_frequences_ind_activity(df):
    traces_freq = dict()
    for trace_idx in df['REQUEST_ID'].unique():
        act_list = list(df[df['REQUEST_ID'] == trace_idx]['ACTIVITY'].values)

        if str_list(act_list) not in traces_freq.keys():
            traces_freq[str_list(act_list)] = 1
        else:
            traces_freq[str_list(act_list)] += 1

    return traces_freq


def evaluate_score_ind_activity(acts_list, predict_activities):
    acts_list = str_list(acts_list)
    total = 0
    pa_present = 0
    for key in freq_dict_for_evaluation.keys():
        if acts_list in key:
            total += freq_dict_for_evaluation[key]  # *key.count(predict_activities)
            if predict_activities in key:
                pa_present += freq_dict_for_evaluation[key] * key.count(predict_activities)
    try:
        return pa_present / total
    except:
        return 'Not found in base dataset'


def get_cluster_df(saved=True):
    if not saved:
        print('implementa da create_cluster.py')
        raise NotImplementedError()

    if saved and (experiment == 'lead_time'):
        None

    if saved and (experiment == 'independent_activity'):
        return pickle.load(open(f'vars/{experiment}/X_test_preprocessed_clustering.pkl', 'rb')), pickle.load(
            open(f'vars/{experiment}/X_test_centroids.pkl', 'rb'))


def get_centroids(saved=True):
    if not saved:
        encoding_df = get_cluster_df()[0]
        centroids = dict()
        labels = encoding_df['cluster'].unique()
        for label in labels:
            # if label != -1 : 
            centroids[str(label)] = np.array(np.mean(encoding_df[encoding_df['cluster'] == label].iloc[:, :-1], axis=0))
        return centroids
    else:
        return pickle.load(open(f'vars/{experiment}/centroids_dict.pkl', 'rb'))


def get_test(X_test, case_id_name):
    df_rec = pd.DataFrame(columns=X_test.columns)

    for idx in X_test[case_id_name].unique():
        df = X_test[X_test[case_id_name] == idx]
        cut = int(len(df) * random.uniform(0, 1)) + 2  # 2 because one for the floor and one for the pred
        df = df.iloc[:cut].reset_index(drop=True)
        df_rec = pd.concat([df_rec, df])

    return df_rec.reset_index(drop=True)


def create_eval_set(X_test, y_test):
    X_test['y'] = y_test
    columns = [col for col in X_test.columns if (col[0] == '#' or col == 'y')]
    eval_df = X_test[columns]
    return eval_df


def trace_to_encoding(trace, activity_name, columns, resize=False):
    encoded_trace = pd.Series()
    for el in [col for col in columns if col[0] == '#']:
        encoded_trace[el] = 0

    # Si può ottimizzare mettendo fuori come comincia una traccia
    for act in trace:
        encoded_trace['# ' + activity_name + '=' + act] += 1
    if resize:
        encoded_trace = (np.array(encoded_trace) / np.sum(encoded_trace)).reshape(1, -1)
    else:
        encoded_trace = np.array(encoded_trace)  # .reshape(1,-1)

    return encoded_trace


def closest_centroid_traces(encoded_trace, centroids, encoding_df):
    closeness_vals = {key: euclidean_distances(encoded_trace, value.reshape(1, -1)) for key, value in centroids.items()}
    centroid = [key for key in closeness_vals.keys() if closeness_vals[key] == min(closeness_vals.values())][0]
    return X_test_centroids[X_test_centroids['cluster'] == int(centroid)].iloc[:, :-1]


def from_trace_to_score(trace, pred_column, activity_name, df_score, columns, predict_activities=None):
    encoded_trace = trace_to_encoding(trace, activity_name, columns, resize=False)
    l = list()
    # list(trace[activity_name]) + [rec_act], pred_column, activity_name, df_score, columns
    if pred_column == 'independent_activity':
        index_pa = list(columns).index('# ACTIVITY='+predict_activities[0]) - sum([('#' not in i) for i in columns])
        for line in df_score[:,:-1]:
            if (encoded_trace <= line).all():
                l.append(line - encoded_trace)
        if len(l) > 0:
            return np.mean(np.array(l), axis=0)[index_pa]
        else:
            return None

    if pred_column == 'remaining_time':
        for idx in range(len(df_score)):
            if (encoded_trace <= df_score[idx][:-1]).all():
                l.append(df_score[idx][-1])

        if len(l) > 0:
            return np.mean(np.array(l))
        else:
            return None

def get_train_test_indexes(path_to_complete_df=str):
    dataset = pd.read_csv(path_to_complete_df)


# Change the encoding of the history wrt prediction work
def change_history(df, activity_name):
    for i, row in df.iterrows():
        act = df.at[i, activity_name]
        if df.at[i, '# ' + activity_name + '=' + act]!=0: df.at[i, '# ' + activity_name + '=' + act] -= 1
    return df

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


