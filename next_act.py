# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:45:44 2021

@author: padel
"""

# %% Import Packages and dataframes
import os
import pickle
import pandas as pd
import numpy as np
import time
from IO import read, write, folders

curr_dir = os.getcwd()
np.random.seed(1618)

import argparse
import json

import os
import numpy as np

from hash_maps import str_list, list_str
import explain_recsys

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def next_act_kpis(trace, traces_hash, model, pred_column, case_id_name, activity_name, quantitative_vars,
                  qualitative_vars, encoding='aggr-hist'):
    if encoding is None:
        raise NotImplementedError()

    if encoding == 'aggr-hist':
        trace_acts = list(trace[activity_name])
        # trace = trace[[col for col in trace.columns if col!='REQUEST_ID']]
        next_acts = traces_hash.get_val(str_list(trace_acts))
        if next_acts == 'No record found':
            raise NotADirectoryError('Activity missed in hash-table')

        kpis = dict()
        for next_act in next_acts:

            # Create a vector for inference
            last = trace.loc[max(trace.index)].copy()

            # Fill with the supposed activity
            last[activity_name] = next_act

            # put in all the columns which are not inferrable a null value
            for var in last.index:
                if var in (set(quantitative_vars).union(qualitative_vars)):
                    last[var] = "none"

            if pred_column == 'remaining_time' :
                kpis[next_act] = model.predict(list(last)[1:])
            elif pred_column == 'independent_activity':
                kpis[next_act] = model.predict_proba(list(last)[1:])[0] # activity case

        df_final = pd.DataFrame(columns=['Trace', 'Next_act', 'kpi_rel'])
        for idx in range(len(kpis)):
            df_final.loc[idx] = pd.Series({'Trace': str_list(trace_acts), 'Next_act': list(kpis.keys())[idx],
                                           'kpi_rel': list(kpis.values())[idx]})

        return df_final


# %% Main algoritm
from utils import from_trace_to_score
import utils

def generate_recommendations(df_rec, df_score, columns, case_id_name, pred_column, activity_name, traces_hash, model,
                             quantitative_vars, qualitative_vars, X_test, experiment_name, predict_activities=None,
                             maximize=bool, save=True, explain=True):

    idx_list = df_rec[case_id_name].unique()
    start_time = time.time()
    results = list()
    expl_df = pd.DataFrame(columns=[i for i in df_rec.columns if i!='y'])
    if not os.path.exists('explanations'):
        os.mkdir('explanations')
    if not os.path.exists(f'explanations/{experiment_name}'):
        os.mkdir(f'explanations/{experiment_name}')

    for trace_idx in idx_list:
        trace = df_rec[df_rec[case_id_name] == trace_idx].reset_index(drop=True)
        # trace = trace.iloc[:(randrange(len(trace)) - 1)]
        trace = trace.reset_index(drop=True) # trace.iloc[:, :-1].reset_index(drop=True)
        import utils
        try:
            # take activity list
            acts = list(df_rec[df_rec[case_id_name] == trace_idx].reset_index(drop=True)[activity_name])

            # Remove the last (it has been added because of the evaluation)
            trace = trace.iloc[:-1].reset_index(drop=True)
        except:
            import ipdb; ipdb.set_trace()


        try:
            next_activities = next_act_kpis(trace, traces_hash, model, pred_column, case_id_name, activity_name,
                                            quantitative_vars, qualitative_vars, encoding='aggr-hist')
        except:
            print('Next activity not found in transition system')
            continue

        try:
            rec_act = next_activities[next_activities['kpi_rel'] == min(next_activities['kpi_rel'])]['Next_act'].values[
                0]
            other_traces = [
                next_activities[next_activities['kpi_rel'] != min(next_activities['kpi_rel'])]['Next_act'].values]
        except:
            try:
                if len(next_activities) == 1:
                    print('No other traces to analyze')
            except:
                print(trace_idx, 'check it')
        score_before = list()
        print(other_traces)
        # Evaluate the score for each trace
        for act in other_traces[0]:
            score_before.append(
                utils.from_trace_to_score(list(trace[activity_name]) + [act], pred_column, activity_name, df_score,
                                          columns, predict_activities=predict_activities))

        score_before = [i for i in score_before if i]  # Remove None values that can come from empty lists

        res_rec = from_trace_to_score(list(trace[activity_name]) + [rec_act], pred_column, activity_name, df_score, columns, predict_activities=predict_activities)

        try:
            score_reality = from_trace_to_score(acts, pred_column, activity_name, df_score, columns, predict_activities=predict_activities)
            diff_reality = score_reality - res_rec
        except:
            None
        try:
            print(
                f'Len trace = {len(trace)}, #following_traces = {len(next_activities)}, KPIno_rec {score_reality}, KPIrec {res_rec}, diff{diff_reality}')
        except:
            print('Not in dataset')

        print(f'The suggested activity is {rec_act}')
        if explain:
            for var in (set(quantitative_vars).union(qualitative_vars)):
                trace[var] = "none"
            groundtruth_explanation = explain_recsys.evaluate_shap_vals(trace, model, X_test, case_id_name)
            trace.loc[len(trace) - 1, activity_name] = rec_act

            delta_explanations = groundtruth_explanation - explain_recsys.evaluate_shap_vals(trace, model, X_test, case_id_name)
            delta_explanations = [a for a in delta_explanations]
            delta_explanations = [trace_idx] + delta_explanations
            delta_explanations = pd.Series(delta_explanations, index=expl_df.columns)
            try:
                expl_df = expl_df.append(delta_explanations, ignore_index=True)
            except:
                None

            expl_df.to_csv(f'explanations/{experiment_name}/{trace_idx}_expl_df.csv')

        try:
            results.append([len(trace), len(next_activities), score_reality, res_rec, acts, rec_act])
        except:
            None  

    total_time = time.time() - start_time
    print(f'The total execution time is {total_time}')
    if save:
        pickle.dump(results, open(f'/home/padela/Scrivania/results_backup/results_{experiment_name}.pkl', 'wb'))
    return results
