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

curr_dir = os.getcwd()
np.random.seed(1618)

import os

from hash_maps import str_list, list_str
import explain_recsys


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def next_act_kpis(trace, traces_hash, model, pred_column, case_id_name, activity_name, quantitative_vars,
                  qualitative_vars, encoding='aggr-hist'):
    if encoding is None:
        raise NotImplementedError()

    if 'Unnamed: 0' in trace.columns:
        del trace['Unnamed: 0']

    if set(trace.columns) != set(model.feature_names_):
        for colname in set(model.feature_names_) - set(trace.columns):
            trace[colname] = np.zeros(len(trace))

    trace = trace[list(model.feature_names_)]
    try :
        del trace[case_id_name]
    except :
        None

    if encoding == 'aggr-hist':
        trace_acts = list(trace[activity_name])
        # trace = trace[[col for col in trace.columns if col!='REQUEST_ID']]
        next_acts = traces_hash.get_val(str_list(trace_acts))
        if next_acts == 'No record found':
            raise NotADirectoryError('Activity missed in hash-table')

        kpis = dict()
        # Create a vector for inference
        last = trace.loc[max(trace.index)].copy()
        last_act = last[activity_name]

        # put in all the columns which are not inferrable a null value
        for var in last.index:
            if var in (set(quantitative_vars).union(qualitative_vars)):
                last[var] = "none"

        # Create a vector with the actual prediction
        if pred_column == 'remaining_time':
            # last['# ACTIVITY=' + last_act] += 1
            actual_prediction = model.predict(list(last))
        elif pred_column == 'independent_activity':
            actual_prediction = model.predict_proba(list(last))[0]  # activity case

        # Update history
        last['# ACTIVITY=' + last_act] += 1

        for next_act in next_acts:
            # Fill with the supposed activity
            last[activity_name] = next_act

            if pred_column == 'remaining_time' :
                kpis[next_act] = model.predict(list(last))
            elif pred_column == 'independent_activity':
                kpis[next_act] = model.predict_proba(list(last))[0] # activity case

        df_final = pd.DataFrame(columns=['Trace', 'Next_act', 'kpi_rel'])
        for idx in range(len(kpis)):
            df_final.loc[idx] = pd.Series({'Trace': str_list(trace_acts), 'Next_act': list(kpis.keys())[idx],
                                           'kpi_rel': list(kpis.values())[idx]})

        return df_final, actual_prediction


# %% Main algoritm
from utils import from_trace_to_score
import utils

def generate_recommendations(df_rec, df_score, columns, case_id_name, pred_column, activity_name, traces_hash, model,
                             quantitative_vars, qualitative_vars, X_test, experiment_name, predict_activities=None,
                             maximize=bool, save=True, explain=False):

    idx_list = df_rec[case_id_name].unique()
    results = list()
    rec_dict = dict()
    real_dict = dict()
    if not os.path.exists('explanations'):
        os.mkdir('explanations')
    if not os.path.exists(f'explanations/{experiment_name}'):
        os.mkdir(f'explanations/{experiment_name}')
    if not os.path.exists(f'recommendations/{experiment_name}'):
        os.mkdir(f'recommendations/{experiment_name}')
    pickle.dump(quantitative_vars, open(f'explanations/{experiment_name}/quantitative_vars.pkl','wb'))
    pickle.dump(qualitative_vars, open(f'explanations/{experiment_name}/qualitative_vars.pkl', 'wb'))

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
            next_activities, actual_prediciton = next_act_kpis(trace, traces_hash, model, pred_column, case_id_name, activity_name,
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
        rec_dict[trace_idx] = {i:j for i,j in zip(next_activities['Next_act'], next_activities['kpi_rel'])}
        real_dict[trace_idx] = {acts[-1] : actual_prediciton}
        pickle.dump(rec_dict, open(f'recommendations/{experiment_name}/rec_dict.pkl','wb'))
        pickle.dump(real_dict, open(f'recommendations/{experiment_name}/real_dict.pkl', 'wb'))

        print(f'The suggested activity is {rec_act}')
        if explain:
            trace_exp = trace.copy()
            start = time.time()
            for var in (set(quantitative_vars).union(qualitative_vars)):
                trace_exp[var] = "none"
            groundtruth_explanation = explain_recsys.evaluate_shap_vals(trace_exp, model, X_test, case_id_name)
            groundtruth_explanation = [a for a in groundtruth_explanation]
            groundtruth_explanation = [trace_idx] + groundtruth_explanation
            groundtruth_explanation = pd.Series(groundtruth_explanation, index=[i for i in df_rec.columns if i!='y'])

            #Save also groundtruth explanations
            groundtruth_explanation.to_csv(f'explanations/{experiment_name}/{trace_idx}_expl_df_gt.csv')
            groundtruth_explanation.drop([case_id_name] + [i for i in (set(quantitative_vars).union(qualitative_vars))],
                                         inplace=True)

            #stampa l'ultima riga di trace normale
            trace.iloc[-1].to_csv(f'explanations/{experiment_name}/{trace_idx}_expl_df_values.csv')
            last = trace.iloc[-1].copy().drop([case_id_name]+[i for i in (set(quantitative_vars).union(qualitative_vars))])
            next_activities = next_activities.iloc[:3] #TODO: Note that is only optimized for minimizing a KPI
            for act in next_activities['Next_act'].values:
                trace_exp.loc[len(trace_exp) - 1, activity_name] = act

                explanations = explain_recsys.evaluate_shap_vals(trace_exp, model, X_test, case_id_name)
                explanations = [a for a in explanations]
                explanations = [trace_idx] + explanations
                explanations = pd.Series(explanations, index=[i for i in df_rec.columns if i!='y'])
                explanations.to_csv(f'explanations/{experiment_name}/{trace_idx}_{act}_expl_df.csv')

                #Take the best-4 deltas
                explanations.drop([case_id_name]+[i for i in (set(quantitative_vars).union(qualitative_vars))], inplace=True)
                deltas_expls = groundtruth_explanation - explanations
                deltas_expls.sort_values(ascending=False, inplace=True)
                idxs_chosen = deltas_expls.index[:4]

                pickle.dump(idxs_chosen, open(f'explanations/{experiment_name}/{trace_idx}_{act}_idx_chosen.pkl', 'wb'))
                pickle.dump(last, open(f'explanations/{experiment_name}/{trace_idx}_last.pkl', 'wb'))
                explain_recsys.plot_explanations_recs(groundtruth_explanation, explanations, idxs_chosen, last, experiment_name, trace_idx, act)

        print(f'The total execution split for 4 explanations generated is {int(time.time()-start)}')

        try:
            results.append([len(trace), len(next_activities), score_reality, res_rec, acts, rec_act])
        except:
            None  

    # total_time = time.time() - start_time
    # print(f'The total execution time is {total_time}')
    if save:
        pickle.dump(results, open(f'/home/padela/Scrivania/results_backup/results_{experiment_name}.pkl', 'wb'))
    return results
