# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:35:43 2021

@author: padel
"""

#%% Import packages
import pandas as pd
import pickle
import numpy as np
import os
import tqdm as tqdm

from scipy import stats
import json 
from statistics import median

from sklearn.metrics import f1_score, confusion_matrix, average_precision_score, roc_auc_score, make_scorer, \
    mean_absolute_error, log_loss
from sklearn.preprocessing import LabelEncoder
import shap
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from catboost.utils import select_threshold
from explainable import (find_explanations_for_completed_cases,
                         find_explanations_for_running_cases)
from write_results import (prepare_csv_results, write_and_plot_results,
                           write_scores, write_grid_results, compare_best_validation_curves)
from logme import log_it
from IO import read, write, folders
from os.path import join, exists

import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(1618)



#%% Implement a prediction


        
#%% Make the dataframe suitable for the ProM tool
def transform_dataset_for_prom(): 
    ct = 0
    X_train['ID ADDED']=np.zeros(len(X_train))
    
    for idx in range(len(X_train)):
        X_train['ID ADDED'][idx] = int(ct)
        if X_train['ACTIVITY'][idx+1]=='Request created':
            ct+=1
            
    X_train.to_csv('traindf.csv', index=False)    

#%% It helps me to train, adjust please
def oversampling(X_train, y_train, idxs):
    from collections import Counter
    counter = Counter(y_train)
    print('Before was', counter)
    from imblearn.over_sampling import SMOTENC
    #Fill idxs with correct number of inputs
    sm = SMOTENC(random_state=1618, categorical_features=idxs)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    counter = Counter(y_train)
    print('then is', counter)
    return X_train, y_train

def undersampling(X_train, y_train):
    
    raise NotImplementedError()
    #Not implemented but already tried
    

#%% Prepare datasets for running/testing analysis


#%%
# t = 1547221888788

def get_cut_test_set(t=float):
    
    start_end_couple = pickle.load(open('vars/start_end_couple.pkl','rb'))
    X_test_idxs = list()
    for line in start_end_couple:
        if (line[1] < t) and (line[2] > t):
            X_test_idxs.append(line[0])

def create_new_test_set(t): 
    
    final_testset = pd.DataFrame(columns=X_test.columns)
    res_list = list()
    ex = 0
    for idx in X_test_idxs:
        df = non_processed_complete[non_processed_complete['REQUEST_ID']==idx].reset_index(drop=True)
        
        l = 0 
        try : 
            while (not ((df['START_DATE'][l] <= t) and (df['END_DATE'][l] >= t))):
                l+=1
            if l!=0 :
                res_list.append([idx, l])
            
        except : 
            ex += 1
            
    len_test_idxs = {key:item for key,item in zip([i[0] for i in res_list], [i[1] for i in res_list])} 
    for idx in [i[0] for i in res_list]: 
        if idx in X_test['REQUEST_ID'].unique():
            df = X_test[X_test['REQUEST_ID']==idx].iloc[:len_test_idxs[idx]+1]
            final_testset = pd.concat([final_testset, df])
            
        if idx in X_train['REQUEST_ID'].unique():
            df = X_train[X_train['REQUEST_ID']==idx].iloc[:len_test_idxs[idx]+1]
            final_testset = pd.concat([final_testset, df])
    
# pickle.dump(final_testset, open('vars/test_set_split_temporarly.pkl','wb'))
# pickle.dump(df_testset, open('vars/test_set_split_temporarly_with_y.pkl','wb'))

#%% put y in final test_set
#TODO: aggiusta codice
def fill_y_values():
    df_testset = pickle.load(open('vars/test_set_split_temporarly.pkl','rb')).reset_index(drop=True)
    df_testset['lead_time'] = np.zeros(len(df_testset))
    y_dict = dict()
    for idx in df_total['REQUEST_ID'].unique():
        df = df_total[df_total['REQUEST_ID']==idx]
        if len(set(df['lead_time']))!=1:
            print(idx)
        else :
            y_dict[str(idx)] = list(set(df['lead_time']))[0]
            
    l = list()
    for idx in df_testset.index: 
        l.append(y_dict[str(df_testset.loc[idx]['REQUEST_ID'])])
    df_testset['lead_time'] = l

#%%
def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    return results


#%%
# ratio_cut = 5
# maximize = False 
# is_real = False
# predict_activities = 'Pending Liquidation Request'

# experiment, maximize, predict_activities = 'indipendent_activity', False, 'Pending Liquidation Request'
# maximize = 'max'*maximize + 'min'*(not maximize)

results = pickle.load(open('/home/padela/Scrivania/PhD/Prescriptive-Analytics/vars/lead_time/test_results_min_None.pkl', 'rb'))
def results_to_df(ratio_cut=float, maximize=bool, is_real=True, predict_activities=str, filling=False, save=False):
    
    print(ratio_cut, maximize, is_real, predict_activities)
    #Set values for importer
    maximize = 'max'*maximize + 'min'*(not maximize)
    is_real = '_real'*(is_real)
    ratio_cut = float(ratio_cut)
    
    #Import results
    # results = pickle.load(open(f'vars/{experiment}/test_results_ratiocut{ratio_cut}_{maximize}_{predict_activities}{is_real}.pkl','rb'))
    results = pickle.load(open(f'vars/{experiment}/test_results_{maximize}_{predict_activities}.pkl', 'rb'))
    
    #Drop lines in which there's not a result
    results = [res for res in results if not isinstance(res[-1], str)]
    
    #Create and fill a dataset
    df_res = pd.DataFrame()
    df_res['result'] = [res[0] for res in results]
    df_res['len_trace_after_cut'] = [res[1] for res in results]
    df_res['#posssible_next_activity'] = [res[2] for res in results]
    df_res['score_reality'] = [res[3] for res in results]
    df_res['score_recommendation'] = [res[4] for res in results]
    df_res['history'] = [res[5][:-1] for res in results]
    df_res['suggested_activity'] = [res[6] for res in results]

    #Create artificial columns
    df_res['ratio_recommendation/reality'] = 1-(np.array(df_res['score_recommendation']))/(np.array(df_res['score_reality']))
    df_res['diff_reality_recommendation'] = np.array(df_res['score_reality'] - np.array(df_res['score_recommendation']))
    df_res['original_trace_kpi'] = [((res[5][int(res[2]):]).count(predict_activities)>0)*1 for res in results]
    df_res['my_score'] =  1- ((np.array(df_res['score_recommendation'])) / (np.array(df_res['score_reality'])))
    
    abs_score = round(df_res['result'].mean()*100, 2)
    poss_acts = df_res['#posssible_next_activity'].mean()
    avg_ratio = round(df_res['ratio_reality_recommendation'].mean(), 2)
    print(f'For cut {ratio_cut}:')
    print(f'The recommendation is correct in the {abs_score} % of cases')
    print(f'The The average ratio between score of real continuation and the recommended continuation is {avg_ratio}')
    print(f'The average number of actions possible is {poss_acts}\n')
    if save :
        pickle.dump(df_res, open(f'vars/{experiment}/df_res{ratio_cut}_{maximize}_{predict_activities}{is_real}.pkl','wb'))
    return df_res


def plot_qini_curve_comparison(maximize=False, is_real=True, predict_activities=str):
    
    #VA BENE COME È SCRITTO SOLO PER MAXIMIZE = FALSE
    curve_val_all = list()
    curve_val_real = list()
    random_class = list()
    maximize = 'max'*maximize + 'min'*(not maximize)
    is_real = '_real'*(is_real)
    
    
    for ratio_cut in range(4,10):
        ratio_cut = float(ratio_cut)
        df = pickle.load(open(f'vars/{experiment}/df_res{ratio_cut}_{maximize}_{predict_activities}{is_real}.pkl','rb'))        
        curve_val_all.append(df['result'].mean()*100)
        curve_val_real.append((sum(df['score_reality']>=df['score_recommendation'])/len(df))*100)
        random_class.append(100/(df['#posssible_next_activity'].mean()))
    
    #Plot the curve
    sns.set_style('darkgrid')    
    plt.title(label=f'Qini curve for minimization of {predict_activities} \n percent of improvements over all dataframe')
    plt.xlabel('Percentage of cut on the total length of trace')
    plt.ylabel('Score')
    sns.lineplot([ratio_cut*10 for ratio_cut in range(4,10)], curve_val_all, color='darkred')   
    sns.lineplot([ratio_cut*10 for ratio_cut in range(4,10)], curve_val_real, color='blue')   
    sns.lineplot([ratio_cut*10 for ratio_cut in range(4,10)], random_class, color='black', linestyle='--')   
    plt.legend(['accuracy compared with all possible ends', 'accuracy compared with real ends', '1/possible_activities'])   
    
    
    
def plot_qini_curve_score(maximize=False, is_real=True, predict_activities=str):
    
    #VA BENE COME È SCRITTO SOLO PER MAXIMIZE = FALSE
    percentage = list()
    random_class = list()
    maximize = 'max'*maximize + 'min'*(not maximize)
    is_real = '_real'*(is_real)
    
    for ratio_cut in range(4,10):
        ratio_cut = float(ratio_cut)
        df = pickle.load(open(f'vars/{experiment}/df_res{ratio_cut}_{maximize}_{predict_activities}{is_real}.pkl','rb'))       
        random_class.append((1/(df['#posssible_next_activity'].mean()))*100)
        percentage.append((df['ratio_reality_recommendation'].mean()-1)*100)
        
    sns.set_style('darkgrid')    
    plt.title(label=f'Qini curve for minimization of {predict_activities} \n average of improvement')
    plt.xlabel('Percentage of cut on the total length of trace')
    plt.ylabel('Score')
    sns.lineplot([ratio_cut*10 for ratio_cut in range(4,10)], random_class, color='black', linestyle='--')   
    sns.lineplot([ratio_cut*10 for ratio_cut in range(4,10)], percentage, color='darkgreen')   
    plt.legend(['1/possible_activities','Percent of improvement compared to reality'])   



def generate_table_values(maximize=False, is_real=True, predict_activities=str):
    
    tables = dict()
    maximize = 'max'*maximize + 'min'*(not maximize)
    is_real = '_real'*(is_real)
    
    for ratio_cut in range(4,10):
        ratio_cut = float(ratio_cut)
        df = pickle.load(open(f'vars/{experiment}/df_res{ratio_cut}_{maximize}_{predict_activities}{is_real}.pkl','rb'))  
        d = len(df) 
        kpi_positive = df[df['original_trace_kpi']>0]
        kpi_positive_ = sum((kpi_positive['score_reality'] - kpi_positive['score_recommendation']).values>=0)
        kpi_null = df[df['original_trace_kpi']==0]
        kpi_null_ = sum((kpi_null['score_reality'] - kpi_null['score_recommendation'])>=0)
        tables[str(ratio_cut)] = np.array([[kpi_null_, len(kpi_null)-kpi_null_, round(kpi_null_/d, 3), round((len(kpi_null)-kpi_null_)/d, 3)],
                                           [kpi_positive_, len(kpi_positive)-kpi_positive_, round(kpi_positive_/d, 3), round((len(kpi_positive)-kpi_positive_)/d, 3)]])
    return tables

def target_val_distribution_plot(ratio_cut=int, bw_method= None, maximize=False, is_real=True, predict_activities=str):
    
    ratio_cut = float(ratio_cut)
    maximize = 'max'*maximize + 'min'*(not maximize)
    is_real = '_real'*(is_real)
    df = pickle.load(open(f'vars/{experiment}/df_res{ratio_cut}_{maximize}_{predict_activities}{is_real}.pkl','rb'))  
    
    df = df_res[df_res['#posssible_next_activity']>1]
    # df = df[df['diff_reality_recommendation']!=0]    
                       
    sns.set_style('darkgrid')    
    fig, ax = plt.subplots(1,2)  
    # plt.suptitle(t = predict_activities+' Distribution of target values without considering 0-values')
    if predict_activities == 'Back-Office Adjustment Requested':
        ax[0].set_xlim(-0.05,.08)
        ax[1].set_ylim(-0.01, 0.05)
    # ax[0].set_xlim(-1e7, 1e7)
    # ax[1].set_ylim(0e7, 1e7)
    sns.kdeplot(data = df['score_recommendation'], ax = ax[0])
    sns.kdeplot(data = df['score_reality'], ax = ax[0])
    sns.kdeplot(data = df['score_recommendation'] - df['score_reality'] , ax = ax[0])
    ax[0].legend(['score_recommendation', 'score_reality' ,'recommendation - reality '])
    sns.boxplot(data = df[['score_recommendation', 'score_reality']], orient="l", palette="Set1", ax = ax[1], showmeans=True)#.xlabel(' ')
    
    sns.boxplot(data = df['ratio_recommendation/reality'], orient="l", palette="Set1", showmeans=True)#.xlabel(' ')
    sns.kdeplot(data = df['ratio_recommendation/reality'])#.xlabel(' ')
    plt.ylim(-1,1)
    plt.title('Improvement on time evaluated using the formula below')
    print('Means are ', np.mean(df['score_recommendation']),' ', np.mean(df['score_reality']))
    print('Stds are ', np.std(df['score_recommendation']),' ', np.std(df['score_reality']))
    print(stats.describe(np.array(df['score_recommendation'])))
    
    
def check_equal_values(ratio_cut=int, bw_method= None, maximize=False, is_real=True, predict_activities=str):

    #Check the effectiveness on the 0-diff values
    ratio_cut = float(ratio_cut)
    maximize = 'max'*maximize + 'min'*(not maximize)
    is_real = '_real'*(is_real)
    df = pickle.load(open(f'vars/{experiment}/df_res{ratio_cut}_{maximize}_{predict_activities}{is_real}.pkl','rb'))     
    
    #Filter on 0-diff values
    sns.set_style('darkgrid')   
    plt.ylim(-0.001, 0.04)
    df = df[df['diff_reality_recommendation']==0]
    plt.title(f'Distribution of original target values without differences in with/without \n reccomendation for {predict_activities}')
    # plt.ylim(0,0.06)
    plt.boxplot(df['score_reality'], showmeans=True)

#%% Run cases
if __name__ == "__main__":
    maximize = False
    is_real = False
    predict_activities = 'Back-Office Adjustment Requested' # 'Pending Liquidation Request'  # 
    print(f'Remember to set parameters in code, now they\' re : \nmaximize = {maximize} \nis_real {is_real} \nactivity: {predict_activities}')
    for ratio_cut in range(4,10):
        df_res = results_to_df(ratio_cut=ratio_cut, maximize=maximize, is_real=is_real, predict_activities=predict_activities, save=True)
    tables = generate_table_values(maximize=maximize, is_real=is_real, predict_activities=predict_activities)
    # plot_qini_curve_score(maximize=maximize, is_real=is_real, predict_activities=predict_activities)
    plot_qini_curve_comparison(maximize=maximize, is_real=is_real, predict_activities=predict_activities)
    target_val_distribution_plot(ratio_cut=5, bw_method=0.15, maximize=maximize, is_real=is_real, predict_activities=predict_activities)
    check_equal_values(ratio_cut = 5, bw_method = .15, maximize=False, is_real=is_real, predict_activities=predict_activities)

#%% 
'''
l = len(X_test.REQUEST_ID.unique())
l_n = len(df_test.REQUEST_ID.unique())

for idx in df_test.REQUEST_ID.unique():
    df = Counter(list(df_test[df_test['REQUEST_ID']==idx]['ACTIVITY']))
    if df['Back-Office Adjustment Requested']!=0:
        print(df['Back-Office Adjustment Requested']/l_n)
'''
# for idx in X_test['REQUEST_ID'].unique():
#     df = X_test[X_test['REQUEST_ID']==idx]
#     if len(set(df['y']))>1: 
#         print(idx)
    
# problema è su real, backoffice, ratio_cut4

#%%  THey were for lead time analysis
'''
results = pickle.load(open('vars/lead_time/test_results_min_None.pkl','rb'))

df_res[df_res['diff_reality_recommendation']!=0]['diff_reality_recommendation'].plot.kde()
np.std(df_res[df_res['diff_reality_recommendation']!=0]['diff_reality_recommendation'])
np.std(df_res['diff_reality_recommendation'])
sns.boxplot(df_res['diff_reality_recommendation'])
'''


