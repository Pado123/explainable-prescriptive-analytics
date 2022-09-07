# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 16:44:12 2021

@author: padel
"""

#%%Define a parser for the txt file 
import pandas as pd
import ast
import pickle 
import os 
import numpy as np

os.chdir('/home/padela/Scrivania/PhD/Prescriptive-Analytics')
#%%
# def import_file(name=str):
my_file = open("/home/padela/Scrivania/PhD/Prescriptive-Analytics/parser_res.txt", "r")
content_list = my_file. readlines()
# return content_list

content_list = [line for line in content_list if line[0]=='S']
content_list = ast.literal_eval(content_list)

results = list()
for line in content_list:
    results.append([letter for letter in line if letter.isdigit()])

#%%
def to_df(results):
    df_res = pd.DataFrame()
    df_res['res'] = [int(results[i][0]) for i in range(len(results))]      
    df_res['cut'] = [results[i][1] for i in range(len(results))]   
    df_res['#acts_possible'] = [results[i][2] for i in range(len(results))] 
    df_res = df_res[df_res['res']!='not found in base dataset']
    df_res = df_res[df_res['#acts_possible']!='not found in base dataset']
    df_res['#acts_possible'] = [int(i) for i in df_res['#acts_possible']]
    df_res['cut'] = [int(i) for i in df_res['cut']]

    return df_res

# print(df_res[df_res['#acts_possible']>1]['res'].mean(), 1/df_res['#acts_possible'].mean())

# df_res.to_csv('vars/results_07_bo_')

# df = pickle.load(open('vars/test_results_07.pkl','rb'))

#%% Graph creation
# import pickle
# df_si = pickle.load(open('vars/results_final_richisi.pkl','rb'))
df_si = to_df(results)
df_si[df_si['#acts_possible']!=1].boxplot('res')
import seaborn as sns
sns.set_theme(style="darkgrid")

(df_si['res'].value_counts()/(len(df_si))).plot(kind='bar', title='Values for #acts>1')
print('None','&', round(df_si['res'].mean(), 3),'&', round(1/(df_si['#acts_possible'].mean()), 3),'&', len(df_si['res']), '\\\\')
print('$#acts>1$','&', round(df_si[df_si['#acts_possible']!=1]['res'].mean(), 3),'&', round(1/(df_si[df_si['#acts_possible']!=1]['#acts_possible'].mean()), 3),'&',
 len(df_si[df_si['#acts_possible']!=1]), '\\\\')
print('$#acts=2$','&', round(df_si[df_si['#acts_possible']==2]['res'].mean(), 3),'&', 1/2,'&', len(df_si[df_si['#acts_possible']==2]['res']), '\\\\')
print('$#acts=3$','&', round(df_si[df_si['#acts_possible']==3]['res'].mean(), 3),'&', 0.333,'&', len(df_si[df_si['#acts_possible']==3]['res']), '\\\\')
print('$#acts=4$','&', round(df_si[df_si['#acts_possible']==4]['res'].mean(), 3),'&', 0.25,'&', len(df_si[df_si['#acts_possible']==4]['res']), '\\\\')
#Confidence ok
# import ipdb; ipdb.set_trace()







