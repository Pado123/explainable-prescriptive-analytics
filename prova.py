

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import utils

experiment_name, case_id_name, activity_name, pred_column = 'exp_time_dueeee', 'REQUEST_ID', 'ACTIVITY', 'remaining_time'
sns.set_style('darkgrid')
print('Starting import of model and data..')
# X_train, X_test, y_train, y_test = utils.import_vars(experiment_name='exp_time_explanationsdb', case_id_name=case_id_name)
X_train, X_test = pickle.load(open('X_train.pkl', 'rb')), pickle.load(open('X_test.pkl', 'rb'))
print('Importing completed...')
quantitative_vars, qualitative_trace_vars, qualitative_vars = utils.variable_type_analysiis(X_test, case_id_name, activity_name)
df_rec = utils.get_test(X_test, case_id_name).reset_index(drop=True)

def save_figures(df_rec, experiment_name, case_id_name, quantitative_vars, qualitative_vars, type='heatmap'):

    # Import the dataset
    df_rec = utils.get_test(X_test, case_id_name).reset_index(drop=True)
    expl_df = expl_df = pickle.load(open('expls_exp_time_dueeee/expl_df.pkl','rb'))
    # pd.read_csv('/home/padela/Scrivania/test_dir/expls_changed_history/prova.csv')
    df_rec = df_rec[expl_df.columns]

    for col in qualitative_vars + quantitative_vars[:-1]:
        del expl_df[col]
        del df_rec[col]

    # Find the correct values for the features associate with shapley values
    for idx in expl_df.index[:-1]:
        id = expl_df.loc[idx][case_id_name]
        shap_values = expl_df.loc[idx].values[1:]
        feature_values = df_rec[df_rec[case_id_name] == id]
        feature_values = feature_values.iloc[-1].values[1:]
        features = df_rec.columns[1:]
        features_plotable = [i+' ='+str(j) for (i,j) in zip(features,feature_values)]
        printable = pd.DataFrame()
        printable['feature'] = features_plotable
        printable[r'$\Delta(\sigma, z_{rec})$'] = shap_values/1000
        printable.sort_values(r'$\Delta(\sigma, z_{rec})$', ascending=False, inplace=True)
        if type == 'barplot':
            ax = printable.plot.barh(x='feature', y=r'$\Delta(\sigma, z_{rec})$', rot=0)
            plt.savefig(f'expls_{experiment_name}/plot_{experiment_name}_{idx}.png', bbox_inches='tight')
            plt.close()
        elif type == 'heatmap':
            printable.index = printable['feature']
            printable = printable[np.abs(printable[r'$\Delta(\sigma, z_{rec})$']) > .15]
            del printable['feature']
            plt.figure(figsize=(10, 15))
            printable = printable[printable.columns].astype(float)
            sns.heatmap(printable, annot=True, fmt=".1f", square=True, linewidths=.005)
            plt.savefig(f'expls_{experiment_name}/plot_{experiment_name}_{idx}.png', bbox_inches='tight')
            plt.close()




save_figures(df_rec, experiment_name, case_id_name, quantitative_vars, qualitative_vars)






















