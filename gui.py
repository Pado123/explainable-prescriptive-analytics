from dash import Dash, Input, Output, callback, dash_table, html, dcc
import pandas as pd
import plotly.express as px
from skimage import io
import pickle
import numpy as np
import shutil
import os
import matplotlib.pyplot as plt


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

df = pd.read_csv('https://git.io/Juf1t')
app = Dash(__name__, external_stylesheets=external_stylesheets)
# fig = plt.plot([i for i in range(20)], [i for i in range(20, 60, 2)])

#Create (eventually, clean) the gui backup folder
try :
    shutil.rmtree('gui_backup_data')
except: None
os.mkdir('gui_backup_data')

#Set the experiment name (TODO: eventually fix and make it interactive)
experiment_name = 'exp_time_VINST'
ids_total = list(pd.read_csv('data/VINST cases incidents.csv')['ACTIVITY'].unique())

# Read the dictionaries with the scores
rec_dict = pickle.load(open(f'recommendations/{experiment_name}/rec_dict.pkl', 'rb'))
real_dict = pickle.load(open(f'recommendations/{experiment_name}/real_dict.pkl', 'rb'))

# Read the variables' types
quantitative_vars = pickle.load(open(f'explanations/{experiment_name}/quantitative_vars.pkl', 'rb'))
qualitative_vars = pickle.load(open(f'explanations/{experiment_name}/qualitative_vars.pkl', 'rb'))

# Make a dictionary with only the best scores
best_scores = dict()
for key in rec_dict.keys():
    best_scores[key] = {min(rec_dict[key], key=rec_dict[key].get): min(rec_dict[key].values())}

# Make a dictionary with only the 3-best activities
best_3_dict = dict()
for key in rec_dict.keys():
    best_3_dict[key] = dict(sorted(rec_dict[key].items(), key=lambda item: item[1], reverse=False))
    best_3_dict[key] = {k: best_3_dict[key][k] for k in list(best_3_dict[key])[:3]}



app.layout = html.Div([html.Div(children=[
    dcc.Graph(
        figure={
            'data': [
                {'x': [list(i.values())[0] for i in list(real_dict.values())], 'y': list(best_scores.keys()),
                 'type': 'bar', 'name': 'Actual value', 'orientation': 'h', 'marker': dict(color='rgba(130, 0, 0, 1)')},
                {'x': [list(i.values())[0] for i in list(best_scores.values())], 'y': list(best_scores.keys()),
                 'type': 'bar', 'name': 'Following recommendation', 'orientation': 'h',
                 'marker': dict(color='rgba(0, 130, 0, 1)')},
            ],
            'layout': {
                'title': 'First dashboard for explainable prescriptive analytics',
                'scrollZoom': True},
        },
    ),
    dcc.Textarea(
        id='textarea-state-dropdown',
        value='Select the trace you want to optimize',
        style={'width': '100%', 'height': 50},
        disabled='True',
    ),
    dcc.Dropdown(list(real_dict.keys()), list(real_dict.keys())[0], id='dropdown_traces'),
    html.Div(id='table_activities'),
    dcc.Textarea(
        id='textarea-state-activity-choice',
        value='Select the activity for which you desire to see the explanation',
        style={'width': '100%', 'height': 50},
        disabled='True',
    ),
    # html.Div(id='dropdown_traces_2'),
    dcc.Dropdown(ids_total, 'No activity has been selected', id='dropdown_activities'),
    html.Div(id='figure_explanation'),
    html.Br()
])
]
)


@app.callback(
    Output('table_activities', 'children'),
    Input('dropdown_traces', 'value')
)
def render_content(value):
    dict_id = best_3_dict[value]
    df = pd.DataFrame(columns=['Next Activity', 'Expected KPI'])
    for i in range(len(dict_id.keys())):
        df.loc[i] = np.array([list(dict_id.keys())[i], dict_id[list(dict_id.keys())[i]]])
    return dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns])


@app.callback(
    Output('figure_explanation', 'children'),
    Input('dropdown_activities', 'value'), Input('dropdown_traces', 'value')
)
def create_expl_fig(act, value):
    try :
        img = io.imread(f'explanations/{experiment_name}/{value}_{act}.png')
        fig = px.imshow(img)
        return dcc.Graph(figure=fig,
                         style={'width': '220', 'height': '90'})
    except :
        return 'Please select one of the activities proposed above'



if __name__ == '__main__':
    app.run_server(debug=True)
