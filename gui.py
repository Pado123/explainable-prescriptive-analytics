from dash import Dash, Input, Output, callback, dash_table, html, dcc
import pandas as pd
import plotly.express as px
from skimage import io
import pickle
import numpy as np
import shutil
import os
import shap

shap.initjs()

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
plt.style.use('ggplot')

import dash_bootstrap_components as dbc
import plotly.graph_objects as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

df = pd.read_csv('https://git.io/Juf1t')
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# fig = plt.plot([i for i in range(20)], [i for i in range(20, 60, 2)])

# Create (eventually, clean) the gui backup folder
try:
    shutil.rmtree('gui_backup_data')
except:
    None
os.mkdir('gui_backup_data')

# Set the experiment name
experiment_name = 'exp_time_VINST'
act_total = list(pd.read_csv('data/VINST cases incidents.csv')['ACTIVITY'].unique())

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

kpis_dict = dict()

real_dict = dict(sorted(real_dict.items(), key=lambda x: list(x[1].values())[0]))
for key in real_dict.keys():
    if list(best_scores[key].values())[0]<=list(real_dict[key].values())[0]:
        kpis_dict[key] = [list(best_scores[key].values())[0], list(real_dict[key].values())[0]]


def read_variables_for_explain(experiment_name, trace_idx, act):
    explanations = pd.read_csv(f'explanations/{experiment_name}/{trace_idx}_{act}_expl_df.csv', index_col=0)
    idxs_chosen = pickle.load(open(f'explanations/{experiment_name}/{trace_idx}_{act}_idx_chosen.pkl', 'rb'))
    groundtruth_explanation = pd.read_csv(f'explanations/{experiment_name}/{trace_idx}_expl_df_gt.csv', index_col=0)
    last = pickle.load(open(f'explanations/{experiment_name}/{trace_idx}_last.pkl', 'rb'))
    return explanations, idxs_chosen, groundtruth_explanation, last

# ids_total = list(kpis_dict.keys())

app.layout = html.Div([html.Div(children=[
    dbc.Row([
            dbc.Col(dcc.Upload(
                id='upload-L_complete',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select a log of complete traces for training the framework')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=False
            )),
        dbc.Col(
            dcc.Upload(
                id='upload-L_run',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Running log for generating explanations')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=False
            ),
    )]),
    dcc.Textarea(
            id='textarea-state-radioitems',
            value='Select the KPI you want to optimize',
            style={'width': '25%', 'height': 35},
            disabled='True',
        ),
    dcc.RadioItems([
        {'label': 'time', 'value': 'Decrease Time'},
        {'label': 'cost', 'value': 'Decrease cost'},
        {'label': 'act', 'value': 'Minimize/Maximize the activity occurrence', 'disabled': False}
    ],
        'MTL',
        labelStyle={'display': 'inline-block'},
        id='radioitem_kpi',
    ),
    dcc.Graph(
        figure={
            'data': [
                {'x': [kpis_dict[i][1] for i in kpis_dict.keys()], 'y': list(kpis_dict.keys()),
                 'type': 'bar', 'name': 'Actual value', 'orientation': 'h', 'marker': dict(color='rgba(130, 0, 0, 1)')},
                {'x': [kpis_dict[i][0] for i in kpis_dict.keys()], 'y': list(kpis_dict.keys()),
                 'type': 'bar', 'name': 'Following recommendation', 'orientation': 'h',
                 'marker': dict(color='rgba(0, 60, 0, 1)')},
            ],

            'layout': {
                'title': 'Dashboard for explainable prescriptive analytics',
                'scrollZoom': False,
                # 'yaxis_rangeslider_visible': True,
                'yaxis_range': list(best_scores.keys())[:40]
            },
        },
    ),
    dcc.Textarea(
        id='textarea-state-dropdown',
        value='Select the trace you want to optimize',
        style={'width': '100%', 'height': 50},
        disabled='True',
    ),
    dcc.Dropdown(list(kpis_dict.keys()), list(kpis_dict.keys())[0], id='dropdown_traces'),
    html.Div(id='table_activities'),
    dcc.Textarea(
        id='textarea-state-activity-choice',
        value='Select the activity for which you desire to see the explanation',
        style={'width': '100%', 'height': 50},
        disabled='True',
    ),
    # html.Div(id='dropdown_traces_2'),
    dcc.Dropdown(act_total, 'No activity has been selected', id='dropdown_activities'),
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

    trace_idx = value
    act = act

    explanations = pd.read_csv(f'explanations/{experiment_name}/{trace_idx}_{act}_expl_df.csv', index_col=0)
    idxs_chosen = pickle.load(open(f'explanations/{experiment_name}/{trace_idx}_{act}_idx_chosen.pkl', 'rb'))
    groundtruth_explanation = pd.read_csv(f'explanations/{experiment_name}/{trace_idx}_expl_df_gt.csv', index_col=0)
    last = pickle.load(open(f'explanations/{experiment_name}/{trace_idx}_last.pkl', 'rb'))

    expl_df = {"Following Recommendation": [float(i) for i in explanations['0'][idxs_chosen].sort_values(ascending=False).values],
               "Actual Value": [float(i) for i in groundtruth_explanation['0'][idxs_chosen].sort_values(ascending=False).values]}

    last = last[idxs_chosen]
    feature_names = [str(i) for i in last.index]
    feature_values = [str(i) for i in last.values]

    index = [feature_names[i] + '=' + feature_values[i] for i in range(len(feature_values))]
    plot_df = pd.DataFrame(data=expl_df)
    plot_df.index = index

    try:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(expl_df['Following Recommendation']),
            y=list(index),
            name='Following recommendation',
            marker_color='darkgreen',orientation='h'
        ))
        fig.add_trace(go.Bar(
            x=list(expl_df['Actual Value']),
            y=list(index),
            name='Actual Value',
            marker_color='darkred', orientation='h'
        ))
        fig.update_layout(title_text=f'Explanations (change in Shapley values) following or not \n the recommendation for the activity {act} and the trace {value}')
        return dcc.Graph(figure=fig)
    except:
        return 'Please select one of the activities proposed above'


if __name__ == '__main__':
    app.run_server(debug=True)
