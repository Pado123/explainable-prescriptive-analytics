from dash import Dash, Input, Output, callback, dash_table, html, dcc, State, MATCH, ALL
from dash.dependencies import Input, Output, State
import pandas as pd
import io
import pickle
import numpy as np
import shap
import base64
import os
import datetime
import pm4py
shap.initjs()

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from utils import convert_to_csv, modify_filename, read_data


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
layout = go.Layout(
    yaxis=dict(
        range=[-1, 12]
    ),
    height=600
)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Set the experiment name
experiment_name = 'exp_time_VINST'
act_total = list(pd.read_csv('data/VINST cases incidents.csv')['ACTIVITY'].unique())

if not os.path.exists(f'gui_backup'):
    os.mkdir('gui_backup')

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
# Added (list(real_dict[key].values())[0]*.1) for showing also not so good cases
for key in real_dict.keys():
    if list(best_scores[key].values())[0] <= list(real_dict[key].values())[0] + (list(real_dict[key].values())[0]*.05):
        kpis_dict[key] = [list(best_scores[key].values())[0], list(real_dict[key].values())[0]]

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            df.to_csv('data/curr_df.csv')
            df.to_csv('gui_backup/curr_df.csv')
            pickle.dump(list(df.columns), open('gui_backup/col_list.pkl','wb'))

        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))

        elif '.xes' in filename:
            #Assume that it is a log
            pm4py.convert_to_dataframe(pm4py.read_xes(io.BytesIO(decoded))).to_csv(path_or_buf=(filename[:-4] + '.csv'), index=None)


    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return list(df.columns)

app.layout = html.Div([html.Div(children=[
    dbc.Row(
        [
            dbc.Col(dcc.Upload(
            id='upload-L_complete',
            children=html.Div([html.A('Select a log of complete traces for training the framework')]),
            style={
                'width': '80%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
        )
        ),
    dbc.Col(html.Button("Analize", id="df_analyzer", n_clicks=0))]),
    html.Div(id='output-L_complete'),
    dcc.Textarea(
            id='textarea-state-radioitems',
            value='Select the KPI you want to optimize',
            style={'width': '25%', 'height': 35, 'borderStyle':'None'},
            disabled='True',
        ),
    dcc.RadioItems([
        {'label': 'Total execution time   ', 'value': 'Decrease Time'},
        # {'label': 'Total cost of the procedure   ', 'value': 'Decrease cost'},  #TODO :Not implemented path
        {'label': 'Minimize activity occurrence   ', 'value': 'Minimize the activity occurrence'},
        {'label': 'Maximize activity occurrence   ', 'value': 'Maximize the activity occurrence'}
    ],
        labelStyle={'display': 'block'},
        id='radioitem_kpi',
    ),
    html.Div(id='unuseful_output'),
    html.Div(id='empy_out'),
    dcc.Textarea(
            id='textarea-state-radioitems-chooseKPI',
            value='Select the activity for which you want to optimize the occurrence',
            style={'width': '25%', 'height': 55, 'borderStyle':'None'},
            disabled='True',
        ),
    html.Div(id='dropdown-container', children=[]),
    html.Div(id='dropdown-container-output'),
    html.Div(id='dropdown_KPIactivity'),
    html.Div(id='Empty_out'),
    html.Button('Train!', id='submit-val', n_clicks=0),
    dcc.Upload(
                    id='upload-L_train',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select a log of running instances the framework')
                    ]),
                    style={
                        'width': '40%',
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
    dcc.Graph(
        figure={
            'data': [
                {'x': [int(kpis_dict[i][1])/3600 for i in kpis_dict.keys()], 'y': list(kpis_dict.keys()),
                 'type': 'bar', 'name': 'Actual value', 'orientation': 'h', 'marker': dict(color='rgba(130, 0, 0, 1)')},
                {'x': [int(kpis_dict[i][0])/3600 for i in kpis_dict.keys()], 'y': list(kpis_dict.keys()),
                 'type': 'bar', 'name': 'Following recommendation', 'orientation': 'h',
                 'marker': dict(color='rgba(0, 60, 0, 1)')},
            ],

            'layout': layout
        },
    ),
    dcc.Textarea(
        id='textarea-state-dropdown',
        value='Select the trace you want to optimize',
        style={'width': '100%', 'height': 50},
        disabled='True',
    ),
    dcc.Dropdown(list(kpis_dict.keys())[::-1], list(kpis_dict.keys())[0], id='dropdown_traces'),
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
    Output('unuseful_output', 'children'),
    Input('radioitem_kpi', 'value'),
)
def chosen_kpi(radiout):
    pickle.dump(radiout, open('gui_backup/chosen_kpi.pkl', 'wb'))
    return None

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

@app.callback(Output('output-L_complete', 'children'),
              Input('upload-L_complete', 'contents'),
              State('upload-L_complete', 'filename'),
              State('upload-L_complete', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]

@app.callback(
    Output('dropdown-container', 'children'),
    Input('df_analyzer', 'n_clicks'),
    State('dropdown-container', 'children'))
def display_dropdowns(n_clicks, children):
    children = list()
    if n_clicks >= 1:
        for i in range(4):
            if i == 0:name = 'Select the Case Id column'
            elif i == 1: name = 'Select the Activity column'
            elif i == 2: name = 'Select the Start Date column'
            elif i == 3: name = 'Select the ResourceName column (optional)'

            new_dropdown = dcc.Dropdown(
                        pickle.load(open('gui_backup/col_list.pkl','rb')),
                        placeholder=name,
                        id={
                            'type': 'columns-dropdown',
                            'index': i+1
                        },
                        style = {
                                    'width': '75%'
                                },
            )
            children.append(new_dropdown)
        return children

@app.callback(
    Output('dropdown-container-output', 'children'),
    Input({'type': 'filter-dropdown', 'index': ALL}, 'value')
)
def display_output(values):
    return html.Div([
        html.Div('Dropdown {} = {}'.format(i + 1, value))
        for (i, value) in enumerate(values)
    ])

@app.callback(
    Output('dropdown_KPIactivity', 'children'),
    Input({'type': 'columns-dropdown', 'index': ALL}, 'value')
)
def print_i(value):
    if (None not in value) and (value!=[]):
        pickle.dump(value[0], open('gui_backup/case_id_name.pkl','wb'))
        pickle.dump(value[1], open('gui_backup/act_name.pkl', 'wb'))
        pickle.dump(value[2], open('gui_backup/start_date_name.pkl', 'wb'))
        pickle.dump(value[3], open('gui_backup/resource_name.pkl', 'wb'))

        if pickle.load(open('gui_backup/chosen_kpi.pkl', 'rb')) in ['Minimize the activity occurrence',
                                                                    'Maximize the activity occurrence']:
            act_col = value[1]
            df = read_data(filename='gui_backup/curr_df.csv', start_time_col=value[2])
            act_list = list(df[act_col].unique())
            return html.Div([dcc.Dropdown(act_list, placeholder='Select the activity to optimize (optional)',
                                          id='Act_Chosen_dropdown', style={ 'width': '75%'},)])

        else: return None

@app.callback(
    Output('Empty_out', 'children'),
    Input('Act_Chosen_dropdown', 'value')
)
def save_activity(value):
    if value is not None:
        pickle.dump(value, open('gui_backup/activity_to_optimize.pkl','wb'))
    return None



if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True)
