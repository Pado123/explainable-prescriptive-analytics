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
import next_act
import tqdm

import load_dataset
import explain_recsys
import next_act

shap.initjs()

from IO import read, folders, create_folders
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import warnings
warnings.filterwarnings("ignore")

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from utils import convert_to_csv, modify_filename, read_data
from load_dataset import prepare_dataset
import hash_maps
import utils
import shutil
from urllib.parse import quote as urlquote
import base64
from flask import Flask, send_from_directory

UPLOAD_DIRECTORY = os.getcwd() + '/gui_backup'
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


# Normally, Dash creates its own Flask server internally. By creating our own,
# we can create a route for downloading files directly:
server = Flask(__name__)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
expl_df = pd.DataFrame(columns=['Following Recommendation', 'Actual Value'])
index = ['#ACTIVITY=Wait - Implementation=0', 'Product=PROD24', 'Country=pl', 'ACTIVITY=Awaiting Assignment']
expl_df['Following Recommendation'] = [0.68, 0.43, -0.08, -0.43]
expl_df['Actual Value'] = [0.78, 0.45, 0.12, -0.29]
def get_layout(d):
    layout = go.Layout(
        xaxis=dict(
            range=[min([d[key][1] for key in list(d.keys())[:20]]), max([d[key][1] for key in list(d.keys())[:20]])]
        ),
        yaxis=dict(),
        height=600
    )
    return layout

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
@server.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)

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

def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)

app.layout = html.Div([html.Div(children=[
    dcc.Upload(id='upload-L_complete',
        children=html.Div([html.A('Select a log of complete traces for training the framework')]),
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
        multiple=True),
    html.Button("Analize", id="df_analyzer", n_clicks=0),
    html.Div(id='output-L_complete'),
    html.Div(id='output-L_run'),
    html.Div(id='unuseful_output'),
    html.Div(id='empy_out'),
    # dcc.Textarea(
    #         id='textarea-state-radioitems-chooseKPI',
    #         value='Select the activity for which you want to optimize the occurrence',
    #         style={'width': '25%', 'height': 55, 'borderStyle':'None'},
    #         disabled='True',
    #     ),
    dcc.Textarea(
        id='textarea-state-radioitems',
        value='Select the KPI you want to optimize',
        style={'width': '25%', 'height': 35, 'borderStyle': 'None'},
        disabled='True',
    ),
    dcc.RadioItems([
        {'label': 'Total execution time   ', 'value': 'Decrease Time'},
        # {'label': 'Total cost of the procedure   ', 'value': 'Decrease cost'},  #TODO :Not implemented path
        {'label': 'Minimize activity occurrence   ', 'value': 'Minimize the activity occurrence'},
        # {'label': 'Maximize activity occurrence   ', 'value': 'Maximize the activity occurrence'} #TODO :Not implemented path
    ],
        labelStyle={'display': 'block'},
        id='radioitem_kpi',
    ),
    html.Div(id='dropdown-container', children=[]),
    html.Div(id='dropdown-container-output'),
    html.Div(id='dropdown_KPIactivity'),
    html.Div(id='eo6'),
    html.Div(id='eo3'),
    html.Div(id='eo4'),
    html.Button('Train', id='submit-values_and_train', n_clicks=0),
    html.Div(id='eo5'),
    html.Div(id='eo2'),
    dcc.Upload(id='upload-L_run',
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
    html.Div(id='file-list'),
    html.Div(id='file-list2'),
    html.Button('Generate prediction', id='generate_prediction_button', n_clicks=0),
    html.Div(id='eo7'),
    html.Button('Show Prediction', id='show_pred_button', n_clicks=0),
    html.Div(id='figure_prediction'),
    dcc.Textarea(
        id='textarea-state-dropdown',
        value='Select the trace you want to optimize',
        style={'width': '100%', 'height': 50},
        disabled='True',
    ),
    html.Div(id='dropdown_traces_id'),
    html.Div(id='table_activities'),
    dcc.Textarea(
        id='textarea-state-activity-choice',
        value='Select the activity for which you desire to see the explanation',
        style={'width': '100%', 'height': 50},
        disabled='True',
    ),
    html.Div(id='dropdown_activities'),
    dcc.Textarea(
            id='textarea-state-expls',
            value='Select how much explanations you want to see',
            style={'width': '100%', 'height': 50},
            disabled='True',
        ),
    dcc.Slider(0, 10, 1,
                   value=3,
                   id='slider_expls'
        ),
    html.Div(id='figure_explanation'),
    html.Div(id='eo9'),
    html.Br()
])
]
)

@app.callback(
    Output('unuseful_output', 'children'),
    Input('radioitem_kpi', 'value'),
)
def chosen_kpi(radiout):
    pickle.dump(radiout, open('gui_backup/chosen_kpi.pkl', 'wb'))
    return None

@app.callback(
    Output('dropdown-container', 'children'),
    Input('df_analyzer', 'n_clicks'),
    State('dropdown-container', 'children'))
def display_dropdowns(n_clicks, children):
    children = list()
    if n_clicks >= 1:
        for i in range(4):
            if i == 0:name = 'Select the case id column'
            elif i == 1: name = 'Select the activity column'
            elif i == 2: name = 'Select the timestamp column'
            elif i == 3: name = 'Select the resource name column (optional)'

            new_dropdown = dbc.Row(
                        [dbc.Col(dcc.Textarea(value = name.split(' ')[2] + ' column',
                                              style={'width':'100%', 'margin': dict(
                                                  l=0,
                                                  r=0,
                                                  b=0,
                                                  t=0)},
                                              )),
                         dbc.Col(dcc.Dropdown(
                        pickle.load(open('gui_backup/col_list_train.pkl','rb')),
                        placeholder=name,
                        id={
                            'type': 'columns-dropdown',
                            'index': i+1
                        },
                        style = {
                                    'width': '100%'
                                },
            )),
            dbc.Col(dcc.Textarea(value=' ')),
            dbc.Col(dcc.Textarea(value=' '))])
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

            return dbc.Row(
                        [dbc.Col(dcc.Textarea(value = 'Activity to optimize column',
                                              style={'width':'100%', 'border': 'None', 'background-color': 'transparent', 'outline': 'None'}
                                              )),
                         dbc.Col(dcc.Dropdown(act_list, placeholder='Select the activity to optimize (optional)',
                                          style={'width': '100%'}, id='dropdown_KPIactivity')),
                         dbc.Col(dcc.Textarea(value=' ', )),
                         dbc.Col(dcc.Textarea(value=' '))])

        else: return None


@app.callback(
    Output('eo6', 'children'),
    Input('dropdown_KPIactivity', 'value')
)
def save_activity(value):
    if value is not None:
        pickle.dump(value, open('gui_backup/activity_to_optimize.pkl','wb'))
    return None

@app.callback(
    Output('eo2', 'children'),
    Input('submit-values_and_train', 'n_clicks'),
)
def train_predictor_and_hashmap(n_clicks):
    if n_clicks > 0:
        import time
        print(f'{time.time()}')
        filename = 'gui_backup/curr_df.csv'
        convert_to_csv(filename)
        filename = modify_filename(filename)
        date_format = "%Y-%m-%d %H:%M:%S"
        start_date_name = pickle.load(open('gui_backup/start_date_name.pkl', 'rb'))
        df = read_data(filename, start_date_name, date_format)
        print(df.shape)
        use_remaining_for_num_targets = None
        custom_attribute_column_name = None
        case_id_name = pickle.load(open('gui_backup/case_id_name.pkl', 'rb'))
        activity_name = pickle.load(open('gui_backup/act_name.pkl', 'rb'))
        pred_column = pickle.load(open('gui_backup/chosen_kpi.pkl', 'rb'))
        resource_column_name = pickle.load(open('gui_backup/resource_name.pkl', 'rb'))
        pred_column = 'independent_activity'*(pred_column=='Minimize the activity occurrence') + \
                      'remaining_time'*(pred_column == 'Decrease Time')
        experiment_name = 'Gui_experiment'
        if pred_column == 'independent_activity':
            predict_activities = [pickle.load(open('gui_backup/activity_to_optimize.pkl', 'rb'))]
        else :
            predict_activities = None
        end_date_name = None # try : pickle.load(open('gui_backup/end_date.pkl', 'rb')) except:
        role_column_name = None #TODO: implement a function which maps the possibility of having the variable
        override, pred_attributes, costs, working_time, lost_activities, retained_activities = True, None, None, \
                                                                                              None, None, None
        create_folders(folders, safe=override)
        shap = False
        #140 RIGHE AL SECONDO
        #240 RIGHE AL SECONDO
        prepare_dataset(df=df, case_id_name=case_id_name, activity_column_name=activity_name,
                        start_date_name=start_date_name, date_format=date_format, end_date_name=end_date_name,
                        pred_column=pred_column, mode="train", experiment_name=experiment_name, override=override,
                        pred_attributes=pred_attributes, costs=costs, working_times=working_time,
                        resource_column_name=resource_column_name, role_column_name=role_column_name,
                        use_remaining_for_num_targets=use_remaining_for_num_targets, predict_activities=predict_activities,
                        lost_activities=lost_activities, retained_activities=retained_activities,
                        custom_attribute_column_name=custom_attribute_column_name, shap=shap)

        # copy results as a backup
        fromDirectory = os.path.join(os.getcwd(), 'experiment_files')
        toDirectory = os.path.join(os.getcwd(), 'experiments', experiment_name)

        # copy results as a backup
        if os.path.exists(toDirectory):
            shutil.rmtree(toDirectory)
            shutil.copytree(fromDirectory, toDirectory)
        else:
            shutil.copytree(fromDirectory, toDirectory)
            print('Data and results saved')

        print('Starting import model and data..')
        if not os.path.exists(f'expls_{experiment_name}'):
            os.mkdir(f'expls_{experiment_name}')
            print('explanation folder created')

        if not os.path.exists(f'explanations/{experiment_name}'):
            os.mkdir(f'explanations/{experiment_name}')
            print('other explanation folder created')

        info = read(folders['model']['data_info'])
        X_train, X_test, y_train, y_test = utils.import_vars(experiment_name=experiment_name, case_id_name=case_id_name)
        model = utils.import_predictor(experiment_name=experiment_name, pred_column=pred_column)
        print('Importing completed...')

        X_train.to_csv('gui_backup/X_train.csv')
        print('Analyze variables...')
        quantitative_vars, qualitative_trace_vars, qualitative_vars = utils.variable_type_analysis(X_train,
                                                                                                   case_id_name,
                                                                                                   activity_name)
        pickle.dump(quantitative_vars, open(f'explanations/{experiment_name}/quantitative_vars.pkl', 'wb'))
        pickle.dump(qualitative_vars, open(f'explanations/{experiment_name}/qualitative_vars.pkl', 'wb'))
        pickle.dump(qualitative_trace_vars, open(f'explanations/{experiment_name}/qualitative_trace_vars.pkl', 'wb'))

        print('Variable analysis done')
        print(f'{time.time()}')
        return 'Training finished'

@app.callback(
    Output('eo5', 'children'),
    Input('submit-values_and_train', 'n_clicks'),
)
def print_expected_time(n_clicks):
    if n_clicks > 0:
        import time
        print(f'{time.time()}')
        filename = 'gui_backup/curr_df.csv'
        convert_to_csv(filename)
        filename = modify_filename(filename)
        date_format = "%Y-%m-%d %H:%M:%S"
        start_date_name = pickle.load(open('gui_backup/start_date_name.pkl', 'rb'))
        df = read_data(filename, start_date_name, date_format)
        seconds = df.shape[0]/210
        h  = seconds//3600
        min = (seconds%3600)//60
        print(df.shape)
        return f'The training will finish approximately in {h} hour and {min} min'

@app.callback(
    Output('eo7', 'children'),
    Input('generate_prediction_button', 'n_clicks'),
)
def print_expected_time_gen(n_clicks):
    if n_clicks > 0:
        import time
        print(f'{time.time()}')
        filename = 'gui_backup/dfrun_preprocessed.csv'
        convert_to_csv(filename)
        filename = modify_filename(filename)
        date_format = "%Y-%m-%d %H:%M:%S"
        start_date_name = pickle.load(open('gui_backup/start_date_name.pkl', 'rb'))
        try:
            df = read_data(filename, start_date_name, date_format)
        except:
            df = read_data(filename, 'time_from_start', date_format)
        seconds = df.shape[0] / 190
        h = seconds // 3600
        min = (seconds % 3600) // 60
        print(df.shape)
        return f'The generation will finish approximately in {h} hour and {min} min'

@app.callback(
    Output('eo4', 'children'),
    Input('generate_prediction_button', 'n_clicks'),
)
def generate_predictions(n_clicks):
    if n_clicks > 0:
        filename = 'gui_backup/run_df.csv'
        convert_to_csv(filename)
        filename = modify_filename(filename)
        date_format = "%Y-%m-%d %H:%M:%S"
        start_date_name = pickle.load(open('gui_backup/start_date_name.pkl', 'rb'))
        df_rec = read_data(filename, start_date_name, date_format)

        print(f'the shape of the submitted_data is {df_rec.shape}')
        use_remaining_for_num_targets = None
        custom_attribute_column_name = None
        case_id_name = pickle.load(open('gui_backup/case_id_name.pkl', 'rb'))
        activity_name = pickle.load(open('gui_backup/act_name.pkl', 'rb'))
        pred_column = pickle.load(open('gui_backup/chosen_kpi.pkl', 'rb'))
        resource_column_name = pickle.load(open('gui_backup/resource_name.pkl', 'rb'))
        if pred_column not in ['independent_activity','remaining_time'] :
            pred_column = 'independent_activity'*(pred_column=='Minimize the activity occurrence') + \
                          'remaining_time'*(pred_column == 'Decrease Time')
        pickle.dump(pred_column, open('gui_backup/pred_column.pkl','wb'))
        experiment_name = 'Gui_experiment'
        if pred_column == 'independent_activity':
            predict_activities = [pickle.load(open('gui_backup/activity_to_optimize.pkl', 'rb'))]
        else:
            predict_activities = None

        for i in df_rec.columns:
            if 'Unnamed' in i:
                del df_rec[i]
        end_date_name = None # try : pickle.load(open('gui_backup/end_date.pkl', 'rb')) except:
        role_column_name = None #TODO: implement a function which maps the possibility of having the variable
        override, pred_attributes, costs, working_time, lost_activities, retained_activities = True, None, None, \
                                                                                              None, None, None
        create_folders(folders, safe=override)
        shap = False
        df_rec = utils.read_data(filename='data/run_df.csv', start_time_col=start_date_name)
        print('Creating hash-map of possible next activities')
        X_train = pd.read_csv('gui_backup/X_train.csv').iloc[:,1:]
        outlier_thrs = 0
        traces_hash = hash_maps.fill_hashmap(X_train=X_train, case_id_name=case_id_name, activity_name=activity_name,
                                             thrs=outlier_thrs)
        print('Hash-map created')

        df_rec = load_dataset.preprocess_df(df=df_rec, case_id_name=case_id_name, activity_column_name=activity_name,
                        start_date_name=start_date_name, date_format=date_format, end_date_name=end_date_name,
                        pred_column=pred_column, mode="train", experiment_name=experiment_name, override=override,
                        pred_attributes=pred_attributes, costs=costs, working_times=working_time,
                        resource_column_name=resource_column_name, role_column_name=role_column_name,
                        use_remaining_for_num_targets=use_remaining_for_num_targets,
                        predict_activities=predict_activities, lost_activities=lost_activities,
                        retained_activities=retained_activities,
                        custom_attribute_column_name=custom_attribute_column_name, shap=shap)
        df_rec.to_csv('gui_backup/dfrun_preprocessed.csv')
        print('Running Data Imported')
        print(4*'\n')
        print('Starting generating recommendations')

        pickle.dump(traces_hash, open('gui_backup/transition_system.pkl', 'wb'))

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

        model = utils.import_predictor(experiment_name=experiment_name, pred_column=pred_column)
        for trace_idx in tqdm.tqdm(idx_list):
            trace = df_rec[df_rec[case_id_name] == trace_idx].reset_index(drop=True)
            trace = trace.reset_index(drop=True)  # trace.iloc[:, :-1].reset_index(drop=True)
            trace.rename(columns={'time_from_midnight': 'daytime'}, inplace=True)
            # trace = trace[list(model.feature_names_)]
            try:
                # take activity list
                acts = list(df_rec[df_rec[case_id_name] == trace_idx].reset_index(drop=True)[activity_name])

                # Remove the last (it has been added because of the evaluation)
                # trace = trace.iloc[:-1].reset_index(drop=True)
            except:
                import ipdb;
                ipdb.set_trace()

            try:
                next_activities, actual_prediciton = next_act.next_act_kpis(trace, traces_hash, model, pred_column,
                                                                            case_id_name,activity_name,
                                                                            quantitative_vars, qualitative_vars,
                                                                            encoding='aggr-hist')
                next_activities['kpi_rel'] = next_activities['kpi_rel'].abs()
            except:
                print('Next activity not found in transition system')
                continue

            try:
                rec_act = \
                next_activities[next_activities['kpi_rel'] == min(next_activities['kpi_rel'])]['Next_act'].values[
                    0]
                other_traces = [
                    next_activities[next_activities['kpi_rel'] != min(next_activities['kpi_rel'])]['Next_act'].values]
            except:
                try:
                    if len(next_activities) == 1:
                        print('No other traces to analyze')
                except:
                    print(trace_idx, 'check it')

            rec_dict[trace_idx] = {i: j for i, j in zip(next_activities['Next_act'], next_activities['kpi_rel'])}
            real_dict[trace_idx] = {acts[-1]: actual_prediciton}
        rec_dict =  {str(A): N for (A, N) in [x for x in rec_dict.items()]}
        real_dict = {str(A): N for (A, N) in [x for x in real_dict.items()]}
        pickle.dump(rec_dict, open(f'recommendations/{experiment_name}/rec_dict.pkl', 'wb'))
        pickle.dump(real_dict, open(f'recommendations/{experiment_name}/real_dict.pkl', 'wb'))
        print('Prediction generation completed')

@app.callback(
    Output('figure_prediction', 'children'),
    Input('show_pred_button', 'n_clicks')
)
def create_expl_fig(n_clicks):
    if n_clicks > 0:
        experiment_name = 'Gui_experiment'

        # Read the dictionaries with the scores
        rec_dict = pickle.load(open(f'recommendations/{experiment_name}/rec_dict.pkl', 'rb'))
        real_dict = pickle.load(open(f'recommendations/{experiment_name}/real_dict.pkl', 'rb'))
        pred_column = pickle.load(open('gui_backup/chosen_kpi.pkl', 'rb'))
        if pred_column in ['Minimize the activity occurrence', 'independent_activity']:
            res_val = 1
        else:
            res_val = 60

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
            if list(best_scores[key].values())[0] <= list(real_dict[key].values())[0] + (
                    list(real_dict[key].values())[0] * .05):
                kpis_dict[key] = [list(best_scores[key].values())[0], list(real_dict[key].values())[0]]

        kpis_dict = {str(A): N for (A, N) in [x for x in kpis_dict.items()]}
        kpis_dict = dict(sorted(kpis_dict.items(), key=lambda item: item[1][1]))
        layout = get_layout(kpis_dict)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list([kpis_dict[k][0] for k in kpis_dict.keys()]),
            y=list(kpis_dict.keys()),
            name='Following recommendation',
            orientation='h',
            marker_color='darkgreen',
        ))
        fig.add_trace(go.Bar(
            x=[kpis_dict[k][1] for k in kpis_dict.keys()],
            y=list(kpis_dict.keys()),
            name='Actual Value',
            orientation='h',
            marker_color='darkred',
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          xaxis=dict(range=[0,max([kpis_dict[k][1] for k in kpis_dict.keys()])]),
                          yaxis=dict(range=[len(kpis_dict)-10, len(kpis_dict)]))
        # fig.update_style(width='60%')
        return dcc.Graph(figure=fig)

    else :
        return None



@app.callback(
    Output('dropdown_traces_id', 'children'),
    Input('show_pred_button', 'n_clicks')
)
def show_trace_dropdown(n_clicks):
    if n_clicks > 0:
        experiment_name = 'Gui_experiment'
        real_dict = pickle.load(open(f'recommendations/{experiment_name}/real_dict.pkl', 'rb'))
        return dcc.Dropdown(list(real_dict.keys())[::-1], list(real_dict.keys())[0], id='dropdown_traces_id')

@app.callback(
    Output('table_activities', 'children'),
    Input('dropdown_traces_id', 'value')
)
def save_result(value):
    if value is not None:
        try:
            pickle.dump(value, open('gui_backup/chosen_trace.pkl', 'wb'))

            act_name = pickle.load(open('gui_backup/act_name.pkl', 'rb'))
            acts = list(pd.read_csv('gui_backup/X_train.csv')[act_name].unique())
            experiment_name = 'Gui_experiment'
            best_3_dict = pickle.load(open(f'recommendations/{experiment_name}/rec_dict.pkl', 'rb'))
            # value = pickle.load(open('gui_backup/chosen_trace.pkl', 'rb'))
            dict_id = best_3_dict[value]
            dict_id = dict(sorted(dict_id.items(), key=lambda x: x[1]))
            dict_id = {A: N for (A, N) in [x for x in dict_id.items()][:3]}
            pickle.dump(dict_id, open('gui_backup/best_act_dict.pkl','wb'))
            df = pd.DataFrame(columns=['Next Activity', 'Expected KPI'])

            for i in range(len(dict_id.keys())):
                df.loc[i] = np.array([list(dict_id.keys())[i], dict_id[list(dict_id.keys())[i]]/60])
            return dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns])
        except :
            return "Running not executed, please insert a running log a generate recommendations"



@app.callback(
    Output("file-list", "children"),
    [Input("upload-L_run", "filename"), Input("upload-L_run", "contents")],
)
def update_output(uploaded_filenames, uploaded_file_contents):
    """Save uploaded files and regenerate the file list."""
    if type(uploaded_filenames) == str:
        uploaded_filenames = [uploaded_filenames]
    if type(uploaded_file_contents) == str:
        uploaded_file_contents = [uploaded_file_contents]

    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            save_file(name, data)

    if uploaded_filenames is not None:
        if '.csv' in uploaded_filenames[0]:
            df = pd.read_csv('gui_backup/'+ uploaded_filenames[0])
            df.to_csv('data/run_df.csv', index=None)
            df.to_csv('gui_backup/run_df.csv', index=None)
            pickle.dump(list(df.columns), open('gui_backup/col_list_train.pkl', 'wb'))
        if '.xes' in uploaded_filenames[0]:
            pm4py.convert_to_dataframe(pm4py.read_xes('gui_backup/'+uploaded_filenames[0])).to_csv(path_or_buf=('gui_backup/'+ uploaded_filenames[0][:-4] + '.csv'), index=None)
            df = pd.read_csv('gui_backup/'+ uploaded_filenames[0][:-4] + '.csv')
            df.to_csv('data/run_df.csv', index=None)
            df.to_csv('gui_backup/run_df.csv', index=None)
            pickle.dump(list(df.columns), open('gui_backup/col_list_train.pkl', 'wb'))
            print('Conversion ok')

@app.callback(
    Output("file-list2", "children"),
    [Input("upload-L_complete", "filename"), Input("upload-L_complete", "contents")],
)
def update_output(uploaded_filenames, uploaded_file_contents):
    """Save uploaded files and regenerate the file list."""
    if type(uploaded_filenames) == str:
        uploaded_filenames = [uploaded_filenames]
    if type(uploaded_file_contents) == str:
        uploaded_file_contents = [uploaded_file_contents]

    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            save_file(name, data)

    if uploaded_filenames is not None:
        if '.csv' in uploaded_filenames[0]:
            df = pd.read_csv('gui_backup/'+ uploaded_filenames[0])
            df.to_csv('data/curr_df.csv', index=None)
            df.to_csv('gui_backup/curr_df.csv', index=None)
            pickle.dump(list(df.columns), open('gui_backup/col_list_train.pkl', 'wb'))
        if '.xes' in uploaded_filenames[0]:
            pm4py.convert_to_dataframe(pm4py.read_xes('gui_backup/'+uploaded_filenames[0])).to_csv(path_or_buf=('gui_backup/'+ uploaded_filenames[0][:-4] + '.csv'), index=None)
            df = pd.read_csv('gui_backup/'+ uploaded_filenames[0][:-4] + '.csv')
            df.to_csv('data/curr_df.csv', index=None)
            df.to_csv('gui_backup/curr_df.csv', index=None)
            pickle.dump(list(df.columns), open('gui_backup/col_list_train.pkl', 'wb'))
            print('Conversion ok')
@app.callback(
    Output('dropdown_activities', 'children'),
    Input('show_pred_button', 'n_clicks')
)
def show_trace_dropdown(n_clicks):
    if n_clicks > 0:
        activities = pickle.load(open('gui_backup/act_name.pkl','rb'))
        act_list = pd.read_csv('data/run_df.csv')[activities].unique()
        return dcc.Dropdown(act_list, 'Select one of the 3 activities proposed above', id='dropdown_activities')


@app.callback(
    Output('figure_explanation', 'children'),
    Input('dropdown_activities', 'value')
)
def create_expl_fig(value):
    if value is not None:
        try :
            activity_name = pickle.load(open('gui_backup/act_name.pkl', 'rb'))
            df_run = pd.read_csv('gui_backup/dfrun_preprocessed.csv')
            case_id_name = pickle.load(open('gui_backup/case_id_name.pkl', 'rb'))
            df_run[case_id_name] = [str(i) for i in df_run[case_id_name]]
            if value in set(df_run[activity_name]):
                trace_idx = pickle.load(open('gui_backup/chosen_trace.pkl','rb'))
                act = value

                for i in df_run.columns:
                    if 'Unnamed' in i:
                        del df_run[i]
                case_id_name = pickle.load(open('gui_backup/case_id_name.pkl', 'rb'))
                experiment_name = 'Gui_experiment'
                X_test = df_run.copy()
                traces_hash = pickle.load(open('gui_backup/transition_system.pkl', 'rb'))


                trace_exp = df_run[df_run[case_id_name]==trace_idx].copy()
                trace = df_run[df_run[case_id_name]==trace_idx].iloc[:,1:].copy()
                trace_exp.rename(columns={'time_from_midnight': 'daytime'}, inplace=True)
                trace.rename(columns={'time_from_midnight': 'daytime'}, inplace=True)

                # start = time.time()
                quantitative_vars = pickle.load(open(f'explanations/{experiment_name}/quantitative_vars.pkl', 'rb'))
                qualitative_vars = pickle.load(open(f'explanations/{experiment_name}/qualitative_vars.pkl', 'rb'))
                pred_column = pickle.load(open('gui_backup/pred_column.pkl','rb'))
                model = utils.import_predictor(experiment_name=experiment_name, pred_column=pred_column)
                rec_dict = pickle.load(open(f'recommendations/{experiment_name}/rec_dict.pkl', 'rb'))[trace_idx]
                rec_dict = dict(sorted(rec_dict.items(), key=lambda x: x[1]))
                rec_dict = {A: N for (A, N) in [x for x in rec_dict.items()][:3]}


                for var in (set(quantitative_vars).union(qualitative_vars)):
                    trace_exp[var] = "none"
                groundtruth_explanation = explain_recsys.evaluate_shap_vals(trace_exp, model, df_run, case_id_name)
                groundtruth_explanation = [a for a in groundtruth_explanation]
                groundtruth_explanation = [trace_idx] + groundtruth_explanation
                groundtruth_explanation = pd.Series(groundtruth_explanation, index=[i for i in df_run.columns if i != 'y'])

                # Save also groundtruth explanations
                groundtruth_explanation.to_csv(f'explanations/{experiment_name}/{trace_idx}_expl_df_gt.csv')
                groundtruth_explanation.drop([case_id_name] + [i for i in (set(quantitative_vars).union(qualitative_vars))],
                                             inplace=True)

                # stampa l'ultima riga di trace normale
                trace_exp.iloc[-1].to_csv(f'explanations/{experiment_name}/{trace_idx}_expl_df_values.csv')
                last = trace_exp.iloc[-1].copy().drop([case_id_name] + [i for i in (set(quantitative_vars).union(qualitative_vars))])
                next_activities = list(rec_dict.keys())  # TODO: Note that is only optimized for minimizing a KPI


                trace_exp.reset_index(drop=True, inplace=True)
                trace_exp.loc[len(trace_exp) - 1, activity_name] = act
                for i in trace_exp.columns:
                    if 'Unnamed' in i:
                        del df_run[i]
                explanations = explain_recsys.evaluate_shap_vals(trace_exp, model, X_test, case_id_name)
                explanations = [a for a in explanations]
                explanations = [trace_idx] + explanations
                explanations = pd.Series(explanations, index=[i for i in trace_exp.columns if i!='y'])
                explanations.to_csv(f'explanations/{experiment_name}/{trace_idx}_{act}_expl_df.csv')

                #Take the best-4 deltas
                explanations.drop([case_id_name]+[i for i in (set(quantitative_vars).union(qualitative_vars))], inplace=True)
                groundtruth_explanation = groundtruth_explanation[list(explanations.index)]
                deltas_expls = groundtruth_explanation - explanations
                deltas_expls.sort_values(ascending=False, inplace=True)
                idxs_chosen = deltas_expls.index[:4]

                pickle.dump(idxs_chosen, open(f'explanations/{experiment_name}/{trace_idx}_{act}_idx_chosen.pkl', 'wb'))
                pickle.dump(last, open(f'explanations/{experiment_name}/{trace_idx}_last.pkl', 'wb'))
                explain_recsys.plot_explanations_recs(groundtruth_explanation, explanations, idxs_chosen, last, experiment_name, trace_idx, act)


                trace_idx = pickle.load(open('gui_backup/chosen_trace.pkl', 'rb'))
                act = value
                experiment_name = 'Gui_experiment'

                print('Generating explanations..')

                explanations = pd.read_csv(f'explanations/{experiment_name}/{trace_idx}_{act}_expl_df.csv', index_col=0)
                idxs_chosen = pickle.load(open(f'explanations/{experiment_name}/{trace_idx}_{act}_idx_chosen.pkl', 'rb'))
                groundtruth_explanation = pd.read_csv(f'explanations/{experiment_name}/{trace_idx}_expl_df_gt.csv', index_col=0)
                last = pickle.load(open(f'explanations/{experiment_name}/{trace_idx}_last.pkl', 'rb'))

                expl_df = {"Following Recommendation": [float(i)/60 for i in explanations['0'][idxs_chosen].sort_values(ascending=False).values],
                           "Actual Value": [float(i)/60 for i in groundtruth_explanation['0'][idxs_chosen].sort_values(ascending=False).values]}

                last = last[idxs_chosen]
                feature_names = [str(i) for i in last.index]
                feature_values = [str(i) for i in last.values]

                index = [feature_names[i] + '=' + feature_values[i] for i in range(len(feature_values))]
                plot_df = pd.DataFrame(data=expl_df)
                plot_df.index = index


        except:
            print('Explanations still not present')

        try:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(expl_df['Following Recommendation']),
                y=list(index),
                name='Following recommendation',
                marker_color='darkgreen', orientation='h'
            ))
            fig.add_trace(go.Bar(
                x=list(expl_df['Actual Value']),
                y=list(index),
                name='Actual Value',
                marker_color='darkred', orientation='h'
            ))
            fig.update_layout(title_text=f'Explanations (change in Shapley values) following or not\n the recommendation',
                              showlegend=True)
            # fig.update_style(width='60%')
            return dcc.Graph(figure=fig)
        except:
            return 'Please select one of the activities proposed above'

@app.callback(
    Output('eo9', 'children'),
    Input('slider_expls', 'value')
)
def create_expl_fig(value):
    pickle.dump(value, open('gui_backup/num_expls', 'wb'))
    return None

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True)

