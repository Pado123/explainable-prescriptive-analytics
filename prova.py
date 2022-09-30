import numpy as np
import pandas as pd
import plotly
import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# df = pd.read_csv('DATA.csv')
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    html.Div(
        dcc.Graph(id='graph',
                  style={'height': 400, 'width': 800}
                  ),
    ),
    html.Div([
                html.Label(['Choose a graph:'],style={'font-weight': 'bold'}),
                dcc.RadioItems(
                    id='radio',
                    options=[
                             {'label': 'graph1', 'value': 'graph1'},
                             {'label': 'graph2', 'value': 'graph2'},
                    ],
                    value='age',
                    style={"width": "60%"}
                ),
        ])

])


# ---------------------------------------------------------------
@app.callback(
    Output('graph', 'figure'),
    [Input(component_id='radio', component_property='value')]
)
def build_graph(value):
    if value == 'graph1':
        return px.scatter(
            df,
            x="D",
            y="B",
            color="C",
            size_max=20,  # differentiate markers by color
            opacity=0.90)

    else:
        return px.scatter(
            df,
            x="A",
            y="B",
            color="C",
            size_max=20,  # differentiate markers by color
            opacity=0.90)

if __name__ == '__main__':
    app.run_server(debug=True)
