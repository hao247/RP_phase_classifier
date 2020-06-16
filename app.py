import base64
import io
import os
from threading import Timer

import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import numpy as np
import pandas as pd
from dash import Dash
from dash.dependencies import Input, Output, State

import utils.gen_figs as plt
from models import nn_classifier, rf_classifier
from params import colors, labels
from utils.helpers import find_peaks, normalization, open_browser

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server


app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.H1([
                    'RP Phase Classifier', 
                    html.Br()
                ], style={'margin': 'auto', 
                          'textAlign': 'center', 
                          'margin-top': '5%'}),
                html.H3([
                    'An app for quickly predicting the type of Ruddlesden-Popper phase from \
                        polycrystalline X-ray diffraction (XRD) spectrum.',
                    html.Br(), html.Br(),
                    'Instruction:', html.Br(),
                    html.Ol([
                        html.Li('Simply upload your csv file.'),
                        html.Li('Check the predictions from both Random forest model and \
                                neutal network model at the bottom. :)'),
                    ])
                ],style={'font-size': '18px', 
                         'textAlign': 'left', 
                         'margin-left': '20px',
                         'margin-top': '10px', 
                         'margin-bottom': '10px'}),
            ]),
            html.Div(style={'boader': '0',
                            'width': '95%',
                            'height': '2px', 
                            'background': '#333',
                            'background-image': 'linear-gradient(to right, #F9F9F9, #626262, #F9F9F9',
                            'margin-top': '10%'}),
            dcc.Upload(
                id='upload_data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]), 
                style={
                    'width': '90%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': 'auto',
                    'margin-top': '30px',
                    'margin-bottom': '30px'
                },
                multiple=False
            ),
            html.Div(id='uploaded_data', 
                     children='', 
                     style={'margin': 'auto', 
                            'textAlign': 'center'}),
            html.Div(id='store_data', 
                     children='', 
                     style={'display': 'none'}),
            html.Div(style={'boader': '0', 
                            'width': '95%', 
                            'height': '2px', 
                            'background': '#333',
                            'background-image': 'linear-gradient(to right, #F9F9F9, #626262, #F9F9F9'}),
        ], style={'width': '24%',
                  'height': '98%',
                  'vertical-align': 'top',
                  'display': 'inline-block',
                  'overflow': 'auto',
                  'font-size': '16px',
                  'margin-left': '30px',
                  'margin-right': '10px',
                  'margin-top': '1%',
                  'margin-bottom': '0px',
                  'background-color': colors['panel'],
                  'border-radius': '10px',
                  'box-shadow': '5px 5px 5px #888888'}),
        html.Div([
            html.Div([
                html.Div([
                    html.H3('Highest peaks: ', 
                            style = {'textAlign': 'center', 
                                    'margin-left': '20px',
                                    'margin-top': '30px', 
                                    'margin-bottom': '10px'}),
                    html.Div(id='highest_peaks', 
                             style={'display':'indent-block', 
                                    'margin': 'auto',
                                    'width': '90%',
                                    'margin-top': '15%'}
                )], style={'background-color': colors['panel'],
                           'width': '27%',
                           'height': '100%',
                           'margin-top': '0%',
                           'margin-bottom': '0%',
                           'margin-left': '0%',
                           'margin-right': '1%',
                           'display': 'inline-block',
                           'border-radius': '10px',
                           'box-shadow': '5px 5px 5px #888888',
                           'overflow': 'auto'}),
                html.Div([
                    html.H3('XRD spectrum'),
                    html.Div(id='plot_raw_data',
                             style={'width': '95%',
                                    'height': '90%',
                                    'margin-top': '2%',
                                    'margin-left': '2%'})
                ], style={'margin': 'auto',
                            'background-color': colors['panel'],
                            'width': '70%',
                            'height': '100%',
                            'margin-top': '0%',
                            'margin-bottom': '0%',
                            'margin-left': '0%',
                            'margin-right': '0%',
                            'display': 'inline-block',
                            'border-radius': '10px',
                            'box-shadow': '5px 5px 5px #888888'}),
            ], id='raw_data',
               style={'background-color': colors['background'],
                      'width': '98%',
                      'height': '50%',
                      'margin-top': '0%',
                      'margin-bottom': '1%',
                      'margin-left': '1%',
                      'margin-right': '0%',
                      'align-items': 'center',
                      'display': 'flex'}),
            html.Div([
                html.Div([
                    html.H3('Random forest'),
                    html.Div(id='rf_probabilities', 
                             style={'width': '80%', 
                                    'margin': 'auto',
                                    'margin-top': '5%'}),
                    html.Div(id='rf_prediction', 
                             style={'width': '80%', 
                                    'margin': 'auto',
                                    'margin-top': '5%'})
                ], style={'margin': 'auto',
                          'background-color': colors['panel'],
                          'width': '48%',
                          'height': '100%',
                          'margin-top': '0%',
                          'margin-bottom': '0%',
                          'margin-left': '0%',
                          'margin-right': '2%',
                          'display': 'inline-block',
                          'border-radius': '10px',
                          'box-shadow': '5px 5px 5px #888888',
                          'overflow': 'auto'}),
                html.Div([
                    html.H3('Neural network'),
                    html.Div(id='nn_probabilities', 
                             style={'width': '80%', 
                                    'margin': 'auto',
                                    'margin-top': '5%'}),
                    html.Div(id='nn_prediction', 
                             style={'width': '80%', 
                                    'margin': 'auto',
                                    'margin-top': '5%'})
                ], style={'margin': 'auto',
                          'background-color': colors['panel'],
                          'width': '48%',
                          'height': '100%',
                          'margin-top': '0%',
                          'margin-bottom': '0%',
                          'margin-left': '0%',
                          'margin-right': '0%',
                          'display': 'inline-block',
                          'vertical-align': 'top',
                          'border-radius': '10px',
                          'box-shadow': '5px 5px 5px #888888',
                          'overflow': 'auto'}),
            ], id='results',
                style={'background-color': colors['background'],
                       'width': '98%',
                       'height': '48%',
                       'margin-top': '0%',
                       'margin-bottom': '0%',
                       'margin-left': '0%',
                       'margin-right': '0%',
                       # 'display': 'inline-block',
                       'align-items': 'center',
                       'textAlign': 'center'}),
        ], style={'width': '70%', 
                  'height': '98%', 
                  'overflow': 'auto',
                  'font-size': '14px', 
                  'vertical-align': 'top',
                  'display': 'inline-block', 
                  'textAlign': 'center',
                  'overflow': 'auto',
                  'margin-left': '0%', 
                  'margin-right': '0%',
                  'margin-top': '1%', 
                  'margin-bottom': '0%',
                  'backgroundColor': colors['background'],
                  'border-radius': '10px'}),
    ], style={'width': '100hh', 
              'height': '96vh'}),
    html.P('hao.zheng(at)colorado.edu 2019', 
           style={'height': '3vh', 
                  'textAlign': 'center',
                  'margin-top': '0px'})
], style={'width': '100hh', 
          'height': '100vh', 
          'background-color': colors['background'], 
          'color': colors['font']})


@app.callback([Output('store_data', 'children'),
               Output('uploaded_data', 'children')],
              [Input('upload_data', 'contents')],
              [State('upload_data', 'filename')])
def input_data(contents, fname):
    """ Import the data from uploaded file and store it on the web page

    Args:
        contents (binary): binary content loaded from file
        fname (str): file name

    Returns:
        list: list containing
        1. jsonified dataframe for storing on the web page
        2. a message string indicating the uploading status
    """    
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), index_col=0)
        columns = [{'label': col, 'value': col} for col in list(df.columns)]
        message = f'{fname} loaded successfully.'
        return df.to_json(), message
    else:
        return '', ''


@app.callback(Output('plot_raw_data', 'children'),
              [Input('store_data', 'children')])
def plot_raw_data(df_json):
    """ plot a spectum figure for imported data

    Args:
        df_json (json): json string stored on the web page

    Returns:
        list: a list containing the spectrum figure
    """    
    if df_json:
        df = pd.read_json(df_json)
        return [dcc.Graph(
            figure=plt.gen_scatter(df['2theta'], 
                                   df['amp'], 
                                   u"Cu-K\u03B1, \u03bb = 1.5406\u00C5", 
                                   u"2\u03B8", 
                                   'cps'),
            style={'height': '85%','width': '100%'}
        )]


@app.callback(Output('highest_peaks', 'children'),
              [Input('store_data', 'children')])
def update_highest_peaks(df_json):
    """ Generate a table showing highest peaks found from data

    Args:
        df_json (json): json string of dataframe stored on web page

    Returns:
        DataTable: table of highest peaks found from  data
    """    
    if df_json:
        df = pd.read_json(df_json)
        peaks = find_peaks(df, 10)
        return dt.DataTable(columns = [{'name': i, 'id': i} for i in peaks.columns],
                            data = peaks.round(3).to_dict('records'),
                            style_as_list_view = True,
                            style_header = {'backgroundColor': colors['background'],
                                            'fontWeight': 'bold'})


@app.callback([Output('rf_prediction', 'children'),
               Output('rf_probabilities', 'children')],
              [Input('store_data', 'children')])
def update_rf_prediction(df_json):
    """ generate a reporting panel for prediction from random forest model

    Args:
        df_json (json): stored json string of dataframe

    Returns:
        list: list containing
        1. a picture of crystal structure for the predicted phase
        2. a table of probability of different phase labels.
    """    
    if df_json:
        df = pd.read_json(df_json)
        normed_data = np.array(normalization(df)).reshape(1, -1)
        prob_ratio = pd.DataFrame(rf_classifier.predict_proba(normed_data)[0]).T
        prob_ratio.columns = list(labels.values())
        prob_ratio.index = ['Probability']
        prob_ratio = prob_ratio.reset_index()
        prediction = rf_classifier.predict(normed_data)[0]
        return [html.Img(src=app.get_asset_url(prediction + '.png'), width=250),
                dt.DataTable(columns = [{'name': i, 'id': i} for i in prob_ratio.columns],
                             data = prob_ratio.round(3).to_dict('records'),
                             style_as_list_view = True,
                             style_header = {'backgroundColor': colors['background'],
                                             'fontWeight': 'bold' },
                             style_cell={'width': '20%'})]
    else:
        return '', ''


@app.callback([Output('nn_prediction', 'children'),
               Output('nn_probabilities', 'children')],
              [Input('store_data', 'children')])
def update_nn_prediction(df_json):
    """ generate a reporting panel for prediction from neural network model

    Args:
        df_json (json): stored json string of dataframe

    Returns:
        list: list containing
        1. a picture of crystal structure for the predicted phase
        2. a table of probability of different phase labels.
    """    
    if df_json:
        df = pd.read_json(df_json)
        normed_data = np.array(normalization(df)).reshape(1, -1)
        pred_array = nn_classifier.predict(normed_data)[0]
        prob_ratio = pd.DataFrame(pred_array).T
        prob_ratio_columns = ['index'] + list(labels.values())
        prob_ratio_dict = {labels[k]: round(v, 3) for k, v in prob_ratio.to_dict('records')[0].items()}
        prob_ratio_dict['index'] = 'Probability'
        prediction = labels[np.array(prob_ratio).argmax()]
        return (html.Img(src=app.get_asset_url(prediction + '.png'), width=250),
                dt.DataTable(columns = [{'name': i, 'id': i} for i in prob_ratio_columns],
                             data = [prob_ratio_dict],
                             style_as_list_view = True,
                             style_header = {'backgroundColor': colors['background'],
                                             'fontWeight': 'bold' },
                             style_cell={'width': '20%'}))
    else:
        return '', ''


if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run_server(
        host='127.0.0.1',
        port=5000,
        debug=False
    )
