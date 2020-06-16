import numpy as np
import pandas as pd 
import plotly.graph_objs as go
from params import colors


def gen_scatter(x, y, title, x_title, y_title):
    """ Generate a scatter figure from dataframe
    Args:
        x (list): list of x values
        y (list): list of y values
        title (str): title of figure
        x_title (str): name of x-label
        y_title (str): name of y-label
    Returns:
        figure: generated figure
    """    
    trace = go.Scatter(
        x = x,
        y = y,
        marker = {
            'symbol': 'circle',
            'size': 10,
            'line': {
                'width': 1
            }
        }
    )
    layout = go.Layout(
        title = {'text': title, 'x': 0.95, 'y': 0.95, 'font': {'color': colors['font']}},
        xaxis = {'title': x_title, 'tickfont': {'size': 12}},
        yaxis = {'title': y_title, 'tickfont': {'size': 12}},
        font= {'size': 12, 'color': colors['font']},
        # height = h,
        margin = {'r':5, 't':5, 'l':5, 'b':5},
        paper_bgcolor = colors['panel'],

    )
    fig = go.Figure(data=[trace], layout=layout)
    return fig


def gen_heatmap(df, title, x_title, y_title, h=400):
    """ Generate a heatmap from dataframe
    Args:
        df (DataFrame): pandas dataframe for plotting
        title (str): title of the figure
        x_title (str): name of x-column
        y_title (str): name of y-column
        h (int, optional): height of figure. Defaults to 400.
    Returns:
        figure: generated figure
    """    
    trace = go.Heatmap(z=np.log(df), x=list(df.index), y=list(df.columns), colorscale = 'viridis', showscale = False)
    layout = go.Layout(
        # title = {'text': title, 'x': 0.2, 'y': 0.9},
        xaxis = {'title': x_title, 'tickfont': {'size': 12}},
        yaxis = {'title': y_title, 'tickfont': {'size': 12}},
        font= {'size': 12, 'color': 'white'},
        height = h,
        paper_bgcolor = colors['panel'],
        margin = {'r':5, 't':5, 'l':5, 'b':5},
    )
    fig = go.Figure(data=[trace], layout=layout)
    return fig