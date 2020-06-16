import pandas as pd
import numpy as np
import webbrowser


def filter_noise(x):
    """ A helper function for filtering noise

    Args:
        x (float): intensity or cps of spectrum

    Returns:
        float: filtered value
    """    
    return x if x > 5 else 0.01


def normalization(df):
    """ Generate normalized pattern for building models

    Args:
        df (DataFrame): a pandas dataframe containing spectrum data

    Returns:
        list: normalized pattern (a list of intensities)
    """    
    peak = df['amp'].max()
    data = df[df['amp'] > 0.05 * peak].fillna(0)
    main_peak_idx = df[df['2theta'] < 70]['amp'].argmax()
    main_peak_loc = df.iloc[main_peak_idx]['2theta']
    df['2theta'] = df['2theta'] / main_peak_loc * 30
    result = [filter_noise(df[(df['2theta'] >= i) & (df['2theta'] < i + 1)]['amp'].sum()) for i in range(10, 90, 1)]
    result = [i / max(result) for i in result]
    return result


def find_peaks(df, n=10):
    """ Find highest peaks from dataframe of spectrum

    Args:
        df (DataFrame): a pandas dataframe storing spectrum data
        n (int, optional): number of highest peaks Defaults to 10.

    Returns:
        DataFrame: a dataframe containing locations and values of highest peaks
    """
    angle = df[df['amp'] > 5]['2theta'].values
    amp = df[df['amp'] > 5]['amp'].values
    peaks = []
    for i in range(1, len(amp)-1):
        if amp[i] > amp[i - 1] and amp[i] > amp[i + 1]:
            peaks.append([angle[i], amp[i]])
    result = pd.DataFrame(peaks, columns=['2theta', 'cps'])
    result = result.sort_values(by='cps', ascending=False).head(n)
    result = result.reset_index()
    result['index'] = np.array([i for i in range(1, n+1)])
    return result


def open_browser():
    """ Open web browser when running the app
    """
    webbrowser.open_new('http://127.0.0.1:5000')
