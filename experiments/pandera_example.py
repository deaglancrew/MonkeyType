import pandas as pd


def makes_df():
    return pd.DataFrame({'hello': [1, 2, 3], 'there': [1.2, 2.3, 3.4], 'I': ['am', 'a', 'dataframe']})


def takes_df(df):
    return df['hello']
