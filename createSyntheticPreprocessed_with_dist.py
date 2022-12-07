import datetime
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt

synthetic_data_path = 'synthetic_data'
save_data_path = 'synthetic_preprocess'


def create_preprocessfile(df, df_preprocess):
    df_preprocess.loc[:, 'c_open'] = df.loc[:, 'Open'] / df.loc[:, 'Close'] - 1
    df_preprocess.loc[:, 'c_high'] = df.loc[:, 'High'] / df.loc[:, 'Close'] - 1
    df_preprocess.loc[:, 'c_low'] = df.loc[:, 'Low'] / df.loc[:, 'Close'] - 1
    close_shifted = pd.Series(df['Close']).shift(1).fillna(method='bfill')
    df_preprocess.loc[:, 'n_close'] = df.loc[:, 'Close'] / close_shifted - 1
    df_preprocess.loc[:, 'n_adj_close'] = df_preprocess.loc[:, 'n_close']
    df_preprocess.loc[:, '5day'] = df.loc[:, 'Close'].rolling(5).mean().fillna(method='bfill')
    # df_preprocess.loc[:, '5day'] = df_preprocess['5day'].fillna(method='bfill')
    df_preprocess.loc[:, '5day'] = df_preprocess.loc[:, '5day'] / df.loc[:, 'Close'] - 1

    df_preprocess.loc[:, '10day'] = df.loc[:, 'Close'].rolling(10).mean().fillna(method='bfill')
    # df_preprocess.loc[:, '10day'] = df_preprocess['10day'].fillna(method='bfill')
    df_preprocess.loc[:, '10day'] = df_preprocess.loc[:, '10day'] / df.loc[:, 'Close'] - 1

    df_preprocess.loc[:, '15day'] = df.loc[:, 'Close'].rolling(15).mean().fillna(method='bfill')
    df_preprocess.loc[:, '15day'] = df_preprocess.loc[:, '15day'] / df.loc[:, 'Close'] - 1

    df_preprocess.loc[:, '20day'] = df.loc[:, 'Close'].rolling(20).mean().fillna(method='bfill')
    df_preprocess.loc[:, '20day'] = df_preprocess.loc[:, '20day'] / df.loc[:, 'Close'] - 1

    df_preprocess.loc[:, '25day'] = df.loc[:, 'Close'].rolling(25).mean().fillna(method='bfill')
    df_preprocess.loc[:, '25day'] = df_preprocess.loc[:, '25day'] / df.loc[:, 'Close'] - 1

    df_preprocess.loc[:, '30day'] = df.loc[:, 'Close'].rolling(30).mean().fillna(method='bfill')
    df_preprocess.loc[:, '30day'] = df_preprocess.loc[:, '30day'] / df.loc[:, 'Close'] - 1

    df_preprocess.loc[:, 'gm_week'] = df.loc[:, 'gm_week']
    df_preprocess.loc[:, 'gm_2_weeks'] = df.loc[:, 'gm_2_weeks']
    df_preprocess.loc[:, 'gm_3_weeks'] = df.loc[:, 'gm_3_weeks']
    df_preprocess.loc[:, 'gm_4_weeks'] = df.loc[:, 'gm_4_weeks']
    df_preprocess.loc[:, 'gm_5_weeks'] = df.loc[:, 'gm_5_weeks']
    df_preprocess.loc[:, 'gm_6_weeks'] = df.loc[:, 'gm_6_weeks']
    df_preprocess.loc[:, 'pr_week'] = df.loc[:, 'pr_week']
    df_preprocess.loc[:, 'pr_2_weeks'] = df.loc[:, 'pr_2_weeks']
    df_preprocess.loc[:, 'pr_3_weeks'] = df.loc[:, 'pr_3_weeks']
    df_preprocess.loc[:, 'pr_4_weeks'] = df.loc[:, 'pr_4_weeks']
    df_preprocess.loc[:, 'pr_5_weeks'] = df.loc[:, 'pr_5_weeks']
    df_preprocess.loc[:, 'pr_6_weeks'] = df.loc[:, 'pr_6_weeks']
    df_preprocess.loc[:, 'label'] = close_shifted / df.loc[:, 'Close'] - 1
    df_preprocess.loc[:, 'label'] = np.where((df_preprocess['label'] <= 0.000055) & (df_preprocess['label'] >= -0.00005), 0,
                                             df_preprocess['label'])
    df_preprocess.loc[df_preprocess['label'] < 0, 'label'] = -1
    df_preprocess.loc[df_preprocess['label'] > 0, 'label'] = 1
    df_preprocess.loc[:, 'Volume'] = 123456


if __name__ == '__main__':
    stock_path = os.path.join(synthetic_data_path, 'stock_with_dist.csv')
    df = pd.read_csv(stock_path)
    columns = ["c_open", "c_high", "c_low", "n_close", "5day", "10day", "15day", "20day", "25day", "30day",
               "gm_week", "gm_2_weeks", "gm_3_weeks", "gm_4_weeks", "gm_5_weeks", "gm_6_weeks", "pr_week",
               "pr_2_weeks", "pr_3_weeks", "pr_4_weeks", "pr_5_weeks", "pr_6_weeks"]
    df_preprocess = pd.DataFrame(columns=columns)
    create_preprocessfile(df=df, df_preprocess=df_preprocess)
    df_preprocess.to_csv(os.path.join(save_data_path, 'stock_0.6sin(dist gm)+1sin(dist pr).csv'), header=None, index=False)
