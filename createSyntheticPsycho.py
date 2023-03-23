import datetime
import pandas as pd
import numpy as np
import random
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler


synthetic_data_path = 'synthetic_data'
save_data_path = 'synthetic_psycho'

mean_holding_days = 15 # =7*0.46368034570523+14*0.290266332400527+21*0.0360918678865935+28*0.0168704867465785+35*0.188624557440505+42*0.00446640982056413
mult_percentage = 100


def create_psychoFile(df, df_preprocess):
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

    df_preprocess.loc[:, 'gm'] = df.loc[:, 'Close'].rolling(mean_holding_days).mean()
    df_preprocess.loc[:, 'gm'] = mult_percentage * (df_preprocess.loc[:, 'gm'] / df.loc[:, 'Close'] - 1)

    df_preprocess.loc[:, 'pr'] = df_preprocess.loc[:, 'n_close'].rolling(mean_holding_days).agg(lambda x: (x>0).mean())
    df_preprocess.loc[:, 'pr'] = MinMaxScaler().fit_transform(df_preprocess.loc[:, 'pr'].values.reshape(-1,1))

    df_preprocess.loc[:, 'label'] = close_shifted / df.loc[:, 'Close'] - 1
    df_preprocess.loc[:, 'label'] = np.where((df_preprocess['label'] <= 0.0055) & (df_preprocess['label'] >= -0.005), 0,
                                             df_preprocess['label'])
    df_preprocess.loc[df_preprocess['label'] < 0, 'label'] = -1
    df_preprocess.loc[df_preprocess['label'] > 0, 'label'] = 1
    df_preprocess.loc[:, 'Volume'] = 123456


if __name__ == '__main__':
    stock_path = os.path.join(synthetic_data_path, 'synthetic_data_2_raw.csv')
    df = pd.read_csv(stock_path)
    columns = ["c_open", "c_high", "c_low", "n_close", "n_adj_close", "5day", "10day", "15day", "20day", "25day", "30day",
               "gm", "pr"]
    df_preprocess = pd.DataFrame(columns=columns)
    create_psychoFile(df=df, df_preprocess=df_preprocess)
    df_preprocess.dropna(axis=0, inplace=True)
    df_preprocess.to_csv(os.path.join(save_data_path, 'synthetic_data_2_psycho.csv'), header=None, index=False)
