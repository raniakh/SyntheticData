import pandas as pd
import numpy as np
import random
from random import seed
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

synthetic_data_path = 'synthetic_data'
synthetic_data_behavioral_features_path = 'synthetic_data_behavioral_features_equation10'
week, weeks2, weeks3, weeks4, weeks5, weeks6 = 8, 15, 22, 29, 36, 43
beta11, beta21, beta31 = 0.3, 0.3, 0.4
beta12, beta22, beta32, beta42, beta52 = 0.1, 0.2, 0.1, 0.2, 0.4
beta13, beta23 = 0.6, 0.4
beta14, beta24 = 0.4, 0.6
beta15, beta25 = 0.5, 0.5
beta16, beta26 = 0.7, 0.3
beta17 = 0.2
beta18, beta28, beta38 = 0.4, 0.3, 0.1
beta19, beta29 = 0.4, 0.2
beta110, beta210, beta310 = 0.1, 0.2, 0.2


# calculate features then from features calculate close, high, low
def add_gm_features(df, df_new, index):
    mult_percentage = 100

    last_week_rows = df_new.iloc[-week:]
    last_week_close = last_week_rows.iloc[:, 3].values
    last_week_close = last_week_close[~np.isnan(last_week_close)]
    df_new.loc[index, 'gm_week'] = np.mean(last_week_close.astype('float'))

    last_2weeks_rows = df_new.iloc[-weeks2:]
    last_2weeks_close = last_2weeks_rows.iloc[:, 3].values
    last_2weeks_close = last_2weeks_close[~np.isnan(last_2weeks_close)]
    df_new.loc[index, 'gm_2_weeks'] = np.mean(last_2weeks_close.astype('float'))

    last_3weeks_rows = df_new.iloc[-weeks3:]
    last_3weeks_close = last_3weeks_rows.iloc[:, 3].values
    last_3weeks_close = last_3weeks_close[~np.isnan(last_3weeks_close)]
    df_new.loc[index, 'gm_3_weeks'] = np.mean(last_3weeks_close.astype('float'))

    last_4weeks_rows = df_new.iloc[-weeks4:]
    last_4weeks_close = last_4weeks_rows.iloc[:, 3].values
    last_4weeks_close = last_4weeks_close[~np.isnan(last_4weeks_close)]
    df_new.loc[index, 'gm_4_weeks'] = np.mean(last_4weeks_close.astype('float'))

    last_5weeks_rows = df_new.iloc[-weeks5:]
    last_5weeks_close = last_5weeks_rows.iloc[:, 3].values
    last_5weeks_close = last_5weeks_close[~np.isnan(last_5weeks_close)]
    df_new.loc[index, 'gm_5_weeks'] = np.mean(last_5weeks_close.astype('float'))

    last_6weeks_rows = df_new.iloc[-weeks6:]
    last_6weeks_close = last_6weeks_rows.iloc[:, 3].values
    last_6weeks_close = last_6weeks_close[~np.isnan(last_6weeks_close)]
    df_new.loc[index, 'gm_6_weeks'] = np.mean(last_6weeks_close.astype('float'))

    tmp = mult_percentage * (df_new.loc[index, 'gm_week'] / df_new.loc[index - 1, 'Close'] - 1)
    df_new.loc[index, 'gm_week'] = tmp

    df_new.loc[index, 'gm_2_weeks'] = mult_percentage * (
            df_new.loc[index, 'gm_2_weeks'] / df_new.loc[index - 1, 'Close'] - 1)

    df_new.loc[index, 'gm_3_weeks'] = mult_percentage * (
            df_new.loc[index, 'gm_3_weeks'] / df_new.loc[index - 1, 'Close'] - 1)

    df_new.loc[index, 'gm_4_weeks'] = mult_percentage * (
            df_new.loc[index, 'gm_4_weeks'] / df_new.loc[index - 1, 'Close'] - 1)

    df_new.loc[index, 'gm_5_weeks'] = mult_percentage * (
            df_new.loc[index, 'gm_5_weeks'] / df_new.loc[index - 1, 'Close'] - 1)

    df_new.loc[index, 'gm_6_weeks'] = mult_percentage * (
            df_new.loc[index, 'gm_6_weeks'] / df_new.loc[index - 1, 'Close'] - 1)


def add_pr_features(df, df_new, index):
    myfunc = lambda x: (x > 0).mean()
    last_week_rows = df_new.iloc[-week:]
    last_week_close = last_week_rows.iloc[:, 3].values.astype('float')
    df_new.loc[index, 'pr_week'] = myfunc(last_week_close)

    last_2weeks_rows = df_new.iloc[-weeks2:]
    last_2weeks_close = last_2weeks_rows.iloc[:, 3].values
    df_new.loc[index, 'pr_2_weeks'] = myfunc(last_2weeks_close)

    last_3weeks_rows = df_new.iloc[-weeks3:]
    last_3weeks_close = last_3weeks_rows.iloc[:, 3].values
    df_new.loc[index, 'pr_3_weeks'] = myfunc(last_3weeks_close)

    last_4weeks_rows = df_new.iloc[-weeks4:]
    last_4weeks_close = last_4weeks_rows.iloc[:, 3].values
    df_new.loc[index, 'pr_4_weeks'] = myfunc(last_4weeks_close)

    last_5weeks_rows = df_new.iloc[-weeks5:]
    last_5weeks_close = last_5weeks_rows.iloc[:, 3].values
    df_new.loc[index, 'pr_5_weeks'] = myfunc(last_5weeks_close)

    last_6weeks_rows = df_new.iloc[-weeks6:]
    last_6weeks_close = last_6weeks_rows.iloc[:, 3].values
    df_new.loc[index, 'pr_6_weeks'] = myfunc(last_6weeks_close)

    # stock_sample.loc[:, 'pr_week'] = MinMaxScaler().fit_transform(stock_sample.loc[:, 'pr_week'].values.reshape(-1, 1))
    # stock_sample.loc[:, 'pr_2_weeks'] = MinMaxScaler().fit_transform(
    #     stock_sample.loc[:, 'pr_2_weeks'].values.reshape(-1, 1))
    # stock_sample.loc[:, 'pr_3_weeks'] = MinMaxScaler().fit_transform(
    #     stock_sample.loc[:, 'pr_3_weeks'].values.reshape(-1, 1))
    # stock_sample.loc[:, 'pr_4_weeks'] = MinMaxScaler().fit_transform(
    #     stock_sample.loc[:, 'pr_4_weeks'].values.reshape(-1, 1))
    # stock_sample.loc[:, 'pr_5_weeks'] = MinMaxScaler().fit_transform(
    #     stock_sample.loc[:, 'pr_5_weeks'].values.reshape(-1, 1))
    # stock_sample.loc[:, 'pr_6_weeks'] = MinMaxScaler().fit_transform(
    #     stock_sample.loc[:, 'pr_6_weeks'].values.reshape(-1, 1))


def add_close_high_low(df, df_new, index):
    # equation 1
    df_new.loc[index, 'Close'] = df_new.loc[index, 'Open'] \
                                 + beta11 * (df_new.loc[index, 'pr_week'] + df_new.loc[index, 'pr_2_weeks']
                                             + df_new.loc[index, 'pr_3_weeks'] + df_new.loc[index, 'pr_4_weeks']
                                             + df_new.loc[index, 'pr_5_weeks'] + df_new.loc[index, 'pr_6_weeks']) \
                                 + beta21 * (df_new.loc[index, 'gm_week'] + df_new.loc[index, 'gm_2_weeks']
                                             + df_new.loc[index, 'gm_3_weeks'] + df_new.loc[index, 'gm_4_weeks']
                                             + df_new.loc[index, 'gm_5_weeks'] + df_new.loc[index, 'gm_6_weeks']) + \
                                 np.random.normal(0, 1)
    # equation 2
    df_new.loc[index, 'Close'] = df_new.loc[index, 'Open'] \
                                 + beta12 * (df_new.loc[index, 'pr_week'] + df_new.loc[index, 'pr_2_weeks']
                                             + df_new.loc[index, 'pr_3_weeks'] + df_new.loc[index, 'pr_4_weeks']
                                             + df_new.loc[index, 'pr_5_weeks'] + df_new.loc[index, 'pr_6_weeks']) \
                                 + beta22 * (df_new.loc[index, 'pr_week'] ** 2 + df_new.loc[index, 'pr_2_weeks'] ** 2
                                             + df_new.loc[index, 'pr_3_weeks'] ** 2 + df_new.loc[index, 'pr_4_weeks'] ** 2
                                             + df_new.loc[index, 'pr_5_weeks'] ** 2 + df_new.loc[index, 'pr_6_weeks'] ** 2) \
                                 + beta32 * (df_new.loc[index, 'gm_week'] + df_new.loc[index, 'gm_2_weeks']
                                             + df_new.loc[index, 'gm_3_weeks'] + df_new.loc[index, 'gm_4_weeks']
                                             + df_new.loc[index, 'gm_5_weeks'] + df_new.loc[index, 'gm_6_weeks']) \
                                 + beta42 * (df_new.loc[index, 'gm_week'] ** 2 + df_new.loc[index, 'gm_2_weeks'] ** 2
                                             + df_new.loc[index, 'gm_3_weeks'] ** 2 + df_new.loc[index, 'gm_4_weeks'] ** 2
                                             + df_new.loc[index, 'gm_5_weeks'] ** 2 + df_new.loc[index, 'gm_6_weeks'] ** 2) \
                                 + beta52 * (df_new.loc[index, 'pr_week'] * df_new.loc[index, 'gm_week']
                                             + df_new.loc[index, 'pr_2_weeks'] * df_new.loc[index, 'gm_2_weeks']
                                             + df_new.loc[index, 'pr_3_weeks'] * df_new.loc[index, 'gm_3_weeks']
                                             + df_new.loc[index, 'pr_4_weeks'] * df_new.loc[index, 'gm_4_weeks']
                                             + df_new.loc[index, 'pr_5_weeks'] * df_new.loc[index, 'gm_5_weeks']
                                             + df_new.loc[index, 'pr_6_weeks'] * df_new.loc[index, 'gm_6_weeks'] ) \
                                 + np.random.normal(0, 1)
    # equation 3
    df_new.loc[index, 'Close'] = df_new.loc[index, 'Open'] \
                                 + beta13 * (df_new.loc[index, 'pr_week'] ** 2 + df_new.loc[index, 'pr_2_weeks'] ** 2
                                             + df_new.loc[index, 'pr_3_weeks'] ** 2 + df_new.loc[index, 'pr_4_weeks'] ** 2
                                             + df_new.loc[index, 'pr_5_weeks'] ** 2 + df_new.loc[index, 'pr_6_weeks'] ** 2) \
                                 + beta23 * (df_new.loc[index, 'gm_week'] + df_new.loc[index, 'gm_2_weeks']
                                             + df_new.loc[index, 'gm_3_weeks'] + df_new.loc[index, 'gm_4_weeks']
                                             + df_new.loc[index, 'gm_5_weeks'] + df_new.loc[index, 'gm_6_weeks']) \
                                 + np.random.normal(0, 1)

    # equation 4
    df_new.loc[index, 'Close'] = df_new.loc[index, 'Open']\
                                 + beta14 * (df_new.loc[index, 'pr_week'] + df_new.loc[index, 'pr_2_weeks']
                                             + df_new.loc[index, 'pr_3_weeks'] + df_new.loc[index, 'pr_4_weeks']
                                             + df_new.loc[index, 'pr_5_weeks'] + df_new.loc[index, 'pr_6_weeks']) \
                                 + beta24 * (df_new.loc[index, 'gm_week'] ** 2 + df_new.loc[index, 'gm_2_weeks'] ** 2
                                             + df_new.loc[index, 'gm_3_weeks'] ** 2 + df_new.loc[index, 'gm_4_weeks'] ** 2
                                             + df_new.loc[index, 'gm_5_weeks'] ** 2 + df_new.loc[index, 'gm_6_weeks'] ** 2) \
                                 + np.random.normal(0, 1)
    # equation 5
    df_new.loc[index, 'Close'] = df_new.loc[index, 'Open'] \
                                 + beta15 * (df_new.loc[index, 'pr_week'] ** 2 + df_new.loc[index, 'pr_2_weeks'] ** 2
                                             + df_new.loc[index, 'pr_3_weeks'] ** 2 + df_new.loc[index, 'pr_4_weeks'] ** 2
                                             + df_new.loc[index, 'pr_5_weeks'] ** 2 + df_new.loc[index, 'pr_6_weeks'] ** 2) \
                                 + beta25 * (df_new.loc[index, 'gm_week'] ** 2 + df_new.loc[index, 'gm_2_weeks'] ** 2
                                             + df_new.loc[index, 'gm_3_weeks'] ** 2 + df_new.loc[index, 'gm_4_weeks'] ** 2
                                             + df_new.loc[index, 'gm_5_weeks'] ** 2 + df_new.loc[index, 'gm_6_weeks'] ** 2) \
                                 + np.random.normal(0, 1)
    # equation 6
    df_new.loc[index, 'Close'] = df_new.loc[index, 'Open'] + \
                                 beta16 * (df_new.loc[index, 'pr_week'] ** 5 + df_new.loc[index, 'pr_2_weeks'] ** 5
                                           + df_new.loc[index, 'pr_3_weeks'] ** 5 + df_new.loc[index, 'pr_4_weeks'] ** 5
                                           + df_new.loc[index, 'pr_5_weeks'] ** 5 + df_new.loc[
                                               index, 'pr_6_weeks'] ** 5) \
                                 + beta26 * (df_new.loc[index, 'gm_week'] + df_new.loc[index, 'gm_2_weeks']
                                             + df_new.loc[index, 'gm_3_weeks'] + df_new.loc[index, 'gm_4_weeks']
                                             + df_new.loc[index, 'gm_5_weeks'] + df_new.loc[index, 'gm_6_weeks'])
    # equation 7
    df_new.loc[index, 'Close'] = df_new.loc[index, 'Open'] \
                                 + beta17 * (df_new.loc[index, 'pr_week'] + df_new.loc[index, 'pr_2_weeks']
                                             + df_new.loc[index, 'pr_3_weeks'] + df_new.loc[index, 'pr_4_weeks']
                                             + df_new.loc[index, 'pr_5_weeks'] + df_new.loc[index, 'pr_6_weeks']) \
                                 * (df_new.loc[index, 'gm_week'] + df_new.loc[index, 'gm_2_weeks']
                                    + df_new.loc[index, 'gm_3_weeks'] + df_new.loc[index, 'gm_4_weeks']
                                    + df_new.loc[index, 'gm_5_weeks'] + df_new.loc[index, 'gm_6_weeks'])
    # equation 8
    df_new.loc[index, 'Close'] = df_new.loc[index, 'Open'] \
                                 + beta18 * (df_new.loc[index, 'pr_week'] + df_new.loc[index, 'pr_2_weeks']
                                             + df_new.loc[index, 'pr_3_weeks'] + df_new.loc[index, 'pr_4_weeks']
                                             + df_new.loc[index, 'pr_5_weeks'] + df_new.loc[index, 'pr_6_weeks']) \
                                 + beta28 * (df_new.loc[index, 'pr_week'] ** 3 + df_new.loc[index, 'pr_2_weeks'] ** 3
                                             + df_new.loc[index, 'pr_3_weeks'] ** 3 + df_new.loc[index, 'pr_4_weeks'] ** 3
                                             + df_new.loc[index, 'pr_5_weeks'] ** 3 + df_new.loc[index, 'pr_6_weeks'] ** 3) \
                                 + beta38 * (df_new.loc[index, 'gm_week'] + df_new.loc[index, 'gm_2_weeks']
                                    + df_new.loc[index, 'gm_3_weeks'] + df_new.loc[index, 'gm_4_weeks']
                                    + df_new.loc[index, 'gm_5_weeks'] + df_new.loc[index, 'gm_6_weeks'])
    # equation 9
    df_new.loc[index, 'Close'] = df_new.loc[index, 'Open'] \
                                 + beta19 * (df_new.loc[index, 'pr_week'] ** 5 + df_new.loc[index, 'pr_2_weeks'] ** 5
                                             + df_new.loc[index, 'pr_3_weeks'] ** 5 + df_new.loc[index, 'pr_4_weeks'] ** 5
                                             + df_new.loc[index, 'pr_5_weeks'] ** 5 + df_new.loc[index, 'pr_6_weeks'] ** 5)  \
                                 + beta29 * (df_new.loc[index, 'gm_week'] ** 3 + df_new.loc[index, 'gm_2_weeks'] ** 3
                                    + df_new.loc[index, 'gm_3_weeks'] ** 3 + df_new.loc[index, 'gm_4_weeks'] ** 3
                                    + df_new.loc[index, 'gm_5_weeks'] ** 3 + df_new.loc[index, 'gm_6_weeks'] ** 3)
    # equation 10
    df_new.loc[index, 'Close'] = df_new.loc[index, 'Open'] \
                                 + beta110 * (df_new.loc[index, 'pr_week'] + df_new.loc[index, 'pr_2_weeks']
                                             + df_new.loc[index, 'pr_3_weeks'] + df_new.loc[index, 'pr_4_weeks']
                                             + df_new.loc[index, 'pr_5_weeks'] + df_new.loc[index, 'pr_6_weeks']) \
                                 * (df_new.loc[index, 'gm_week'] + df_new.loc[index, 'gm_2_weeks']
                                    + df_new.loc[index, 'gm_3_weeks'] + df_new.loc[index, 'gm_4_weeks']
                                    + df_new.loc[index, 'gm_5_weeks'] + df_new.loc[index, 'gm_6_weeks']) \
                                 + beta210 * (df_new.loc[index, 'pr_week'] ** 3 + df_new.loc[index, 'pr_2_weeks'] ** 3
                                             + df_new.loc[index, 'pr_3_weeks'] ** 3 + df_new.loc[index, 'pr_4_weeks'] ** 3
                                             + df_new.loc[index, 'pr_5_weeks'] ** 3 + df_new.loc[index, 'pr_6_weeks'] ** 3) \
                                 + beta310 * (df_new.loc[index, 'gm_week'] + df_new.loc[index, 'gm_2_weeks']
                                    + df_new.loc[index, 'gm_3_weeks'] + df_new.loc[index, 'gm_4_weeks']
                                    + df_new.loc[index, 'gm_5_weeks'] + df_new.loc[index, 'gm_6_weeks'])

    if df_new.loc[index, 'Open'] > df_new.loc[index, 'Close']:
        df_new.loc[index, 'High'] = random.uniform(df_new.loc[index, 'Open'], df_new.loc[index, 'Open'] + 3)
    else:
        df_new.loc[index, 'High'] = random.uniform(df_new.loc[index, 'Close'], df_new.loc[index, 'Close'] + 3)

    if df_new.loc[index, 'Open'] < df_new.loc[index, 'High']:
        df_new.loc[index, 'Low'] = random.uniform(df_new.loc[index, 'Open'] - 3, df_new.loc[index, 'Open'])
    else:
        df_new.loc[index, 'Low'] = random.uniform(df_new.loc[index, 'Close'] - 3, df_new.loc[index, 'Close'])

    return df_new.loc[index, 'Close']


if __name__ == '__main__':
    for stock in os.listdir(synthetic_data_path):
        stock_path = os.path.join(synthetic_data_path, '{}'.format(stock))
        df = pd.read_csv(stock_path)
        data = df.iloc[-1]
        open_value = data[0]
        close_value = data[3]
        # column_names = ["Open", "High", "Low", "Close", "gm_week", "gm_2_weeks", "gm_3_weeks", "gm_4_weeks",
        #                 "gm_5_weeks", "gm_6_weeks", "pr_week", "pr_2_weeks", "pr_3_weeks", "pr_4_weeks",
        #                 "pr_5_weeks", "pr_6_weeks"]
        # df_new = pd.DataFrame(columns=column_names)
        df_new = df.copy()
        # stock_sample_data_testing = [1, -2, 3, -4, 5, 6, 7, -8, 9, 10, 11, 12]
        # stock_sample = pd.DataFrame(stock_sample_data_testing, columns=["Adj Close"])
        start_index = df_new.last_valid_index() + 1
        df_new.loc[start_index, 'Open'] = open_value
        last_index = 652 + start_index
        for index in range(start_index, last_index):
            add_gm_features(df, df_new, index=index)
            col_adj_t_minus_1 = 'Close'
            norm_vector = pd.Series(df_new['Close']).shift(1).fillna(method='bfill')
            df_close_backup = df_new.loc[:, 'Close'].copy()
            df_new[col_adj_t_minus_1] = df_new[col_adj_t_minus_1] / norm_vector - 1
            add_pr_features(df, df_new, index=index)
            new_close = add_close_high_low(df, df_new, index=index)
            df_close_backup = df_close_backup.drop(df_close_backup.size - 1)
            df_close_backup = df_close_backup.append(pd.Series(new_close))
            df_new.loc[:, 'Close'] = df_close_backup.values
            if index < last_index - 1:
                df_new.loc[index + 1, 'Open'] = df_new.loc[index, 'Close']
        df_new.to_csv(os.path.join(synthetic_data_behavioral_features_path, '{}.csv'.format(stock)), index=False)
