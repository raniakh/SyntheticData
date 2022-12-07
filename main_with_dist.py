import datetime
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import math

mu_const = 0.002
sigma_const = 0.01
start_price = 30
week, weeks2, weeks3, weeks4, weeks5, weeks6 = 7, 14, 21, 28, 35, 42
beta1, beta2 = 0.8, 0.2
save_data_path = 'synthetic_data'


def handle_df(prices):
    todays_date = datetime.datetime.now().date()
    index = pd.date_range(end=todays_date, periods=3287, freq='D')
    columns = ["Open", "High", "Low", "Close", "gm_week", "gm_2_weeks", "gm_3_weeks", "gm_4_weeks", "gm_5_weeks",
               "gm_6_weeks", "pr_week", "pr_2_weeks", "pr_3_weeks", "pr_4_weeks", "pr_5_weeks", "pr_6_weeks"]
    df = pd.DataFrame(index=index, columns=columns)
    last_close = prices[-1:].item()  # must be updated with each new row
    prices_copy = np.copy(prices)
    for date in index:
        df.loc[date, 'Open'] = last_close
        add_gm_features(prices_copy, df, date)
        add_pr_features(prices_copy, df, date)
        last_close, prices_copy = add_close_high_low(prices, df, date)
    return df


def add_close_high_low(prices, df, index):
    # mu = beta1 * np.sin(dist[0]*df.loc[index, 'gm_week'] + dist[1]*df.loc[index, 'gm_2_weeks'] +
    #                     dist[2]*df.loc[index, 'gm_3_weeks'] + dist[3]*df.loc[index, 'gm_4_weeks'] +
    #                     dist[4]*df.loc[index, 'gm_5_weeks'] + dist[5]*df.loc[index, 'gm_6_weeks']) \
    #      + beta2 * np.sin(dist[0]*df.loc[index, 'pr_week'] + dist[1]*df.loc[index, 'pr_2_weeks'] +
    #                       dist[2]*df.loc[index, 'pr_3_weeks'] + dist[3]*df.loc[index, 'pr_4_weeks'] +
    #                       dist[4]*df.loc[index, 'pr_5_weeks'] + dist[5]*df.loc[index, 'pr_6_weeks'])

    # mu = beta1 * (dist[0] * np.sin(df.loc[index, 'gm_week']) + dist[1] * np.sin(df.loc[index, 'gm_2_weeks'])
    #               + dist[2] * np.sin(df.loc[index, 'gm_3_weeks']) + dist[3] * np.sin(df.loc[index, 'gm_4_weeks'])
    #               + dist[4] * np.sin(df.loc[index, 'gm_5_weeks']) + dist[5] * np.sin(df.loc[index, 'gm_6_weeks'])) \
    #      + beta2 * (dist[0] * np.sin(df.loc[index, 'pr_week']) + dist[1] * np.sin(df.loc[index, 'pr_2_weeks'])
    #                 + dist[2] * np.sin(df.loc[index, 'pr_3_weeks']) + dist[3] * np.sin(df.loc[index, 'pr_4_weeks'])
    #                 + dist[4] * np.sin(df.loc[index, 'pr_5_weeks']) + dist[5] * np.sin(df.loc[index, 'pr_6_weeks']))
    func = lambda f, c1, c2: 2 * math.pi * c1 * f + c2
    mu = beta1 * (dist[0] * np.sin(func(df.loc[index, 'gm_week'], 100, 0)) +
                  dist[1] * np.sin(func(df.loc[index, 'gm_2_weeks'], 200, 2)) +
                  dist[2] * np.sin(func(df.loc[index, 'gm_3_weeks'], 3, 3)) +
                  dist[3] * np.sin(func(df.loc[index, 'gm_4_weeks'], 10, 4)) +
                  dist[4] * np.sin(func(df.loc[index, 'gm_5_weeks'], 40, 5)) +
                  dist[5] * np.sin(func(df.loc[index, 'gm_6_weeks'], 1, 6))) \
         + beta2 * (dist[0] * np.sin(func(df.loc[index, 'pr_week'], 300, 7)) +
                    dist[1] * np.sin(func(df.loc[index, 'pr_2_weeks'], 5, 8)) +
                    dist[2] * np.sin(func(df.loc[index, 'pr_3_weeks'], 20, 9)) +
                    dist[3] * np.sin(func(df.loc[index, 'pr_4_weeks'], 30, 10)) +
                    dist[4] * np.sin(func(df.loc[index, 'pr_5_weeks'], 60, 11)) +
                    dist[5] * np.sin(func(df.loc[index, 'pr_6_weeks'], 6, 12)))

    close_price = df.loc[index, 'Open'] + np.random.normal(loc=mu, scale=sigma_const, size=1).item()
    df.loc[index, 'Close'] = close_price
    prices = np.append(prices, close_price)

    if df.loc[index, 'Open'] > df.loc[index, 'Close']:
        df.loc[index, 'High'] = random.uniform(df.loc[index, 'Open'], df.loc[index, 'Open'] + 3)
    else:
        df.loc[index, 'High'] = random.uniform(df.loc[index, 'Close'], df.loc[index, 'Close'] + 3)

    if df.loc[index, 'Open'] < df.loc[index, 'High']:
        df.loc[index, 'Low'] = random.uniform(df.loc[index, 'Open'] - 3, df.loc[index, 'Open'])
    else:
        df.loc[index, 'Low'] = random.uniform(df.loc[index, 'Close'] - 3, df.loc[index, 'Close'])

    return close_price, prices


def add_pr_features(prices, df, index):
    myfunc = lambda x: (x > 0).mean()
    prices_series = pd.Series(data=prices)
    norm_vector = pd.Series(prices_series).shift(1).fillna(method='bfill')
    prices_PrAdjusted = prices / norm_vector - 1

    pr_week = myfunc(prices_PrAdjusted[-week:])
    df.loc[index, 'pr_week'] = pr_week.item()
    pr_2_weeks = myfunc(prices_PrAdjusted[-weeks2:])
    df.loc[index, 'pr_2_weeks'] = pr_2_weeks.item()
    pr_3_weeks = myfunc(prices_PrAdjusted[-weeks3:])
    df.loc[index, 'pr_3_weeks'] = pr_3_weeks.item()
    pr_4_weeks = myfunc(prices_PrAdjusted[-weeks4:])
    df.loc[index, 'pr_4_weeks'] = pr_4_weeks.item()
    pr_5_weeks = myfunc(prices_PrAdjusted[-weeks5:])
    df.loc[index, 'pr_5_weeks'] = pr_5_weeks.item()
    pr_6_weeks = myfunc(prices_PrAdjusted[-weeks6:])
    df.loc[index, 'pr_6_weeks'] = pr_6_weeks.item()


def add_gm_features(prices, df, index):
    mult_percentage = 100
    gm_week = np.mean(prices[-week:])
    gm_2_weeks = np.mean(prices[-weeks2:])
    gm_3_weeks = np.mean(prices[-weeks3:])
    gm_4_weeks = np.mean(prices[-weeks4:])
    gm_5_weeks = np.mean(prices[-weeks5:])
    gm_6_weeks = np.mean(prices[-weeks6:])
    gm_week = mult_percentage * (gm_week / prices[-1:] - 1)
    df.loc[index, 'gm_week'] = gm_week.item()
    gm_2_weeks = mult_percentage * (gm_2_weeks / prices[-1:] - 1)
    df.loc[index, 'gm_2_weeks'] = gm_2_weeks.item()
    gm_3_weeks = mult_percentage * (gm_3_weeks / prices[-1:] - 1)
    df.loc[index, 'gm_3_weeks'] = gm_3_weeks.item()
    gm_4_weeks = mult_percentage * (gm_4_weeks / prices[-1:] - 1)
    df.loc[index, 'gm_4_weeks'] = gm_4_weeks.item()
    gm_5_weeks = mult_percentage * (gm_5_weeks / prices[-1:] - 1)
    df.loc[index, 'gm_5_weeks'] = gm_5_weeks.item()
    gm_6_weeks = mult_percentage * (gm_6_weeks / prices[-1:] - 1)
    df.loc[index, 'gm_6_weeks'] = gm_6_weeks.item()


if __name__ == '__main__':
    np.random.seed(0)
    returns = np.random.normal(loc=mu_const, scale=sigma_const, size=60)
    prices = start_price * (1 + returns).cumprod()
    plt.plot(prices)
    plt.title('first 60 days')
    plt.show()
    print(prices)
    dist = pd.read_csv('dist_for_synthetic.csv', header=None)
    dist = dist.to_numpy()
    dist = dist * 6
    df = handle_df(prices)
    plt.plot(df.loc[:, 'Close'])
    plt.title(
        'Close prices generated from behavioral features \nBeta: {},{}, sum[dist * sin(feature)]'.format(beta1, beta2),
        loc='center')
    plt.xlabel('years')
    plt.ylabel('price')
    # plt.figure(figsize=(30,30))
    plt.show()
    prices_new = df.loc[:, 'Close']
    prices_new = prices_new[60:]
    plt.plot(prices_new)
    plt.title('without 60 first days')
    plt.xlabel('years')
    plt.ylabel('price')
    plt.show()
    # prices_new_new = prices_new[60:90]
    # plt.plot(prices_new_new)
    # plt.title('60:90')
    # plt.show()
    df.to_csv(os.path.join(save_data_path, '{}.csv'.format('stock_synthetic_9years_dist_sin')), index=True)
