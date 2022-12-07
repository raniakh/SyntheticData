import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from random import seed
import os

synthetic_data_path = 'synthetic_data'


def random_walk(df, threshold=0.5, step_size=2,
                min_value=-np.inf, max_value=np.inf, drift=1):
    start_value = random.randint(-1, 100)
    previous_value = start_value
    for index in range(45):
        if previous_value < min_value:
            previous_value = min_value
        if previous_value > max_value:
            previous_value = max_value
        probability = random.random()
        if probability >= threshold:
            df.loc[index, 'Close'] = drift + previous_value + step_size
        else:
            df.loc[index, 'Close'] = drift + previous_value - step_size
        df.loc[index, 'Open'] = previous_value
        if df.loc[index, 'Open'] > df.loc[index, 'Close']:
            df.loc[index, 'High'] = random.randrange(df.loc[index, 'Open'], df.loc[index, 'Open'] + 3)
        else:
            df.loc[index, 'High'] = random.randrange(df.loc[index, 'Close'], df.loc[index, 'Close'] + 3)

        if df.loc[index, 'Open'] < df.loc[index, 'High']:
            df.loc[index, 'Low'] = random.randrange(df.loc[index, 'Open']-3, df.loc[index, 'Open'])
        else:
            df.loc[index, 'Low'] = random.randrange(df.loc[index, 'Close']-3, df.loc[index, 'Close'])
        previous_value = df.loc[index, 'Close']
    return df


if __name__ == '__main__':
    seed(1)
    # stock = 'test6'
    stocks = ["AAPL", "ABB", "ABBV", "AEP", "AGFS", "AMGN", "AMZN", "BA", "BABA", "BAC", "BBL", "BCH",
              "BHP", "BP", "BRK-A", "BSAC", "BUD", "C", "CAT", "CELG", "CHL", "CHTR", "CMCSA", "CODI",
              "CSCO", "CVX", "D", "DHR", "DIS", "DUK", "EXC", "FB", "GD", "GE", "GOOG", "HD", "HON",
              "HRG", "HSBC", "IEP", "INTC", "JNJ", "JPM", "KO", "LMT", "MA", "MCD", "MDT", "MMM",
              "MO", "MRK", "MSFT", "NEE", "NGG", "NVS", "ORCL", "PCG", "PCLN", "PEP", "PFE", "PG",
              "PICO", "PM", "PPL", "PTR", "RDS-B", "REX", "SLB", "SNP", "SNY", "SO", "SPLP", "SRE",
              "T", "TM", "TOT", "TSM", "UL", "UN", "UNH", "UPS", "UTX", "V", "VZ", "WFC", "WMT", "XOM"]
    column_names = ["Open", "High", "Low", "Close"]
    for stock in stocks:
        df = pd.DataFrame(columns=column_names)
        df_n = random_walk(df=df)
        df_n.to_csv(os.path.join(synthetic_data_path, '{}.csv'.format(stock)), index=False)
    # try:
    #     for stock in stocks:
    #         df = pd.DataFrame(columns=column_names)
    #         df_n = random_walk(df=df)
    #         df_n.to_csv(os.path.join(synthetic_data_path, '{}.csv'.format(stock)))
    # except Exception as e:
    #     print(e)


