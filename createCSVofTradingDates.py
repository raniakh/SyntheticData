import datetime
import pandas as pd
import os

save_data_path = ''
if __name__ == '__main__':
    tradingDates = pd.bdate_range(start='30/12/2019', end='29/12/2022') # format M/d/yyyy
    tradingDates = tradingDates.strftime("%d/%m/%Y")
    df = pd.DataFrame(data=tradingDates)
    df.to_csv(os.path.join(save_data_path, '{}.csv'.format('trading_dates_simulated')), index=False)

