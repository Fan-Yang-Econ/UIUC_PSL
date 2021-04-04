import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta

FOLDER = '/Users/fanyang/Dropbox/uiuc/cs598/UIUC_SPL/UIUC_PSL/Project2/'

train = pd.read_csv(FOLDER +'train_ini.csv', parse_dates = ['Date'])
# df_train.dtypes.value_counts()
test = pd.read_csv(FOLDER + 'test.csv', parse_dates = ['Date'])
# data processing - change date fto datetime format and get year, month and week indicator
# df_train['Date'] = pd.to_datetime(df_train['Date'])
def data_clean(data):
    data['Week'] = data['Date'].dt.isocalendar().week
    data['Year'] = data['Date'].dt.isocalendar().year
    data['Month'] = pd.DatetimeIndex(data['Date']).month
    data['Week'] = data.apply(lambda x: x['Week'] - 1 if x['Year'] == 2010 else x['Week'], axis=1)

    return data
# train = data_clean(train)
# test = data_clean(test)
# df_train['Week'] = df_train['Date'].dt.isocalendar().week
# df_train['Year'] = df_train['Date'].dt.isocalendar().year
# df_train['Month'] = pd.DatetimeIndex(df_train['Date']).month


def mypredict(train, test, next_fold, t):
    # delta_m = 2*(t-1)
    # start_date = datetime.date(2011, 3, 1) + relativedelta(months=delta_m)
    # end_date = datetime.date(2011, 5, 1) + relativedelta(months=delta_m)
    # test['fold'] = 0
    clean_train = data_clean(train)
    clean_test = data_clean(test)

    if t >1:
        clean_next_fold = data_clean(next_fold)
        clean_train = pd.concat([train, clean_next_fold], ignore_index= True)

    clean_test['Last_Year'] = clean_test['Year'] - 1

    test_pred = clean_test.merge(clean_train,
                                 left_on=['Store', 'Dept', 'Last_Year', 'Week'],
                                 right_on=['Store', 'Dept', 'Year', 'Week'],
                                 # indicator=True,
                                 how = 'left',
                                 suffixes=('', '_pre'))
    test_pred =test_pred.rename(columns ={'Weekly_Sales': 'Weekly_Pred'})


    return (clean_train,test_pred)

# next_fold = pd.read_csv(FOLDER + 'fold_1.csv', parse_dates=['Date'])
# scoring_df = next_fold.merge(test_pred, on=['Date', 'Store', 'Dept'], how='left')
# weights = scoring_df['IsHoliday_x'].apply(lambda is_holiday: 5 if is_holiday else 1).to_numpy()
# actuals = scoring_df['Weekly_Sales'].to_numpy()
# preds = scoring_df['Weekly_Pred'].fillna(0).to_numpy()
# wae.append((np.sum(weights * np.abs(actuals - preds)) / np.sum(weights)).item())

# df.date + pd.DateOffset(months=plus_month_period)
# save weighed mean absolute error WMAE
n_folds = 10
next_fold = None
wae = []

# time-series CV
for t in range(1, n_folds+1):
    print(f'Fold{t}...')

    # *** THIS IS YOUR PREDICTION FUNCTION ***
    train, test_pred = mypredict(train, test, next_fold, t)

    # Load fold file
    # You should add this to your training data in the next call to mypredict()
    fold_file = 'fold_{t}.csv'.format(t=t)
    next_fold = pd.read_csv(FOLDER + fold_file, parse_dates=['Date'])

    # extract predictions matching up to the current fold
    scoring_df = next_fold.merge(test_pred, on=['Date', 'Store', 'Dept'], how='left', indicator=True)

    # extract weights and convert to numpy arrays for wae calculation
    weights = scoring_df['IsHoliday_x'].apply(lambda is_holiday:5 if is_holiday else 1).to_numpy()
    actuals = scoring_df['Weekly_Sales'].to_numpy()
    preds = scoring_df['Weekly_Pred'].fillna(0).to_numpy()

    wae.append((np.sum(weights * np.abs(actuals - preds)) / np.sum(weights)).item())

print(wae)
print(sum(wae)/len(wae))
# mypredict requirement:
# https://piazza.com/class/kjvsp15j2g07ac?cid=258

# mask = (train_raw['Date_fmt'] >= '2010-02-01') & (train_raw['Date_fmt'] <'2011-03-01')
# df_train = train_raw.loc[mask]
# df_test = train_raw.loc[~mask]

# save to csv file
# df_train.to_csv('train_ini.csv', index =False)
# df_test.loc[:, df_test.columns != 'Weekly_Sales'].to_csv('test.csv', index = False)


