import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import datetime
from dateutil.relativedelta import relativedelta
# use last year to predict
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
    # data['Week'] = data.apply(lambda x: x['Week'] - 1 if x['Year'] == 2010 else x['Week'], axis=1)
    # dummy for holiday
    holiday_list = [2, 9, 11, 12]
    for i, m in zip(range(1, 5), holiday_list):
        name = 'holiday' + str(i)
        data[name] = data.apply(lambda x: 1 if x['IsHoliday'] == True and x['Month'] ==m else 0, axis =1)
    # dummy for week
    for i in range(1, 52):
        name = 'wk' + str(i)
        data[name] = data.apply(lambda x: 1 if x['Week'] == i else 0, axis=1)
    return data
#
# def _get_one_hot_encoder_df(_col_name, df_train):
#     enc = OneHotEncoder(handle_unknown='ignore')
#     _cate_matrix = [i[[_col_name]] for row_i, i in df_train.iterrows()]
#     enc.fit(_cate_matrix)
#     df_hot_code = pd.DataFrame(enc.transform(_cate_matrix).toarray())
#     df_hot_code.columns = [_col_name + '__' + str(i) for i in df_hot_code.columns]
#     return {'data_frame': df_hot_code, "encode": enc}
#
# def _get_one_hot_encoder_df_test(_col_name, df_test, dict_one_hot_encoder):
#     enc = dict_one_hot_encoder[_col_name]
#     _cate_matrix = [i[[_col_name]] for row_i, i in df_test.iterrows()]
#     df_hot_code = pd.DataFrame(enc.transform(_cate_matrix).toarray())
#     df_hot_code.columns = [_col_name + '__' + str(i) for i in df_hot_code.columns]
#     return df_hot_code
#
#
# def transform_category_vars(df):
#
#     dict_one_hot_encoder_df = {}
#     dict_one_hot_encoder = {}
#     LIST_X = ['Store', 'Dept']
#     for _col_name in LIST_X:
#         encode_result = _get_one_hot_encoder_df(_col_name, df)
#         dict_one_hot_encoder_df[_col_name] = encode_result["data_frame"]
#         dict_one_hot_encoder[_col_name] = encode_result["encode"]
#     for _col_name in LIST_X:
#         # del df[_col_name]
#         for new_col in dict_one_hot_encoder_df[_col_name]:
#             df[new_col] = dict_one_hot_encoder_df[_col_name][new_col]
#
#     return (df, dict_one_hot_encoder)
#
#
# def transform_category_vars_test(df_test,  dict_one_hot_encoder):
#     LIST_X = ['Store', 'Dept']
#     dict_one_hot_encoder_df = {}
#     # dict_one_hot_encoder = {}
#     for _col_name in LIST_X:
#         dict_one_hot_encoder_df[_col_name] = _get_one_hot_encoder_df_test(_col_name, df_test, dict_one_hot_encoder)
#
#
#     for _col_name in LIST_X:
#         # del df_test[_col_name]
#         for new_col in dict_one_hot_encoder_df[_col_name]:
#             df_test[new_col] = dict_one_hot_encoder_df[_col_name][new_col]
#
#     return df_test
# df_train_numeric, dict_one_hot_encoder = transform_category_vars(df_train)
# df_test_numeric = transform_category_vars_test(df_test, dict_one_hot_encoder)

# x_list = [col for col in df_train_numeric.columns if col not in ['Date', 'Store', 'Dept', 'Weekly_Sales', 'IsHoliday', 'Week', 'Month']]
# y = df_train_numeric['Weekly_Sales']
# reg = LinearRegression().fit(df_train_numeric[x_list], y)
# df_test_numeric['Weekly_Pred'] = reg.predict(df_test_numeric[x_list])
# X = df_train_numeric[x_list]
#
# lasso = linear_model.Lasso(fit_intercept=True,normalize=True,random_state=0, max_iter=10000)
# alphas = np.logspace(-6, 0.1, 20)
#
# tuned_parameters = [{'alpha': alphas}]
# n_folds = 5
# clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False, scoring='neg_mean_squared_error')
# clf.fit(X , y)
# # clf.cv_results_['mean_test_score']
# best_alpha = clf.best_params_['alpha']
# print(best_alpha)




df_train = data_clean(train)
df_test = data_clean(test)
t =1

delta_m = 2*(t-1)
start_date = pd.Timestamp(datetime.date(2011, 3, 1) + relativedelta(months=delta_m))
end_date = pd.Timestamp(datetime.date(2011, 5, 1) + relativedelta(months=delta_m))
mask = (df_test['Date'] >= start_date) & (df_test['Date'] <end_date)
current_test = df_test[mask]
df_test['Weekly_Pred'] = 0.0

store_list =  current_test['Store'].unique().tolist()
for store in store_list:
    for dept in current_test[current_test['Store'] == store]['Dept'].unique().tolist():
        mask = (df_train['Store'] == store) & (df_train['Dept'] == dept)
        current_train = df_train[mask]
        y = current_train['Weekly_Sales']
        x_list = [col for col in df_train.columns if col not in ['Date', 'Store', 'Dept', 'Weekly_Sales', 'IsHoliday', 'Week', 'Month']]
        X = current_train[x_list]
        reg = LinearRegression().fit(X, y)
        test_mask = (df_test['Store'] == store) & (df_test['Dept'] == dept)
        tmp_test = df_test[test_mask]
        df_test.loc[test_mask, 'Weekly_Pred'] = reg.predict(tmp_test[x_list])








def mypredict(train, test, next_fold, t):

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


