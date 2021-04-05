import os
import logging
from copy import deepcopy
import warnings
import datetime
import time
import threading

from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso


def start_multi_threading(thread_list,
                          max_threads=20,
                          all_threads_have_to_be_success=True):
    if len(thread_list) == 0:
        return None
    
    for count_i, thread in enumerate(thread_list):
        thread.daemon = True
        thread.start()
        logging.info("""Thread {thread_id} started""".format(thread_id=thread.name))
        
        while len([thread_i for thread_i in thread_list if thread_i.is_alive()]) >= max_threads:
            time.sleep(0.05)
            logging.debug('You have run too many threads! Have a rest!!')
    
    for thread in thread_list:
        
        thread.join()
        
        if all_threads_have_to_be_success and \
                'exitcode' in dir(thread) and \
                thread.exitcode == 1:
            
            raise threading.ThreadError(f"{thread.name} failed")
        else:
            logging.info(f"{thread.name} Done")



# data processing - change date fto datetime format and get year, month and week indicator
# df_train['Date'] = pd.to_datetime(df_train['Date'])

def encode_calender(data, max_week=None):
    """
    SOME HOLIDAY ALWAYS HAPPEN IN CERTAIN WEEK IN TRAINING DATA
    THUS DELETE THE WEEKLY DUMMY THERE, AS HOLIDAY MAY HAPPEN IN ANOTHER WEEK ~~~

    :param data:
    :param max_week:
    :return:
    """
    data.loc[:, 'Week'] = data['Date'].dt.isocalendar().week
    if max_week is None:
        max_week = data['Week'].max()
    data.loc[:, 'Year'] = data['Date'].dt.isocalendar().year
    data.loc[:, 'Month'] = pd.DatetimeIndex(data['Date']).month
    # data['Week'] = data.apply(lambda x: x['Week'] - 1 if x['Year'] == 2010 else x['Week'], axis=1)
    # dummy for holiday
    
    list_holiday_names = []
    holiday_list = [2, 9, 11, 12]
    for i, m in zip(range(1, 5), holiday_list):
        name = 'holiday' + str(i)
        data.loc[:, name] = (data['IsHoliday'] & (data['Month'] == m)).apply(lambda x: 1 if x else 0)
        list_holiday_names.append(name)
    
    # dummy for week
    for week_i in range(1, max_week):
        week_name = 'wk' + str(week_i)
        data[week_name] = (data['Week'] == week_i).apply(lambda x: 1 if x else 0)
        for holiday_name in list_holiday_names:
            #     SOME HOLIDAY ALWAYS HAPPEN IN CERTAIN WEEK IN TRAINING DATA
            #     THUS DELETE THE WEEKLY DUMMY THERE, AS HOLIDAY MAY HAPPEN IN ANOTHER WEEK ~~~
            if week_name in data and (data[holiday_name] == data[week_name]).sum() == len(data):
                del data[week_name]
    #
    # for week_name in list_week_names:
    #     for holiday_name in list_holiday_names:
    #         holiday_week_ind = (data[holiday_name] & data[week_name])
    #         if holiday_week_ind.sum() > 0:
    #             data.loc[:, holiday_name + '-' + week_name] = holiday_week_ind
    #
    # add a trend variable
    data['Week_count'] = data['Week'] + (data['Year'] - START_YEAR) * 52
    
    return data


def encode_cate(train_, next_folder_, cate_vars=('Store', 'Dept')):
    for cat_var_ in cate_vars:
        for _id in set(train_[cat_var_].unique().tolist() + next_folder_[cat_var_].unique().tolist()):
            for _df in [train_, next_folder_]:
                _df.loc[:, f'{cat_var_}_{_id}'] = (_df[cat_var_] == _id).apply(lambda x: 1 if x else 0)
    
    return train_, next_folder_


def eva_error(next_fold_, test_pred_, abs=True):
    # extract predictions matching up to the current fold
    scoring_df = next_fold_.merge(test_pred_, on=['Date', 'Store', 'Dept'], how='left', indicator=True)
    
    # extract weights and convert to numpy arrays for wae calculation
    weights = scoring_df['IsHoliday_x'].apply(lambda is_holiday: 5 if is_holiday else 1).to_numpy()
    actuals = scoring_df['Weekly_Sales'].to_numpy()
    preds = scoring_df['Weekly_Pred'].fillna(0).to_numpy()
    if abs:
        error = np.sum(weights * np.abs(actuals - preds)) / np.sum(weights).item()
    else:
        error = np.sum(weights * (actuals - preds)) / np.sum(weights).item()
    return error


def _predict(list_storage, _df_test, _df_train, x_list, y):
    reg = Lasso(alpha=0.001, normalize=True, max_iter=10000).fit(_df_train[x_list], y)
    _df_test.loc[:, 'Weekly_Pred'] = reg.predict(_df_test[x_list])
    list_storage.append(_df_test)


def mypredict(train, test, next_fold, t, show_training_error=False, target_fold=None, thread_n=1):
    """
    
    :param train:
    :param test:
    :param next_fold:
    :param t:
        t = 1
    :param show_training_error:
        show_training_error = True
    :return:
    """
    if next_fold is None:
        new_train = deepcopy(train)
    else:
        new_train = pd.concat([train, next_fold])
        new_train = new_train.drop_duplicates(['Date', 'Store', 'Dept'])
    
    test_start_dt = datetime.datetime(2011, 3, 1) + relativedelta(months=2 * (t - 1))
    test_end_dt = datetime.datetime(2011, 3, 1) + relativedelta(months=2 * t)
    
    new_test = test[(test['Date'] >= test_start_dt) & (test['Date'] < test_end_dt)]
    
    if target_fold is not None and t != target_fold:
        new_test['Weekly_Pred'] = 0
        return new_train, new_test[['Date', 'Store', 'Dept', 'Weekly_Pred', 'IsHoliday']]
    
    df_train = deepcopy(new_train)
    df_train = encode_calender(data=df_train)
    df_new_test = encode_calender(data=deepcopy(new_test), max_week=df_train['Week'].max())
    
    x_list = [col for col in set(list(df_train.columns) + list(df_new_test.columns))
              if col not in ['Date', 'Store', 'Dept',
                             'Weekly_Sales', 'Weekly_Sales_lag', 'diff',
                             'IsHoliday', 'Week', 'Month']]
    
    # make both train and test have the same variables
    for x in x_list:
        if x not in df_new_test:
            df_new_test[x] = 0
        if x not in df_train:
            df_train[x] = 0
    
    LIST_DF_PRED = []
    list_thread = []
    for dept, _df_one_dept in df_new_test.groupby('Dept'):
        for store, _df in _df_one_dept.groupby('Store'):
            logging.info(f' === Processing folder {t}: dept-{dept} and store-{store} ===')
            # dept = 23
            # store = 43
            df_new_test_one_dept = _df[(_df['Dept'] == dept) & (_df['Store'] == store)]
            df_train_one_dept = df_train[(df_train['Dept'] == dept) & (df_train['Store'] == store)]
            
            if df_train_one_dept.empty:
                df_train_one_dept = df_train[(df_train['Dept'] == dept)]
            
            y = df_train_one_dept['Weekly_Sales']
            if thread_n > 1:
                _thread = threading.Thread(target=_predict,
                                 kwargs={'list_storage': LIST_DF_PRED,
                                         'y': y,
                                         'x_list': x_list,
                                         '_df_test': df_new_test_one_dept,
                                         '_df_train': df_train_one_dept
                                         })
                
                list_thread.append(_thread)
                
            else:
                reg = Lasso(alpha=0.001, normalize=True, max_iter=10000).fit(df_train_one_dept[x_list], y)
                df_new_test_one_dept.loc[:, 'Weekly_Pred'] = reg.predict(df_new_test_one_dept[x_list])
                LIST_DF_PRED.append(df_new_test_one_dept)
                
                if show_training_error:
                    df_train_one_dept['Weekly_Pred'] = reg.predict(df_train_one_dept[x_list])
                    
                    training_error = eva_error(
                        next_fold_=df_train_one_dept[['Date', 'Store', 'Dept', 'Weekly_Sales', 'IsHoliday']],
                        test_pred_=df_train_one_dept[['Date', 'Store', 'Dept', 'Weekly_Pred', 'IsHoliday']]
                    )
                    logging.info(f'dept {dept} and store {store}: training_error: {training_error}')
        
    if list_thread:
        start_multi_threading(list_thread, max_threads=thread_n)
        
    test_pred = pd.concat(LIST_DF_PRED)[['Date', 'Store', 'Dept', 'Weekly_Pred', 'IsHoliday']]
    
    return new_train, test_pred


def set_logging(level=10,
                path=None):
    format = '%(levelname)s-%(name)s-%(funcName)s:\n %(message)s'
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    if path:
        logging.basicConfig(level=level, format=format, filename=path)
    else:
        logging.basicConfig(level=level, format=format)


warnings.simplefilter(action="ignore")

# use last year to predict

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 40)

# FOLDER = '/Users/fanyang/Dropbox/uiuc/cs598/UIUC_SPL/UIUC_PSL/Project2/'
FOLDER = '/Users/yafa/Dropbox/Library/UIUC_PSL/UIUC_PSL/Project2'

train = pd.read_csv(os.path.join(FOLDER, 'train_ini.csv'), parse_dates=['Date'])
# df_train.dtypes.value_counts()
test = pd.read_csv(os.path.join(FOLDER, 'test.csv'), parse_dates=['Date'])

START_YEAR = 2010


set_logging(level=20)

n_folds = 10
next_fold = None
wae = []

target_fold = 5

# time-series CV
for t in range(1, n_folds + 1):
    print(f'Fold{t}...')
    # t = 1
    # *** THIS IS YOUR PREDICTION FUNCTION ***
    train, test_pred = mypredict(train, test, next_fold, t, target_fold=target_fold)
    
    # Load fold file
    # You should add this to your training data in the next call to mypredict()
    fold_file = 'fold_{t}.csv'.format(t=t)
    next_fold = pd.read_csv(os.path.join(FOLDER, fold_file), parse_dates=['Date'])
    
    # extract predictions matching up to the current fold
    scoring_df = next_fold.merge(test_pred, on=['Date', 'Store', 'Dept'], how='left', indicator=True)
    
    # extract weights and convert to numpy arrays for wae calculation
    weights = scoring_df['IsHoliday_x'].apply(lambda is_holiday: 5 if is_holiday else 1).to_numpy()
    actuals = scoring_df['Weekly_Sales'].to_numpy()
    preds = scoring_df['Weekly_Pred'].fillna(0).to_numpy()
    
    weighted_error = np.sum(weights * np.abs(actuals - preds)) / np.sum(weights).item()
    
    print(f"error: {weighted_error}")
    
    wae.append(weighted_error)
    
    if target_fold is not None and t == target_fold:
        break

print(wae)
print(sum(wae) / len(wae))
