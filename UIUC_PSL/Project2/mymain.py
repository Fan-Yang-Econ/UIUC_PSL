import os
import logging
from copy import deepcopy
import warnings
import datetime

from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso

START_YEAR = 2010


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
    
    # known holidays by month
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
    data['Month_count'] = data['Month'] + (data['Year'] - START_YEAR) * 12
    
    return data


def encode_cate(train_, next_folder_, cate_vars=('Store', 'Dept')):
    """
    Not used
    
    :param train_:
    :param next_folder_:
    :param cate_vars:
    :return:
    """
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


def mypredict(train, test,
              next_fold, t,
              show_training_error=True,
              target_fold=None,
              dept_list=None,
              X_TO_EXCLUDE=('Date', 'Store', 'Dept',
                            'Weekly_Sales',
                            'Weekly_Sales_lag',
                            'diff',
                            'Week_count',
                            'Month_count',
                            'IsHoliday', 'Week', 'Month')):
    """
        t = 1
        show_training_error = True

    :param train:
    :param test:
    :param next_fold:
    :param t:
    :param show_training_error:
    :param target_fold:
    :param dept_list:
    :param X_TO_EXCLUDE:
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
    new_train = new_train[(new_train['Date'] < test_start_dt)]
    
    if target_fold is not None and t != target_fold:
        new_test['Weekly_Pred'] = 0
        return new_train, new_test[['Date', 'Store', 'Dept', 'Weekly_Pred', 'IsHoliday']]
    
    df_train = deepcopy(new_train)
    df_train = encode_calender(data=df_train)
    df_new_test = encode_calender(data=deepcopy(new_test), max_week=df_train['Week'].max())
    
    # add the Weekly_Sales_lag
    LAG_KEY = 'Week_count'
    LAG_CYCLE = 9
    LAR_VAR = 'Weekly_Sales_lag'
    PRIMARY_KEYS_FOR_LAG = ['Dept', 'Store', LAG_KEY]
    
    df_avg_week = df_train.groupby(PRIMARY_KEYS_FOR_LAG, as_index=False)['Weekly_Sales'].mean()
    df_avg_week[LAG_KEY] = df_avg_week[LAG_KEY] + LAG_CYCLE
    
    df_train = pd.merge(df_train, df_avg_week, on=PRIMARY_KEYS_FOR_LAG, suffixes=('', '_lag'), how='left')
    df_train.loc[df_train[LAR_VAR].isna(), LAR_VAR] = 0
    
    
    LIST_DF_PRED = []
    list_training_error = []
    
    if dept_list is not None:
        df_new_test = df_new_test[df_new_test['Dept'].isin(dept_list)]
    
    for dept, _df_one_dept in df_new_test.groupby('Dept'):
        for store, _df in _df_one_dept.groupby('Store'):
            # store = None
            logging.info(f' === Processing fold {t}: dept-{dept} and store-{store} ===')
            # dept = 92
            # store = 41
            
            df_new_test_one_dept = df_new_test[(df_new_test['Dept'] == dept)]
            df_train_one_dept = df_train[(df_train['Dept'] == dept) & (df_train['Store'] == store)]
            
            if store is not None:
                df_new_test_one_dept = df_new_test_one_dept[(df_new_test_one_dept['Store'] == store)]
            if store is not None:
                df_train_one_dept = df_train_one_dept[(df_train_one_dept['Store'] == store)]
            
            if df_train_one_dept.empty:
                df_train_one_dept = df_train[(df_train['Dept'] == dept)]
                if df_train_one_dept.empty:
                    if store is not None:
                        df_train_one_dept = df_train[(df_train['Store'] == store)]
                    else:
                        df_train_one_dept = deepcopy(df_train)
            
            x_list = [col for col in set(list(df_train_one_dept.columns))
                      if col not in X_TO_EXCLUDE]
            
            # make both train and test have the same variables
            x_to_delete = []
            
            for x in x_list:
                if df_train_one_dept[x].sum() == 0:
                    x_to_delete.append(x)
            
            for x in x_to_delete:
                x_list.remove(x)
            
            for x in x_list:
                if x not in df_new_test_one_dept:
                    df_new_test_one_dept[x] = 0
            
            if LAR_VAR not in X_TO_EXCLUDE:
                LAST_Y_SAME_MONTH_ind = ((df_train_one_dept['Date'] >= test_start_dt - relativedelta(years=1)) &
                                         (df_train_one_dept['Date'] <= test_end_dt - relativedelta(years=1)))
                
                df_train_one_dept.loc[~LAST_Y_SAME_MONTH_ind, LAR_VAR] = 0
                
                # df_train_one_dept.loc[df_train_one_dept['Weekly_Sales_lag'] > 0]
                
                if LAR_VAR in df_new_test_one_dept:
                    del df_new_test_one_dept[LAR_VAR]
                
                df_new_test_one_dept_with_lag_ = pd.merge(df_new_test_one_dept,
                                                          df_avg_week[PRIMARY_KEYS_FOR_LAG + ['Weekly_Sales']],
                                                          on=PRIMARY_KEYS_FOR_LAG,
                                                          suffixes=('', '_lag'))
                
                if len(df_new_test_one_dept_with_lag_) < len(df_new_test_one_dept) or (
                        df_train_one_dept.loc[LAST_Y_SAME_MONTH_ind, LAR_VAR] == 0).sum():
                    logging.info('Drop Weekly_Sales_lag due to missing data')
                    if LAR_VAR in x_list:
                        x_list.remove(LAR_VAR)
                else:
                    df_new_test_one_dept = df_new_test_one_dept_with_lag_.rename(columns={'Weekly_Sales': LAR_VAR})
            
            y = df_train_one_dept['Weekly_Sales']
            
            reg = Lasso(alpha=0.001, normalize=True, max_iter=10000).fit(df_train_one_dept[x_list], y)
            df_new_test_one_dept.loc[:, 'Weekly_Pred'] = reg.predict(df_new_test_one_dept[x_list])
            LIST_DF_PRED.append(df_new_test_one_dept)
            
            if show_training_error:
                df_train_one_dept['Weekly_Pred'] = reg.predict(df_train_one_dept[x_list])
                
                training_error = eva_error(
                    next_fold_=df_train_one_dept[['Date', 'Store', 'Dept', 'Weekly_Sales', 'IsHoliday']],
                    test_pred_=df_train_one_dept[['Date', 'Store', 'Dept', 'Weekly_Pred', 'IsHoliday']]
                )
                list_training_error.append({f'dept {dept} and store {store}': training_error})
                logging.info(f'dept {dept} and store {store}: training_error: {training_error}')
    
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



def _run():
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
    
    start = datetime.datetime.now()
    set_logging(level=20)
    
    n_folds = 10
    next_fold = None
    wae = []
    
    target_fold = None
    # target_fold = None
    # dept_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 29, 92]
    # dept_list = [39]
    # dept_list = [92]
    dept_list = None
    
    # time-series CV
    for t in range(1, n_folds + 1):
        print(f'Fold {t}...')
        # t = 1
        # *** THIS IS YOUR PREDICTION FUNCTION ***
        train, test_pred = mypredict(train, test, next_fold, t)
        
        # Load fold file
        # You should add this to your training data in the next call to mypredict()
        fold_file = 'fold_{t}.csv'.format(t=t)
        next_fold = pd.read_csv(os.path.join(FOLDER, fold_file), parse_dates=['Date'])
        
        # extract predictions matching up to the current fold
        scoring_df = next_fold.merge(test_pred, on=['Date', 'Store', 'Dept'], how='left', indicator=True)
        if dept_list is not None:
            scoring_df = scoring_df[scoring_df['Dept'].isin(dept_list)]
        if scoring_df['Weekly_Pred'].isna().sum() > 0:
            raise KeyError('Missing predictions')
        
        # extract weights and convert to numpy arrays for wae calculation
        weights = scoring_df['IsHoliday_x'].apply(lambda is_holiday: 5 if is_holiday else 1).to_numpy()
        actuals = scoring_df['Weekly_Sales'].to_numpy()
        preds = scoring_df['Weekly_Pred'].fillna(0).to_numpy()
        
        weighted_error = np.sum(weights * np.abs(actuals - preds)) / np.sum(weights).item()
        
        print(f"error: {weighted_error}")
        
        wae.append(weighted_error)
        
        if target_fold is not None and t == target_fold:
            break

    end_time = datetime.datetime.now()
    
    print(wae)
    print(sum(wae) / len(wae))

    print(start)
    print(end_time)

    # Ford 5: all depts
    # 2341 vs 2337
    
    # Ford 6
    # BAD
    # Ford 7
    # BAD
    
    #
    # # Simple model peformance
    # [2053.167060130686, 1476.5989977618594, 1459.056801047943, 1598.325720981124, 2340.961897031291, 1679.7237542894243, 1725.9148932138892, 1431.0930879612342, 1446.63182487708, 1445.6296619697382]
    #
    # # Complex model: primary key as dept + lag
    # [2053.4433733368887, 2025.2563612760553, 8272.399578392624, 1583.508884298707, 2337.0325496496944, 1714.9412744997078, 1827.0368548483643, 1562.0488984789363, 1639.4062452123544, 1539.6111069302083]
    #
    # # Complex model: primary key as dept + store + lag
    # [2057.112553936628, 1620.8506756163936, 1559.2715135095202, 1578.7301956534338, 2407.518196450271, 1778.071222342405, 1869.398703968537, 1601.850124848687, 1647.4025702722702, 1534.168132376286]
    #
    #
    # pd.Series([2053.167060130686, 1476.5989977618594, 1459.056801047943, 1598.325720981124, 2340.961897031291, 1679.7237542894243, 1725.9148932138892, 1431.0930879612342, 1446.63182487708, 1445.6296619697382]).mean()
    # pd.Series([2053.167060130686, 1476.5989977618594, 1459.056801047943, 1583.508884298707, 2337.0325496496944, 1679.7237542894243, 1725.9148932138892, 1431.0930879612342, 1446.63182487708, 1445.6296619697382]).mean()