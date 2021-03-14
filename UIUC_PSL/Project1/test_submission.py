"""
Run 10 times of the model, on 10 different training/test data pairs
"""
import os
from pprint import pprint
from datetime import datetime

import pandas as pd

# from UIUC_PSL.Project1.mymain import transform_category_vars, Y, LassoModel, BoostingTreeMode, error_evaluation, transform_category_vars_test

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 40)

import math
# import argparse
import os
from copy import deepcopy
# import logging
import numpy as np

import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# from sklearn.preprocessing import StandardScaler
# import xgboost as xgb
# import matplotlib
# import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

def _get_one_hot_encoder_df(_col_name, df_train):
    """
    hot code for one single category vars

    :param _col_name:
    :param df_train:
    :return:
    """

    enc = OneHotEncoder(handle_unknown='ignore')
    _cate_matrix = [i[[_col_name]] for row_i, i in df_train.iterrows()]
    enc.fit(_cate_matrix)
    df_hot_code = pd.DataFrame(enc.transform(_cate_matrix).toarray())
    df_hot_code.columns = [_col_name + '__' + str(i) for i in df_hot_code.columns]

    return {'data_frame': df_hot_code, "encode": enc}


def _get_one_hot_encoder_df_test(_col_name, df_test, dict_one_hot_encoder):
    """
    hot code for one single category vars

    :param _col_name:
    :param df_train:
    :return:
    """
    enc = dict_one_hot_encoder[_col_name]
    _cate_matrix = [i[[_col_name]] for row_i, i in df_test.iterrows()]
    df_hot_code = pd.DataFrame(enc.transform(_cate_matrix).toarray())
    df_hot_code.columns = [_col_name + '__' + str(i) for i in df_hot_code.columns]

    return df_hot_code


def transform_category_vars(df, Y=Y):
    """
    hot code for all category vars
    :param df:
    :return:
    """

    LIST_X = [i for i in df.columns if i not in [Y, 'PID']]
    LIST_CATEGORY_X = []
    LIST_NUMERIC_C = []

    for x in LIST_X:
        if  df[x].dtypes != 'object':

            LIST_NUMERIC_C.append(x)
        else:
            LIST_CATEGORY_X.append(x)
            # len(LIST_CATEGORY_X)
            # len(LIST_NUMERIC_C)

    dict_one_hot_encoder_df = {}
    dict_one_hot_encoder = {}
    for _col_name in LIST_CATEGORY_X:
        # print(_col_name)
        encode_result = _get_one_hot_encoder_df(_col_name, df)
        dict_one_hot_encoder_df[_col_name] = encode_result["data_frame"]
        dict_one_hot_encoder[_col_name] = encode_result["encode"]

    df_numeric = deepcopy(df[LIST_X])
    for _col_name in LIST_CATEGORY_X:
        del df_numeric[_col_name]
        for new_col in dict_one_hot_encoder_df[_col_name]:
            df_numeric[new_col] = dict_one_hot_encoder_df[_col_name][new_col]

    return (df_numeric, dict_one_hot_encoder)


def transform_category_vars_test(df_test,  dict_one_hot_encoder, Y=Y):
    """
    hot code for all category vars
    :param df:
    :return:
    """

    LIST_X = [i for i in df_test.columns if i not in [Y, 'PID']]
    LIST_CATEGORY_X = []
    LIST_NUMERIC_C = []

    for x in LIST_X:
        if  df_test[x].dtypes != 'object':

            LIST_NUMERIC_C.append(x)
        else:
            LIST_CATEGORY_X.append(x)
            # len(LIST_CATEGORY_X)
            # len(LIST_NUMERIC_C)

    dict_one_hot_encoder_df = {}
    # dict_one_hot_encoder = {}
    for _col_name in LIST_CATEGORY_X:
        dict_one_hot_encoder_df[_col_name] = _get_one_hot_encoder_df_test(_col_name, df_test, dict_one_hot_encoder)

    df_numeric = deepcopy(df_test[LIST_X])
    for _col_name in LIST_CATEGORY_X:
        del df_numeric[_col_name]
        for new_col in dict_one_hot_encoder_df[_col_name]:
            df_numeric[new_col] = dict_one_hot_encoder_df[_col_name][new_col]

    return df_numeric


def log_y(y_series):
    """
    Log the housing price
    :param y_series:
    :return:
    """
    return y_series.apply(lambda y: math.log(y))


def error_evaluation(predicted_y, true_y):
    tmp =((
                    pd.Series(predicted_y).apply(lambda y: math.log(y)) -
                    pd.Series(true_y).apply(lambda y: math.log(y))
            ) ** 2).mean()

    return np.sqrt(tmp)

def predict(model, new_data, Y_COL_NAME=Y):
    # df_new_data_numeric = standardize_df(new_data)
    predictions = model.predict(new_data)
    predictions = pd.Series(predictions).apply(
        lambda logged_y: math.exp(logged_y) if logged_y > 0 else min(df_train[Y_COL_NAME]))

    return predictions

Y = 'Sale_Price'


def prepare_data(df_ames, TEST_ID, FOLDER, str_testID, write_to_csv=True):
    # prepare data
    
    df_test_id = pd.DataFrame([i.strip().split() for i in str_testID])
    df_test_id[TEST_ID] = df_test_id[TEST_ID].apply(lambda x: int(x))
    
    df_test_full = df_ames[df_ames['PID'].index.isin(df_test_id[TEST_ID])]
    df_train_full = df_ames[~df_ames['PID'].index.isin(df_test_id[TEST_ID])]
    
    df_train_full = df_train_full.reset_index()
    df_test_full = df_test_full.reset_index()
    
    if write_to_csv:
        df_train_full.to_csv(os.path.join(FOLDER, "train.csv"), index=False)
        df_test_full[[i for i in df_test_full.columns if i != 'Sale_Price']].to_csv(os.path.join(FOLDER, "test.csv"), index=False)
        df_test_full[[i for i in df_test_full.columns if i == 'Sale_Price']].to_csv(os.path.join(FOLDER, "test_y.csv"), index=False)
    
    return {'df_test': df_test_full, 'df_train': df_train_full}


start_time = datetime.now()

# FOLDER = '/Users/yafa/Dropbox/Library/UIUC_PSL/UIUC_PSL/Project1/'
FOLDER = '/Users/fanyang/Dropbox/uiuc/cs598/UIUC_SPL/UIUC_PSL/Project1/'
#
df_ames = pd.read_csv(os.path.join(FOLDER, 'Ames_data.csv'))
# df_ames_numeric = transform_category_vars(df_ames)
#
# df_ames_numeric['PID'] = df_ames['PID']
# df_ames_numeric[Y] = df_ames[Y]
# list_x = [i for i in df_ames_numeric.columns if i not in ['PID', Y]]

with open(os.path.join(FOLDER, 'project1_testIDs.dat')) as f:
    str_testID = f.readlines()

list_result = []

for TEST_ID in range(0, 10):
    # TEST_ID = 6
    DICT_DATA = prepare_data(df_ames, TEST_ID, FOLDER, write_to_csv=False, str_testID=str_testID)
    # DICT_DATA = prepare_data(df_ames_numeric, TEST_ID, FOLDER, write_to_csv=False, str_testID=str_testID)
    df_train = DICT_DATA['df_train']
    df_test = DICT_DATA['df_test']

    
    df_train_numeric = transform_category_vars(df=df_train)
    df_test_numeric = transform_category_vars(df=df_test)
    
    common_vars = set(df_test_numeric.columns).intersection(set(df_train_numeric.columns))
    
    df_train_numeric = df_train_numeric[common_vars]
    df_test_numeric = df_test_numeric[common_vars]
    
    for model_cls in [LassoModel, BoostingTreeMode]:
        model_obj = model_cls(df_train=df_train_numeric, y_series=df_train[Y])
        model_obj.train()
        
        # self=model_obj
        
        list_result.append({
            'test_id': TEST_ID,
            'testing_error': error_evaluation(model_obj.predict(new_data=df_test_numeric), df_test[Y]),
            'train_error': error_evaluation(model_obj.predict(new_data=df_train_numeric), df_train[Y]),
            'model_name': model_cls.__name__
        })
    
    #
    # df_train_numeric, dict_one_hot_encoder = transform_category_vars(df=df_train)
    # df_test_numeric = transform_category_vars_test(df_test, dict_one_hot_encoder)


    var_outlier = ['Lot_Area', 'Mas_Vnr_Area', 'BsmtFin_SF_2', 'Bsmt_Unf_SF', 'Total_Bsmt_SF',
                   'First_Flr_SF', 'Low_Qual_Fin_SF', 'Gr_Liv_Area', 'Bsmt_Half_Bath',
                   'Bedroom_AbvGr', 'Kitchen_AbvGr', 'TotRms_AbvGrd', 'Garage_Area', 'Wood_Deck_SF', 'Open_Porch_SF',
                   'Enclosed_Porch', 'Three_season_porch', 'Screen_Porch', 'Misc_Val']


    def clean_data(df_data, var_outlier, training=False):
        df_data['Garage_Yr_Blt'] = df_data['Garage_Yr_Blt'].fillna(0)

        if training:

            for col in var_outlier:
                cap_value = df_data[col].quantile(0.95)
                if cap_value > 0:
                    df_data[col].apply(lambda x: cap_value if x > cap_value else x)


    clean_data(df_train, var_outlier, training=True)
    clean_data(df_test, var_outlier, training=False)

    remove_var = ['Street', 'Utilities', 'Condition_2', 'Roof_Matl', 'Heating',
                  'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude', 'Latitude',
                  Y, 'PID']
    select_var = [i for i in df_train.columns if i not in remove_var]

    df_train_numeric, dict_one_hot_encoder = transform_category_vars(df=df_train[select_var])
    df_test_numeric = transform_category_vars_test(df_test[select_var], dict_one_hot_encoder)
    # missing value


    logged_y_training = log_y(df_train[Y])

    # cross validation for alpha selection
    lasso = linear_model.Lasso(fit_intercept=True, normalize=True, random_state=0, max_iter=10000)
    alphas = np.logspace(-6, 0.5, 20)

    tuned_parameters = [{'alpha': alphas}]
    n_folds = 5
    clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False, scoring='neg_mean_squared_error')
    clf.fit(df_train_numeric, logged_y_training)
    # scores = clf.cv_results_['mean_test_score']
    # scores_std = clf.cv_results_['std_test_score']
    best_alpha = clf.best_params_['alpha']
    print(best_alpha)

    model = linear_model.Lasso(
        fit_intercept=True,
        normalize=True,
        alpha=best_alpha,
        max_iter=10000
    )

    model.fit(df_train_numeric, logged_y_training)
    predicted_y = predict(model, new_data=df_train_numeric)
    print(error_evaluation(predicted_y, true_y=df_train[Y]))

    predicted_y = predict(model, new_data=df_test_numeric)
    print(error_evaluation(predicted_y, true_y=df_test[Y]))

    # for model_cls in [LassoModel]:
    #     model_obj = model_cls(df_train=df_train_numeric, y_series=df_train[Y])
    #     model_obj.train()
    #
    #     # self=model_obj
    #
    #     list_result.append({
    #         'test_id': TEST_ID,
    #         'testing_error': error_evaluation(model_obj.predict(new_data=df_test_numeric), df_test[Y]),
    #         'train_error': error_evaluation(model_obj.predict(new_data=df_train_numeric), df_train[Y]),
    #         # 'model_name': model_cls.__name__
    #     })


pprint(list_result)

df_errors = pd.DataFrame(list_result)

for model in ['BoostingTreeMode', 'LassoModel']:
    print(model)
    print(df_errors[df_errors['model_name'] == model]['testing_error'].apply(lambda x: int(x * 1000) / 1000).tolist())
    print(df_errors[df_errors['model_name'] == model]['testing_error'].mean())

df_errors[df_errors['model_name'] == 'LassoModel']['testing_error'].mean()
df_errors[df_errors['model_name'] == 'BoostingTreeMode']['testing_error'].mean()

end_time = datetime.now()
print(start_time, '---', end_time)
