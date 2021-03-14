import math
# import argparse
import os
from copy import deepcopy
# import logging
import numpy as np

import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import StandardScaler
import xgboost as xgb
# import matplotlib
# import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


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
        lambda logged_y: math.exp(logged_y)
        # if logged_y > 0 else min(df_train[Y_COL_NAME])
    )

    return predictions

Y = 'Sale_Price'
FOLDER = '/Users/fanyang/Dropbox/uiuc/cs598/UIUC_SPL/UIUC_PSL/Project1/'

df_train = pd.read_csv(os.path.join(FOLDER, "train.csv"))
df_train = df_train.reindex()


df_test = pd.read_csv(os.path.join(FOLDER, "test.csv"))
df_test = df_test.reindex()


df_test_y = pd.read_csv(os.path.join(FOLDER, "test_y.csv"))
df_test_y = df_test_y.reindex()

logged_y_training = log_y(df_train[Y])

# for outliers, replace extreme values
var_outlier = ['Lot_Area','Mas_Vnr_Area','BsmtFin_SF_2','Bsmt_Unf_SF','Total_Bsmt_SF',
 'First_Flr_SF','Low_Qual_Fin_SF','Gr_Liv_Area',
 'Lot_Frontage', 'Second_Flr_SF',
               
               # 'Bsmt_Half_Bath','Bedroom_AbvGr',
               # 'Kitchen_AbvGr','TotRms_AbvGrd',
 'Garage_Area','Wood_Deck_SF','Open_Porch_SF',
 'Enclosed_Porch','Three_season_porch','Screen_Porch','Misc_Val']
#


def clean_data(df_data, var_outlier, training=False):
    # df_data['Garage_Yr_Blt'] = df_data['Garage_Yr_Blt'].fillna(0)

    if training:

        for col in var_outlier:
            cap_value = df_data[col].quantile(0.98)
            if cap_value > 0:
                df_data[col] = df_data[col].apply(lambda x: cap_value if x > cap_value else x)




# feature engineer
def new_feature(df):
    df['yr_since_built'] = df['Year_Sold'] - df['Year_Built']
    df['yr_since_remod'] = df['Year_Sold'] - df['Year_Remod_Add']
    df['remod_ind'] = np.where(df['Year_Remod_Add'] == df['Year_Built'], 1, 0)
    df['new_house'] = np.where(df['Year_Sold'] == df['Year_Built'], 1, 0)

    df['sold_2006'] = np.where(df['Year_Sold'] == 2006, 1, 0)
    df['sold_2007'] = np.where(df['Year_Sold'] == 2007, 1, 0)
    df['sold_2008'] = np.where(df['Year_Sold'] == 2008, 1, 0)
    df['sold_2009'] = np.where(df['Year_Sold'] == 2009, 1, 0)
    df['sold_2010'] = np.where(df['Year_Sold'] == 2010, 1, 0)
    df['total_bath'] = df['Full_Bath'] + 0.5 * df['Half_Bath'] + df['Bsmt_Full_Bath'] + 0.5 * df['Bsmt_Half_Bath']

    df['grp_neighbor_1'] = df.apply(
        lambda x: 1 if x['Neighborhood'] in ['Iowa_DOT_and_Rail_Road', 'Meadow_Village', 'Briardale'] else 0, axis=1)

    df['grp_neighbor_2'] = df.apply(lambda x: 1 if x['Neighborhood'] in ['Brookside',
                                                                         'Old_Town', 'Edwards',
                                                                         'South_and_West_of_Iowa_State_University',
                                                                         'Sawyer', 'Northpark_Villa', 'North_Ames',
                                                                         'Blueste', 'Mitchell'] else 0, axis=1)

    df['grp_neighbor_3'] = df.apply(
        lambda x: 1 if x['Neighborhood'] in ['Sawyer_West', 'Northwest_Ames', 'Gilbert', 'College_Creek',
                                             'Bloomington_Heights', 'Crawford', 'Greens', 'Clear_Creek', 'Somerset',
                                             'Timberland', 'Veenker'] else 0, axis=1)

    df['grp_neighbor_4'] = df.apply(
        lambda x: 1 if (x['grp_neighbor_1'] == 0) and (x['grp_neighbor_2'] == 0) and (x['grp_neighbor_3'] == 0) else 0,
        axis=1)


# clean_data(df_test, var_outlier, training=False)


# remove some categorical data
remove_var = ['Street', 'Utilities', 'Condition_2', 'Roof_Matl', 'Heating',
              'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude','Latitude',
              Y, 'PID'
              ,'Garage_Yr_Blt'
    #           ,'TotRms_AbvGrd',,'Garage_Area'
    #           ,'Neighborhood'
    #           , 'Year_Sold', 'Year_Built', 'Year_Remod_Add','Half_Bath','Full_Bath','Bsmt_Full_Bath','Bsmt_Half_Bath'
    # , 'Garage_Qual','Garage_Type','Garage_Finish'

              ]

select_var = [i for i in df_train.columns if i not in remove_var]
clean_data(df_train,var_outlier,training=True)
new_feature(df_train)
new_feature(df_test)

df_train_numeric, dict_one_hot_encoder = transform_category_vars(df=df_train[select_var])
df_test_numeric = transform_category_vars_test(df_test[select_var], dict_one_hot_encoder)
# missing value
# df_test_numeric['Garage_Yr_Blt'] = df_test_numeric['Garage_Yr_Blt'].fillna(0)

before_vars =  select_var
after_vars = df_train_numeric.columns
# dummy_vars = [col for col in after_vars if col not in select_var]
#
# sparse_var_list= []
# # ratio_list =[]
# n = df_train_numeric.shape[0]
# for var in dummy_vars:
#     count_ind = df_train_numeric[var].sum()
#     ratio = 1.0*count_ind/n
#     # ratio_list.append(ratio)
#     if ratio < 0.02:
#         # print(ratio)
#         sparse_var_list.append(var)
#
# final_candidate = [col for col in df_train_numeric.columns if col not in sparse_var_list]

final_candidate = after_vars
X = df_train_numeric[final_candidate]
y = logged_y_training


# cross validation for alpha selection

lasso = linear_model.Lasso(fit_intercept=True,normalize=True,random_state=0, max_iter=10000)
alphas = np.logspace(-6, 0.1, 20)

tuned_parameters = [{'alpha': alphas}]
n_folds = 5
clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False, scoring='neg_mean_squared_error')
clf.fit(X , y)
# clf.cv_results_['mean_test_score']
best_alpha = clf.best_params_['alpha']
print(best_alpha)


model = linear_model.Lasso(fit_intercept=True,normalize=True, alpha = best_alpha,max_iter=100000)
model.fit(X , y)

predicted_y = predict(model, new_data=X)
print(error_evaluation(predicted_y, true_y=df_train[Y]))
predicted_y = predict(model, new_data=df_test_numeric[final_candidate])
print(error_evaluation(predicted_y, true_y=df_test_y[Y]))


df_model_coef = pd.DataFrame(model.coef_, index = X.columns, columns=['coef']).sort_values('coef', ascending=False)
# df_model_coef.to_csv('lasso_coef.csv')

# ridge
lasso_var = df_model_coef[abs(df_model_coef['coef'])>0].index.tolist()

ridge = linear_model.Ridge(fit_intercept=True,normalize=True,random_state=0, max_iter=10000)
alphas = np.logspace(-5, 0.2, 20)

tuned_parameters = [{'alpha': alphas}]
n_folds = 5
clf = GridSearchCV(ridge, tuned_parameters, cv=n_folds, refit=False, scoring='neg_mean_squared_error')
clf.fit(X[lasso_var] , y)
# clf.cv_results_['mean_test_score']
best_alpha_ridge = clf.best_params_['alpha']
print(best_alpha_ridge)

RidgeModel = linear_model.Ridge(fit_intercept=True,normalize=True, alpha = best_alpha_ridge,max_iter=100000)
RidgeModel.fit(X[lasso_var] , y)

predicted_y = predict(RidgeModel, new_data=X[lasso_var])
print(error_evaluation(predicted_y, true_y=df_train[Y]))
predicted_y = predict(RidgeModel, new_data=df_test_numeric[lasso_var])
print(error_evaluation(predicted_y, true_y=df_test_y[Y]))



#
#
# RFmodel = RandomForestRegressor(max_depth = 10, n_estimators = 200)
# RFmodel.fit(df_train_numeric, logged_y_training)
#
# predicted_y = predict(RFmodel, new_data=df_train_numeric)
# print(error_evaluation(predicted_y, true_y=df_train[Y]))
# predicted_y = predict(RFmodel, new_data=df_test_numeric)
# print(error_evaluation(predicted_y, true_y=df_test_y[Y]))
#
# #
# importance = RFmodel.feature_importances_
# # summarize feature importance
#
# feature_importances = pd.DataFrame(RFmodel.feature_importances_,
#                                    index = df_train_numeric.columns,
#                                    columns=['importance']).sort_values('importance', ascending=False)
#
# feature_importances.head(50)