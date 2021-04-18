# Step 0: Load necessary libraries
###########################################
import math
import argparse
import os
from copy import deepcopy
import logging
import numpy as np
import pandas as pd
import xgboost as xgb_model

from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
<<<<<<< Updated upstream
from sklearn.model_selection import GridSearchCV

# import xgboost as xgb

=======
# import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# Data processing function
>>>>>>> Stashed changes
Y = 'Sale_Price'
BEST_ALPHA_RIDGE = 0.449398459072167


def set_logging(level=10,
                path=None):
    format = '%(levelname)s-%(name)s-%(funcName)s:\n %(message)s'

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if path:
        logging.basicConfig(level=level, format=format, filename=path)
    else:
        logging.basicConfig(level=level, format=format)


def _get_one_hot_encoder_df(_col_name, df_train, one_hot_encoder=None):
    """
    hot code for one single category vars

    :param _col_name:
    :param df_train:
    :return:
    """

    _cate_matrix = [i[[_col_name]] for row_i, i in df_train.iterrows()]

    if one_hot_encoder is None:
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        one_hot_encoder.fit(_cate_matrix)

    df_hot_code = pd.DataFrame(one_hot_encoder.transform(_cate_matrix).toarray())
    df_hot_code.columns = [_col_name + '__' + str(i) for i in df_hot_code.columns]

    return {'df': df_hot_code, "one_hot_encoder": one_hot_encoder}


def transform_category_vars(df, dict_one_hot_encoder=None, Y=Y):
    """
    hot code for all category vars
    :param df:
    :return:
    """

    LIST_X = [i for i in df.columns if i not in [Y, 'PID']]
    LIST_CATEGORY_X = []
    LIST_NUMERIC_C = []

    for x in LIST_X:
        if df[x].dtypes != 'object':
            LIST_NUMERIC_C.append(x)
        else:
            LIST_CATEGORY_X.append(x)

    logging.info(f'Number of category vars: `{len(LIST_CATEGORY_X)}`')
    logging.info(f'Number of numeric vars: `{len(LIST_NUMERIC_C)}`')

    dict_one_hot_encoder_all = {}
    # dict_one_hot_encoder = {}
    for _col_name in LIST_CATEGORY_X:
        logging.info(f'Encode `{_col_name}`')

        dict_one_hot_encoder_ = \
            _get_one_hot_encoder_df(_col_name,
                                    df,
                                    one_hot_encoder=dict_one_hot_encoder[_col_name][
                                        'one_hot_encoder'] if dict_one_hot_encoder is not None else None)

        dict_one_hot_encoder_all[_col_name] = dict_one_hot_encoder_

    df_numeric = deepcopy(df[LIST_X])

    for _col_name in LIST_CATEGORY_X:
        del df_numeric[_col_name]
        if _col_name in dict_one_hot_encoder_all:
            for new_col in dict_one_hot_encoder_all[_col_name]['df']:
                df_numeric[new_col] = dict_one_hot_encoder_all[_col_name]['df'][new_col]

    return df_numeric, dict_one_hot_encoder_all


def standardize_df(df_train_numeric, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df_train_numeric)

    df_train_numeric_stand = pd.DataFrame(scaler.transform(df_train_numeric))
    df_train_numeric_stand.columns = df_train_numeric.columns

    return {'df_train_numeric_stand': df_train_numeric_stand, 'scaler': scaler}


def log_y(y_series):
    """
    Log the housing price
    :param y_series:
    :return:
    """
    return y_series.apply(lambda y: math.log(y))


def error_evaluation(predicted_y, true_y):
    error_squared_sum = ((
                             pd.Series(predicted_y).apply(lambda y: math.log(y)) -
                             pd.Series(true_y).apply(lambda y: math.log(y))
                         ) ** 2).mean()

    return np.sqrt(error_squared_sum)


def clean_data(df, is_training_data=False):
    if is_training_data:
        # for outliers, replace extreme values
        # used boxplot

        VAR_OUTLIERS = ['Lot_Area',
                        'Mas_Vnr_Area',
                        'BsmtFin_SF_2',
                        'Bsmt_Unf_SF',
                        'Total_Bsmt_SF',
                        'First_Flr_SF',
                        'Low_Qual_Fin_SF',
                        'Gr_Liv_Area',
                        'Lot_Frontage',
                        'Second_Flr_SF',
                        'Garage_Area',
                        'Wood_Deck_SF',
                        'Open_Porch_SF',
                        'Enclosed_Porch',
                        'Three_season_porch',
                        'Screen_Porch',
                        'Misc_Val']
        # for col in VAR_OUTLIERS:
        #     cap_value = df[col].quantile(0.98)
        #     if cap_value > 0:
        #         df[col] = df[col].apply(lambda x: cap_value if x > cap_value else x)

    # ===== feature engineer =====
    df['old_house_ind'] = df['Year_Built'] < 1940

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

    # ===== remove some categorical data =====
    remove_var = ['Street', 'Utilities', 'Condition_2', 'Roof_Matl', 'Heating',
                  'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude', 'Latitude',
                  Y,
                  'PID',
                  'Garage_Yr_Blt'
                  ] + [
                     'Bldg_Type__0',
                     'Bldg_Type__4',
                     'BsmtFin_SF_1',
                     'BsmtFin_Type_1__4',
                     'BsmtFin_Type_2__4',
                     'Bsmt_Cond__3',
                     'Bsmt_Qual__3',
                     'Exterior_1st__0',
                     'Exterior_1st__11',
                     'Exterior_1st__4',
                     'Exterior_1st__5',
                     'Exterior_1st__6',
                     'Exterior_2nd__13',
                     'Exterior_2nd__2',
                     'Garage_Area',
                     'Garage_Cond__3',
                     'Garage_Qual__3',
                     'Garage_Type__6',
                     'House_Style__1',
                     'House_Style__4',
                     'House_Style__5',
                     'MS_SubClass__6',
                     'MS_Zoning__2',
                     'Sale_Condition__5',
                 ]

    return df[[i for i in df.columns if i not in remove_var]]


class LassoModel:
    DEFAULT_ALPHA = 8.439481965654006e-05
    Y_COL_NAME = Y
    MAX_ITER = 10000

    def __init__(self, df_train, y_series, tuning_parameters=None):
        self.df_train = df_train
        self.y_series = y_series

        if tuning_parameters is None:
            self.tuning_parameters = {}
        else:
            self.tuning_parameters = tuning_parameters

    def train(self, tune_ridge=False):

        self.lasso_model = linear_model.Lasso(
            alpha=self.tuning_parameters.get('alpha', self.DEFAULT_ALPHA),
            # fit_intercept=False,
            # normalize=False,
            max_iter=self.tuning_parameters.get('max_iter', self.MAX_ITER),
            fit_intercept=True,
            normalize=True
        )

        self.logged_y_training = log_y(self.y_series)
        feature_matrix = [i.tolist() for row_i, i in self.df_train.iterrows()]
        # model.fit(df_train_matrix, (self.logged_y_training - self.logged_y_training.mean()).tolist())
        self.lasso_model.fit(feature_matrix,
                             self.logged_y_training)

        logging.info(f"""Training data error:
        {error_evaluation(pd.Series(self.lasso_model.predict(feature_matrix)).apply(
            lambda logged_y: math.exp(logged_y) if logged_y > 0 else min(self.y_series)), self.y_series)}""")

        df_model_coef = pd.DataFrame(self.lasso_model.coef_, index=self.df_train.columns, columns=['coef']). \
            sort_values('coef', ascending=False)

        # ridge
        self.lasso_var = df_model_coef[abs(df_model_coef['coef']) > 0].index.tolist()

        if tune_ridge:
            ridge = linear_model.Ridge(fit_intercept=True,
                                       normalize=True,
                                       random_state=0,
                                       max_iter=10000)

            alphas = np.logspace(-5, 0.2, 20)

            tuned_parameters = [{'alpha': alphas}]
            n_folds = 5
            clf = GridSearchCV(ridge, tuned_parameters, cv=n_folds, refit=False, scoring='neg_mean_squared_error')

            clf.fit(self.df_train[self.lasso_var], y=self.logged_y_training)
            # clf.cv_results_['mean_test_score']
            best_alpha_ridge = clf.best_params_['alpha']
        else:
            best_alpha_ridge = BEST_ALPHA_RIDGE

        model = linear_model.Ridge(fit_intercept=True,
                                   normalize=True,
                                   alpha=best_alpha_ridge,
                                   max_iter=100000)
        model.fit(self.df_train[self.lasso_var], y=self.logged_y_training)
        self.model = model

        return model

    def predict(self, new_data, Y_COL_NAME=Y):
        predictions = self.model.predict([i.tolist() for row_i, i in new_data[self.lasso_var].iterrows()])
        predictions = pd.Series(predictions).apply(
            lambda logged_y: math.exp(logged_y) if logged_y > 0 else min(self.y_series))

        return predictions


class BoostingTreeMode(LassoModel):
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    MAX_DEPTH = 4

    # Step size shrinkage used in update to prevents overfitting. After each boosting step,
    # we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.
    ETA = 0.01

    NUM_ROUND = 1000

    def train(self):
        self.logged_y_training = log_y(self.y_series)
        num_round = self.tuning_parameters.get('NUM_ROUND', self.NUM_ROUND)

        xgb = XGBRegressor(objective='reg:squarederror',
                           learning_rate=0.03,
                           subsample=0.8,
                           colsample_bytree=0.8,
                           n_estimators=700,
                           max_depth=4,
                           gamma=0
                           )

        self.model = xgb.fit(self.df_train, y=self.logged_y_training)
        # predicted_y = self.predict(new_data=self.df_train)
        # logging.info(f'Training error: {error_evaluation(predicted_y, true_y=self.y_series)}')

    def print_tree(self, tree_id):
        df_tree = self.model.trees_to_dataframe()
        return df_tree[df_tree['Tree'] == tree_id]

    def predict(self, new_data, Y=Y):
        # ypred = self.model.predict(xgb_model.DMatrix(new_data))
        ypred = self.model.predict(new_data)
        ypred = (pd.Series(ypred)).apply(
            lambda logged_y: math.exp(logged_y) if logged_y > 0 else min(self.y_series))

        return ypred


def impute_missing_data(df):
    # deal with missing data
    for col_i in df:
        if df[col_i].isna().sum() > 0:
            df.loc[df[col_i].isna(), col_i] = df[col_i].mean()

    return df


if __name__ == '__main__':
    set_logging(10)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 40)

    parser = argparse.ArgumentParser(
        description="run_project_1.py")

    parser.add_argument('--folder',
                        help='folder for data',
                        default=None)
<<<<<<< Updated upstream
    
    # FOLDER = '/Users/yafa/Dropbox/Library/UIUC_PSL/UIUC_PSL/Project1/'
    # FOLDER = '/Users/fanyang/Dropbox/uiuc/cs598/UIUC_SPL/UIUC_PSL/Project1/'
    # parsed_args = parser.parse_args(['--folder', '/Users/yafa/Dropbox/Library/UIUC_PSL/UIUC_PSL/Project1/', '--stage', 'tuning'])
    
=======
    parser.add_argument('--stage',
                        help='Whether it is used for `tuning` or `submission`',
                        default='submission')

>>>>>>> Stashed changes
    parsed_args = parser.parse_args()

    FOLDER = parsed_args.folder
    if FOLDER is None:
        FOLDER = os.path.dirname(os.path.realpath(__file__))
<<<<<<< Updated upstream
    
=======
    # FOLDER = '/Users/fanyang/Dropbox/uiuc/cs598/UIUC_SPL/UIUC_PSL/Project1'
    TUNING_OR_SUBMISSION = parsed_args.stage

>>>>>>> Stashed changes
    print(parsed_args)

    ###########################################
    # Step 1: Preprocess training data
    #         and fit two models
<<<<<<< Updated upstream
    # '/Users/fanyang/Dropbox/uiuc/cs598/UIUC_SPL/Project1/train.csv'
    # '/Users/fanyang/Dropbox/uiuc/cs598/UIUC_SPL/UIUC_PSL/Project1/train.csv'
    
    df_train = pd.read_csv(os.path.join(FOLDER, "submit/train.csv"))
    df_train = df_train.reindex()
    y_series_train = df_train[Y]
    
    df_test = pd.read_csv(os.path.join(FOLDER, "submit/test.csv"))
=======


    df_train = pd.read_csv(os.path.join(FOLDER, "train.csv"))
    df_train = df_train.reindex()
    y_series_train = df_train[Y]

    df_train = clean_data(df_train, is_training_data=True)
    df_train_numeric, dict_one_hot_encoder = transform_category_vars(df=df_train, dict_one_hot_encoder=None)
    df_train_numeric = impute_missing_data(df_train_numeric)


    # Step 2: Preprocess test data
    df_test = pd.read_csv(os.path.join(FOLDER, "test.csv"))
>>>>>>> Stashed changes
    df_test = df_test.reindex()
    pid_series_test = df_test['PID']
    df_test = clean_data(df_test, is_training_data=False)
    df_test_numeric, dict_one_hot_encoder = transform_category_vars(df=df_test,
                                                                    dict_one_hot_encoder=dict_one_hot_encoder)
    df_test_numeric = impute_missing_data(df_test_numeric)
<<<<<<< Updated upstream
    
    # common_vars = set(df_test_numeric.columns).intersection(set(df_train_numeric.columns))
    #
    # df_train_numeric = df_train_numeric[common_vars]
    # df_test_numeric = df_test_numeric[common_vars]
    #
    
    # DO NOT LOAD test_y
    # https://piazza.com/class/kjvsp15j2g07ac?cid=297
    # df_test_y = pd.read_csv(os.path.join(FOLDER, "test_y.csv"))
    # df_test_y = df_test_y.reindex()
    
    for file_name, model_cls in [('mysubmission1.txt', LassoModel), ('mysubmission2.txt', BoostingTreeMode)]:
        # model_cls = LassoModel
        # model_cls = BoostingTreeMode
        model_obj = model_cls(df_train=df_train_numeric, y_series=y_series_train)
        model_obj.train()
        predicted_y = model_obj.predict(new_data=df_test_numeric)
        # self=model_obj
        # print(error_evaluation(predicted_y, true_y=df_test_y[Y]))
        
        pd.DataFrame({'PID': pid_series_test,
                      'Sale_Price': predicted_y}).to_csv(os.path.join(FOLDER, file_name), index=False)
=======
    # df_test_y = pd.read_csv(os.path.join(FOLDER, "test_y.csv"))
    # df_test_y = df_test_y.reindex()

    # Step3: model training and result
    if TUNING_OR_SUBMISSION == 'submission':

        for file_name, model_cls in [('mysubmission1.txt', LassoModel), ('mysubmission2.txt', BoostingTreeMode)]:
            model_obj = model_cls(df_train=df_train_numeric, y_series=y_series_train)
            model_obj.train()
            predicted_y = model_obj.predict(new_data=df_test_numeric)

            # print(error_evaluation(predicted_y, true_y=df_test_y[Y]))

            pd.DataFrame({'PID': pid_series_test,
                          'Sale_Price': predicted_y}).to_csv(os.path.join(FOLDER, file_name), index=False)
>>>>>>> Stashed changes
