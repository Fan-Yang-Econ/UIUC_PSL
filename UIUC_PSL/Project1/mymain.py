# Step 0: Load necessary libraries
###########################################
# Step 0: Load necessary libraries
#
import math
import argparse
import os
from copy import deepcopy
import logging

import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt

Y = 'Sale_Price'

def set_logging(level=10,
                path=None):
    format = '%(levelname)s-%(name)s-%(funcName)s:\n %(message)s'
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    if path:
        logging.basicConfig(level=level, format=format, filename=path)
    else:
        logging.basicConfig(level=level, format=format)


def _get_one_hot_encoder_df(_col_name, df_train):
    """
    hot code for one single category vars
    
    :param _col_name:
    :param df_train:
    :return:
    """
    enc = OneHotEncoder(handle_unknown='error')
    _cate_matrix = [i[[_col_name]] for row_i, i in df_train.iterrows()]
    enc.fit(_cate_matrix)
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
        try:
            df[x].astype(int)
            LIST_NUMERIC_C.append(x)
        except ValueError:
            LIST_CATEGORY_X.append(x)
            # len(LIST_CATEGORY_X)
            # len(LIST_NUMERIC_C)
            
    dict_one_hot_encoder_df = {}
    for _col_name in LIST_CATEGORY_X:
        dict_one_hot_encoder_df[_col_name] = _get_one_hot_encoder_df(_col_name, df)
    
    df_numeric = deepcopy(df[LIST_X])
    for _col_name in LIST_CATEGORY_X:
        del df_numeric[_col_name]
        for new_col in dict_one_hot_encoder_df[_col_name]:
            df_numeric[new_col] = dict_one_hot_encoder_df[_col_name][new_col]
    
    return df_numeric


def standardize_df(df_train_numeric):
    scaler = StandardScaler()
    scaler.fit(df_train_numeric)
    df_train_numeric_stand = pd.DataFrame(scaler.transform(df_train_numeric))
    df_train_numeric_stand.columns = df_train_numeric.columns
    
    return df_train_numeric_stand


def log_y(y_series):
    """
    Log the housing price
    :param y_series:
    :return:
    """
    return y_series.apply(lambda y: math.log(y))


def error_evaluation(predicted_y, true_y):
    return ((
                    pd.Series(predicted_y).apply(lambda y: math.log(y)) -
                    pd.Series(true_y).apply(lambda y: math.log(y))
            ) ** 2).mean()


class LassoModel:
    DEFAULT_ALPHA = 0.1
    Y_COL_NAME = Y
    
    def __init__(self, df_train, y_series, tuning_parameters=None):
        self.df_train = df_train
        self.y_series = y_series
        
        if tuning_parameters is None:
            self.tuning_parameters = {}
        else:
            self.tuning_parameters = tuning_parameters
    
    def train(self):
        df_train_numeric_stand = standardize_df(self.df_train)
        
        model = linear_model.Lasso(
            alpha=self.tuning_parameters.get('alpha', self.DEFAULT_ALPHA),
            fit_intercept=False,
            normalize=False)
        
        df_train_matrix = [i.tolist() for row_i, i in df_train_numeric_stand.iterrows()]
        
        self.logged_y_training = log_y(self.y_series)
        model.fit(df_train_matrix, (self.logged_y_training - self.logged_y_training.mean()).tolist())
        self.model = model
        
        return model
    
    def predict(self, new_data, Y_COL_NAME=Y):
        df_new_data_numeric = standardize_df(new_data)
        predictions = self.model.predict(df_new_data_numeric)
        predictions = pd.Series(predictions + self.logged_y_training.mean()).apply(
            lambda logged_y: math.exp(logged_y) if logged_y > 0 else min(df_train[self.Y_COL_NAME]))
        
        return predictions


class BoostingTreeMode(LassoModel):
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    MAX_DEPTH = 3
    
    # Step size shrinkage used in update to prevents overfitting. After each boosting step,
    # we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.
    ETA = 0.3
    
    NUM_ROUND = 3
    
    def train(self):
        self.logged_y_training = log_y(self.y_series)
        
        dtrain = xgb.DMatrix(self.df_train, label=self.logged_y_training - self.logged_y_training.mean())
        param = {'max_depth': self.tuning_parameters.get('max_depth', self.MAX_DEPTH),
                 'eta': self.tuning_parameters.get('eta', self.ETA),
                 'objective': 'reg:squarederror'}
        num_round = self.tuning_parameters.get('num_round', 5)
        
        self.model = xgb.train(param, dtrain, num_round)
    
    def predict(self, new_data):
        ypred = self.model.predict(xgb.DMatrix(new_data))
        ypred = (pd.Series(ypred) + self.logged_y_training.mean()).apply(
            lambda logged_y: math.exp(logged_y) if logged_y > 0 else min(df_train[self.Y_COL_NAME]))
        
        return ypred


if __name__ == '__main__':
    set_logging(10)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 40)
    
    parser = argparse.ArgumentParser(
        description="run_project_1.py")
    
    parser.add_argument('--folder',
                        help='folder for data',
                        default='.')
    parser.add_argument('--stage',
                        help='Whether it is used for `tuning` or `submission`',
                        default='submission')
    
    # FOLDER = '/Users/yafa/Dropbox/Library/UIUC_PSL/UIUC_PSL/Project1/'
    # parsed_args = parser.parse_args(['--folder', '/Users/yafa/Dropbox/Library/UIUC_PSL/UIUC_PSL/Project1/', '--stage', 'tuning'])
    
    parsed_args = parser.parse_args()
    
    FOLDER = parsed_args.folder
    if FOLDER == '.':
        FOLDER = os.path.dirname(os.path.realpath(__file__))
    
    TUNING_OR_SUBMISSION = parsed_args.stage
    
    print(parsed_args)
    ###########################################
    # Step 1: Preprocess training data
    #         and fit two models
    
    df_train = pd.read_csv(os.path.join(FOLDER, "train.csv"))
    df_train = df_train.reindex()
    
    df_test = pd.read_csv(os.path.join(FOLDER, "test.csv"))
    df_test = df_test.reindex()
    
    df_train_numeric = transform_category_vars(df=df_train)
    df_test_numeric = transform_category_vars(df=df_test)

    common_vars = set(df_test_numeric.columns).intersection(set(df_train_numeric.columns))
    
    df_train_numeric = df_train_numeric[common_vars]
    df_test_numeric = df_test_numeric[common_vars]

    df_test_y = pd.read_csv(os.path.join(FOLDER, "test_y.csv"))
    df_test_y = df_test_y.reindex()

    if TUNING_OR_SUBMISSION == 'submission':
    
        for file_name, model_cls in [('mysubmission1.txt', LassoModel), ('mysubmission2.txt', BoostingTreeMode)]:
            model_obj = model_cls(df_train=df_train_numeric, y_series=df_train[Y])
            model_obj.train()
            predicted_y = model_obj.predict(new_data=df_test_numeric)
            print(error_evaluation(predicted_y, true_y=df_test_y[Y]))
            
            pd.DataFrame({'PID': df_test['PID'], 'Sale_Price': predicted_y}).to_csv(os.path.join(FOLDER, file_name), index=False)
    
    elif TUNING_OR_SUBMISSION == 'tuning':
    
        def cv(list_tuning_parameters, model_class, df_train, y_series_train, df_test, y_series_test):
            dict_result = {
                'list_models': [],
                'best_model': None
            }
        
            for tuning_parameters in list_tuning_parameters:
                model_obj = model_class(df_train=df_train, y_series=y_series_train, tuning_parameters=tuning_parameters)
                model_obj.train()
                predicted_y = model_obj.predict(df_test)
            
                test_error = error_evaluation(predicted_y, true_y=y_series_test)
            
                _result_dict = {
                    'model': model_obj,
                    'tuning_parameter': tuning_parameters,
                    'test_error': test_error
                }
            
                if dict_result['best_model'] is None:
                    dict_result['best_model'] = _result_dict
            
                dict_result['list_models'].append(_result_dict)
        
            return dict_result
    
        def plot_bst_tree(bst):
            fig = matplotlib.pyplot.gcf()
            xgb.plot_importance(bst)
            xgb.plot_tree(bst, num_trees=4)
            fig.set_size_inches(100, 50)
            plt.figure(figsize=[100., 50.]).show()
            plt.savefig('/tmp/tree2.png')
    
    
        dict_lasso_result = cv(
            list_tuning_parameters=[{'alpha': 0.1}, {'alpha': 0.5}, {'alpha': 0.01}], model_class=LassoModel,
            df_train=df_train_numeric,
            df_test=df_test_numeric,
            y_series_train=df_train[Y],
            y_series_test=df_test_y[Y]
        )
        
        dict_boosting_tree_result = cv(
            list_tuning_parameters=[{'max_depth': 3, 'num_round': 3},
                                    {'max_depth': 4, 'num_round': 4},
                                    {'max_depth': 5, 'num_round': 4},
                                    {'max_depth': 4, 'num_round': 5},
                                    {'max_depth': 5, 'num_round': 5}], model_class=LassoModel,
            df_train=df_train_numeric,
            df_test=df_test_numeric,
            y_series_train=df_train[Y],
            y_series_test=df_test_y[Y]
        )
        
        print(dict_lasso_result['best_model'])
        print(dict_boosting_tree_result['best_model'])

