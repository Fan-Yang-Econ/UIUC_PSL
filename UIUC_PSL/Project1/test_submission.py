import os
from pprint import pprint
from copy import deepcopy

import pandas as pd

from UIUC_PSL.Project1.mymain import transform_category_vars, Y, \
    LassoModel, BoostingTreeMode, error_evaluation, clean_data, set_logging

set_logging(20)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 40)

FOLDER = '/Users/yafa/Dropbox/Library/UIUC_PSL/UIUC_PSL/Project1/'
df_ames = pd.read_csv(os.path.join(FOLDER, 'Ames_data.csv'))
#
# df_ames_numeric, dict_onehot_encoder = \
#     transform_category_vars(df=clean_data(deepcopy(df_ames), is_training_data=True),
#                             dict_one_hot_encoder=None)
#
# df_ames_numeric['PID'] = df_ames['PID']
# df_ames_numeric[Y] = df_ames[Y]

with open(os.path.join(FOLDER, 'project1_testIDs.dat')) as f:
    str_testID = f.readlines()


def prepare_data(df_all, TEST_ID, FOLDER, write_to_csv=True, str_testID=str_testID):
    # prepare data
    
    df_test_id = pd.DataFrame([i.strip().split() for i in str_testID])
    df_test_id[TEST_ID] = df_test_id[TEST_ID].apply(lambda x: int(x))
    
    df_test_full = df_all[df_all['PID'].index.isin(df_test_id[TEST_ID])]
    df_train_full = df_all[~df_all['PID'].index.isin(df_test_id[TEST_ID])]
    
    df_train_full = df_train_full.reset_index()
    df_test_full = df_test_full.reset_index()
    
    if write_to_csv:
        df_train_full.to_csv(os.path.join(FOLDER, "train.csv"), index=False)
        df_test_full[[i for i in df_test_full.columns if i != 'Sale_Price']].to_csv(os.path.join(FOLDER, "test.csv"), index=False)
        df_test_full[[i for i in df_test_full.columns if i == 'Sale_Price']].to_csv(os.path.join(FOLDER, "test_y.csv"), index=False)
    
    return {'df_test': df_test_full, 'df_train': df_train_full}


list_result = []

for TEST_ID in range(0, 10):
    # TEST_ID = 6
    DICT_DATA = prepare_data(df_all=df_ames, TEST_ID=TEST_ID, FOLDER=FOLDER, write_to_csv=False)
    df_train = DICT_DATA['df_train']
    df_test = DICT_DATA['df_test']

    df_train_numeric, dict_onehot_encoder = \
        transform_category_vars(df=clean_data(deepcopy(df_train), is_training_data=False),
                                dict_one_hot_encoder=None)
    
    df_test_numeric, dict_onehot_encoder_ = \
        transform_category_vars(df=clean_data(deepcopy(df_test), is_training_data=False),
                                dict_one_hot_encoder=dict_onehot_encoder)

    common_vars = set(df_test_numeric.columns).intersection(set(df_train_numeric.columns))
    
    df_train_numeric = df_train_numeric[common_vars]
    df_test_numeric = df_test_numeric[common_vars]
    
    for model_cls in [LassoModel, BoostingTreeMode]:
        model_obj = model_cls(df_train=df_train_numeric, df_test=df_test_numeric, y_series=df_train[Y])
        model_obj.train()
        
        # self=model_obj
        result_dict = {
            'test_id': TEST_ID,
            'testing_error': error_evaluation(model_obj.predict(new_data=df_test_numeric), df_test[Y]),
            'train_error': error_evaluation(model_obj.predict(new_data=df_train_numeric), df_train[Y]),
            'model_name': model_cls.__name__
        }
        
        pprint(result_dict)
        list_result.append(result_dict)
        
        
    
    # pprint(list_result)
    # [{'model_name': 'LassoModel',
    #   'test_id': 6,
    #   'testing_error': 0.043669030888144056,
    #   'train_error': 0.0552906514912787},
    #  {'model_name': 'BoostingTreeMode',
    #   'test_id': 6,
    #   'testing_error': 0.035617502853840154,
    #   'train_error': 0.03624405277329622}]

pprint(list_result)
