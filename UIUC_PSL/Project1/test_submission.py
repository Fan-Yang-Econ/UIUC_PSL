import os
from pprint import pprint
from copy import deepcopy
import timeit
import pandas as pd

from UIUC_PSL.Project1.mymain import transform_category_vars, Y, \
    LassoModel, BoostingTreeMode, error_evaluation, clean_data, set_logging, impute_missing_data

set_logging(20)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 40)

# FOLDER = '/Users/yafa/Dropbox/Library/UIUC_PSL/UIUC_PSL/Project1/'
FOLDER = '/Users/fanyang/Dropbox/uiuc/cs598/UIUC_SPL/UIUC_PSL/Project1/'
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
        df_train_full.to_csv(os.path.join(FOLDER, "submit/train.csv"), index=False)
        df_test_full[[i for i in df_test_full.columns if i != 'Sale_Price']].to_csv(os.path.join(FOLDER, "submit/test.csv"), index=False)
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
    
    df_train_numeric = impute_missing_data(df_train_numeric)
    df_test_numeric = impute_missing_data(df_test_numeric)

    # LassoModel, BoostingTreeMode
    for model_cls in [LassoModel, BoostingTreeMode]:
        # model_cls = BoostingTreeMode
        start = timeit.default_timer()
        model_obj = model_cls(df_train=df_train_numeric,
                              y_series=df_train[Y])
        model_obj.train()
        stop = timeit.default_timer()
        total_time = stop - start
        print('Time: ', stop - start)
        # self=model_obj
        result_dict = {
            'test_id': TEST_ID,
            'testing_error': error_evaluation(model_obj.predict(new_data=df_test_numeric), true_y=df_test[Y]),
            'train_error': error_evaluation(model_obj.predict(new_data=df_train_numeric), true_y=df_train[Y]),
            'model_name': model_cls.__name__,
            'run_time' : total_time
        }
        
        pprint(result_dict)
        list_result.append(result_dict)
    
  
pprint(list_result)
# pd.DataFrame.from_dict(list_result)

# pprint(list_result)
# [{'model_name': 'LassoModel',
#   'test_id': 0,
#   'testing_error': 0.13106525309837272,
#   'train_error': 0.11817997192245801},
#  {'model_name': 'BoostingTreeMode',
#   'test_id': 0,
#   'testing_error': 0.13028082359066268,
#   'train_error': 0.07797490341214985},
#  {'model_name': 'LassoModel',
#   'test_id': 1,
#   'testing_error': 0.13082922733014238,
#   'train_error': 0.1139399469977012},
#  {'model_name': 'BoostingTreeMode',
#   'test_id': 1,
#   'testing_error': 0.1172836825411173,
#   'train_error': 0.08071898085566728},
#  {'model_name': 'LassoModel',
#   'test_id': 2,
#   'testing_error': 0.13989368149677284,
#   'train_error': 0.10927589729794303},
#  {'model_name': 'BoostingTreeMode',
#   'test_id': 2,
#   'testing_error': 0.12661889986310007,
#   'train_error': 0.07612962060248453},
#  {'model_name': 'LassoModel',
#   'test_id': 3,
#   'testing_error': 0.14555826296254745,
#   'train_error': 0.1096872597017607},
#  {'model_name': 'BoostingTreeMode',
#   'test_id': 3,
#   'testing_error': 0.1302344165633408,
#   'train_error': 0.07795253051368149},
#  {'model_name': 'LassoModel',
#   'test_id': 4,
#   'testing_error': 0.1438737487218149,
#   'train_error': 0.11268733482491186},
#  {'model_name': 'BoostingTreeMode',
#   'test_id': 4,
#   'testing_error': 0.1369888821690201,
#   'train_error': 0.07762578477128719},
#  {'model_name': 'LassoModel',
#   'test_id': 5,
#   'testing_error': 0.1311258372929508,
#   'train_error': 0.11818719127739506},
#  {'model_name': 'BoostingTreeMode',
#   'test_id': 5,
#   'testing_error': 0.1309290491076485,
#   'train_error': 0.07814801229732302},
#  {'model_name': 'LassoModel',
#   'test_id': 6,
#   'testing_error': 0.13081220184362388,
#   'train_error': 0.11392394554786284},
#  {'model_name': 'BoostingTreeMode',
#   'test_id': 6,
#   'testing_error': 0.11787923043551188,
#   'train_error': 0.08093200532246132},
#  {'model_name': 'LassoModel',
#   'test_id': 7,
#   'testing_error': 0.1398348253197081,
#   'train_error': 0.10927131146501394},
#  {'model_name': 'BoostingTreeMode',
#   'test_id': 7,
#   'testing_error': 0.1259887676187715,
#   'train_error': 0.07614085071463444},
#  {'model_name': 'LassoModel',
#   'test_id': 8,
#   'testing_error': 0.14527590295356613,
#   'train_error': 0.10969190526217241},
#  {'model_name': 'BoostingTreeMode',
#   'test_id': 8,
#   'testing_error': 0.13008339599855728,
#   'train_error': 0.0780994877407595},
#  {'model_name': 'LassoModel',
#   'test_id': 9,
#   'testing_error': 0.14408135916427195,
#   'train_error': 0.112659425116243},
#  {'model_name': 'BoostingTreeMode',
#   'test_id': 9,
#   'testing_error': 0.13672582459749558,
#   'train_error': 0.07769310329083137}]
