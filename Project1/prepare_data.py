
import os
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 40)

FOLDER = '/Users/yafa/Dropbox/Library/UIUC_PSL/Project1/'
df_ames = pd.read_csv(os.path.join(FOLDER, 'Ames_data.csv'))


with open(os.path.join(FOLDER, 'project1_testIDs.dat')) as f:
    str_testID = f.readlines()

TEST_ID = 6
df_test_id = pd.DataFrame([i.strip().split() for i in str_testID])
df_test_id[TEST_ID] = df_test_id[TEST_ID].apply(lambda x: int(x))

df_test_full = df_ames[df_ames['PID'].index.isin(df_test_id[6])]
df_train_full = df_ames[~df_ames['PID'].index.isin(df_test_id[6])]

df_train_full.to_csv(os.path.join(FOLDER, "train.csv"), index=False)
df_test_full[[i for i in df_test_full.columns if i != 'Sale_Price']].to_csv(os.path.join(FOLDER, "test.csv"), index=False)
df_test_full[[i for i in df_test_full.columns if i == 'Sale_Price']].to_csv(os.path.join(FOLDER, "test_y.csv"), index=False)
