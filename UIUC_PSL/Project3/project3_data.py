import pandas as pd

# split data to 5 set of train/test
data = pd.read_csv("/Users/fanyang/Dropbox/uiuc/cs598/UIUC_SPL/UIUC_PSL/Project3/alldata.csv")
testIDs = pd.read_csv("/Users/fanyang/Dropbox/uiuc/cs598/UIUC_SPL/UIUC_PSL/Project3/splits_S21.csv")
for i in range(5):
    # i = 0
    id1 = testIDs.iloc[:,i].tolist()
    df_train  = data[data['id'].isin(id1)][['id', 'sentiment', 'review']]
    df_test = data[~data['id'].isin(id1)][['id', 'review']]
    df_test_y = data[~data['id'].isin(id1)][['id','sentiment', 'score']]
    df_train.to_csv('train_' + str(i) + '.csv' , index=False)
    df_test.to_csv('test_' + str(i) + '.csv' , index=False)
    df_test_y.to_csv('test_y_' + str(i) + '.csv' , index=False)