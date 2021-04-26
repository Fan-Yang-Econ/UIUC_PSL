
import pandas as pd
import os
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
import ssl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

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




# convert review to words; clean data, remove non-letters, stop words
def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text()
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()

    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))


FOLDER = '/Users/fanyang/Dropbox/uiuc/cs598/UIUC_SPL/UIUC_PSL/Project3/'


data = 'alldata' + '.csv'
df_train = pd.read_csv(os.path.join(FOLDER, data))
num_reviews = df_train['review'].size

ini_clean_train_reviews = []
# Loop over each review; create an index i that goes from 0 to the length of the movie review list
for j in range( num_reviews ):
    # Call our function for each one, and add the result to the list of clean reviews
    ini_clean_train_reviews.append(review_to_words( df_train["review"][j] ))

# Initialize the "CountVectorizer" object
train_vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,  \
                             ngram_range = (1,4),
                             max_features = 30000
)

ini_train_data_features = train_vectorizer.fit_transform(ini_clean_train_reviews)
ini_train_data_features = ini_train_data_features.toarray()

vocab = train_vectorizer.get_feature_names()
# dist = np.sum(ini_train_data_features, axis=0)
# features = pd.DataFrame(zip(vocab, dist), columns=['features', 'count']).sort_values(by=['count'])
features = pd.DataFrame(vocab, columns=['features'])


#############################################################################################
#method 1 run lasso model to select words
X = ini_train_data_features
y = df_train['sentiment']

# cross validation for alpha selection
lasso = LogisticRegression(penalty='l1', solver='liblinear')
alphas = np.logspace(-1, 0, 5)
tuned_parameters = [{'C': alphas}]
n_folds = 5
clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False, scoring='roc_auc')
clf.fit(X , y)
# clf.cv_results_['mean_test_score']
best_alpha = clf.best_params_['C']
# print(best_alpha)
lasso_model = LogisticRegression(penalty='l1', solver='liblinear', C = best_alpha)
lasso_model.fit(X , y)
df_model_coef = pd.DataFrame(lasso_model.coef_.reshape(-1,), columns=['coef']).sort_values('coef', ascending=False)
lasso_var = df_model_coef[abs(df_model_coef['coef'])>0].index.tolist()
lasso_vocab = features.loc[lasso_var,:]['features'].tolist()


##############################################################
# method 2
# sub_train_data = ini_train_data_features[]
mean1 = np.mean(ini_train_data_features[y==1, :],axis=0)
mean2 = np.mean(ini_train_data_features[y==0, :],axis=0)
n1 = y.sum()
n2 = len(y) - n1
var1 = np.var(ini_train_data_features[y==1, :],axis=0)
var2 = np.var(ini_train_data_features[y==0, :],axis=0)
t_num = mean1 -mean2
t_den = np.sqrt(var1/n1 + var2/n2)
t_result = t_num/t_den
abs_result = np.abs(t_result)

df_tresult = pd.DataFrame(zip(t_result,abs_result), columns=['t_test', 'abs_value']).sort_values('abs_value', ascending=False)
# sort_result = -np.sort(-np.abs(t_result) )[:1000]

word_id = df_tresult.iloc[:1000].index.tolist()
pos_id =df_tresult.loc[word_id,][df_tresult.loc[word_id,]['t_test']>0].index.tolist()
neg_id =df_tresult.loc[word_id,][df_tresult.loc[word_id,]['t_test']<0].index.tolist()

# features.loc[pos_id[:50]]
# features.loc[neg_id[:50]]
# result from t-test


# ###############################
# load the vocab file
with open(os.path.join(FOLDER, 'myvocab_lasso.txt')) as f:
        content = f.readlines()
lasso_vocab = [x.strip() for x in content]

with open(os.path.join(FOLDER, 'myvocab_t_test1000.txt')) as f:
    content = f.readlines()
t1000_vocab = [x.strip() for x in content]




# lasso_t_inner = list(set(word_id) & set(lasso_var))
# all_select_vocab = features.loc[lasso_t_inner,:]['features'].tolist()
# with open(os.path.join(FOLDER,'myvocab.txt'), '+w') as f:
#     f.write('\n'.join(all_select_vocab))
#
# lasso_rank = df_tresult.loc[lasso_var].sort_values('abs_value', ascending=False)
# lasso_top_1000_id = lasso_rank.iloc[:1000].index.tolist()
# lasso_top_1000 = features.loc[lasso_top_1000_id,:]['features'].tolist()
#
# final_id =lasso_t_inner
# pos_id =df_tresult.loc[final_id,][df_tresult.loc[final_id,]['t_test']>0].index.tolist()
# # features.loc[pos_id[:50]]
# neg_id =df_tresult.loc[final_id,][df_tresult.loc[final_id,]['t_test']<0].index.tolist()
# features.loc[neg_id[:50]]

final_vocab = t1000_vocab
final_result = []

for i in range(5):
    ######
    ##  read training data
    data = 'train_' + str(i) + '.csv'
    df_train = pd.read_csv(os.path.join(FOLDER, data))

    #######
    #train model
    num_reviews = df_train['review'].size
    clean_train_reviews = []
    for j in range( num_reviews ):
        clean_train_reviews.append( review_to_words(df_train["review"][j] ))
    vectorizer = CountVectorizer(analyzer = "word", vocabulary = final_vocab)
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()
    X = train_data_features
    y = df_train['sentiment']
    # run ridge model to select words
    ridge = LogisticRegression(penalty='l2', solver='liblinear')
    np.logspace(-2, 0, 10)

    # alphas = np.logspace(-1, 1, 5)
    # clf.cv_results_['mean_test_score']

    tuned_parameters = [{'C': alphas}]
    n_folds = 5
    clf = GridSearchCV(ridge, tuned_parameters, cv=n_folds, refit=False, scoring='roc_auc')
    clf.fit(X, y)
    best_alpha = clf.best_params_['C']
    print('for data_'+str(i) + ': best parameter is ' + str(best_alpha))

    model = LogisticRegression(penalty='l2', solver='liblinear', C=best_alpha)
    model.fit(X, y)
    # training auc
    pred_y = model.predict_proba(X)[:, 1]
    train_auc = roc_auc_score(y, pred_y)
    print('for data_'+str(i) + ': training auc is ' + str(train_auc))

    #get test data
    test = 'test_' + str(i) + '.csv'
    test_y = 'test_y_' + str(i) + '.csv'
    df_test = pd.read_csv(os.path.join(FOLDER, test))
    df_test_y = pd.read_csv(os.path.join(FOLDER, test_y))

    num_reviews = len(df_test["review"])
    clean_test_reviews = []

    for k in range(num_reviews):
        clean_review = review_to_words(df_test["review"][k] )
        clean_test_reviews.append( clean_review )
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    # test auc
    pred_test_y = model.predict_proba(test_data_features)[:,1]
    test_auc = roc_auc_score(df_test_y['sentiment'],pred_test_y)
    print('for data_'+str(i) + ': test auc is ' + str(test_auc))

    result_dict = {
        'data': i,
        'number of word': X.shape[1],
        'best parameter': best_alpha,
        'training auc': train_auc,
        'test_auc': test_auc,
    }

    final_result.append(result_dict)
    # final_result_lasso_ridge - use intersection of lasso and t-test, then run ridge
    # final_result_lasso - use top 1000 lasso variable, then run model
    # final_result_lasso_top_1000

pd.DataFrame.from_dict(final_result)

