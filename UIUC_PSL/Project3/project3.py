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
import timeit
from sklearn.ensemble import RandomForestClassifier
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
#
# nltk.download()


# load first set
FOLDER = '/Users/fanyang/Dropbox/uiuc/cs598/UIUC_SPL/UIUC_PSL/Project3/'
max_features = 2000;


# convert review to words; clean data, remove non-letters, stop words
def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)

    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text()
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z0-9]", " ", review_text)
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
data = 'train_0' + '.csv'
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
                             # max_features = 30000
                             min_df = 10
)

ini_train_data_features = train_vectorizer.fit_transform(ini_clean_train_reviews)
ini_train_data_features = ini_train_data_features.toarray()

vocab = train_vectorizer.get_feature_names()
# dist = np.sum(ini_train_data_features, axis=0)
# features = pd.DataFrame(zip(vocab, dist), columns=['features', 'count']).sort_values(by=['count'])
features = pd.DataFrame(vocab, columns=['features'])
X = ini_train_data_features
y = df_train['sentiment']

# cross validation for alpha selection
lasso = LogisticRegression(penalty='l1', solver='liblinear')
alphas = np.logspace(-2, 0, 10)
tuned_parameters = [{'C': alphas}]
n_folds = 3
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
with open(os.path.join(FOLDER,'myvocab_lasso_v2.txt'), '+w') as f:
    f.write('\n'.join(lasso_vocab))



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


df_tresult.to_csv(FOLDER + 't_test_result_v2.csv')
word_id = df_tresult.iloc[:2000].index.tolist()
# pos_id =df_tresult.loc[word_id,][df_tresult.loc[word_id,]['t_test']>0].index.tolist()
# neg_id =df_tresult.loc[word_id,][df_tresult.loc[word_id,]['t_test']<0].index.tolist()
word_id_1000 = df_tresult.iloc[:1000].index.tolist()
t_test_2000 = features.loc[word_id,:]['features'].tolist()
t_test_1000 = features.loc[word_id_1000,:]['features'].tolist()
with open(os.path.join(FOLDER,'myvocab_t_test2000_v2.txt'), '+w') as f:
    f.write('\n'.join(t_test_2000))
with open(os.path.join(FOLDER,'myvocab_t_test1000_v2.txt'), '+w') as f:
    f.write('\n'.join(t_test_1000))

pos_id =df_tresult.loc[word_id,][df_tresult.loc[word_id,]['t_test']>0].index.tolist()
neg_id =df_tresult.loc[word_id,][df_tresult.loc[word_id,]['t_test']<0].index.tolist()
features.loc[pos_id[:50]]
features.loc[neg_id][:50]
###################################################################



# ###############################
# load the vocab file
with open(os.path.join(FOLDER, 'myvocab_lasso.txt')) as f:
        content = f.readlines()
lasso_vocab = [x.strip() for x in content]

with open(os.path.join(FOLDER, 'myvocab_t_test1000.txt')) as f:
    content = f.readlines()
t1000_vocab = [x.strip() for x in content]


with open(os.path.join(FOLDER, 'myvocab_t_test2000.txt')) as f:
    content = f.readlines()
t2000_vocab = [x.strip() for x in content]
lasso_t2000_combine=list(set(t2000_vocab) & set(lasso_vocab))

# df_tresult = pd.read_csv(FOLDER + 't_test_result.csv')
with open(os.path.join(FOLDER, 'myvocab.txt')) as f:
    content = f.readlines()
final_vocab = [x.strip() for x in content]

# final_vocab = t1000_vocab
# final_vocab = t2000_vocab
final_vocab = lasso_t2000_combine


final_result = []
for i in range(5):
    ######
    start = timeit.default_timer()
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
    alphas = np.logspace(-2, 0, 10)
    # clf.cv_results_['mean_test_score']

    tuned_parameters = [{'C': alphas}]
    n_folds = 5
    clf = GridSearchCV(ridge, tuned_parameters, cv=n_folds, refit=False, scoring='roc_auc')
    clf.fit(X, y)
    best_alpha = clf.best_params_['C']
    print('for data_'+str(i) + ': best parameter is ' + str(best_alpha))

    model = LogisticRegression(penalty='l2', solver='liblinear', C=best_alpha)
    model.fit(X, y)
    stop = timeit.default_timer()
    # print('Time: ', stop - start)
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
        'best_parameter': best_alpha,
        'training_auc': train_auc,
        'test_auc': test_auc,
        'running_time': stop - start
    }

    final_result.append(result_dict)
    # final_result_lasso_ridge - use intersection of lasso and t-test, then run ridge
    # final_result_lasso - use top 1000 lasso variable, then run model
    # final_result_lasso_top_1000

# df_t1000 = pd.DataFrame.from_dict(final_result)
# df_t2000 = pd.DataFrame.from_dict(final_result)
# df_t2000_lasso_combo = pd.DataFrame.from_dict(final_result)
df_final = pd.DataFrame.from_dict(final_result)




