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



# ###############################
# load the vocab file
with open(os.path.join(FOLDER, 'myvocab.txt')) as f:
        content = f.readlines()
final_vocab = [x.strip() for x in content]

# final_vocab = lasso_top_1000
# final_vocab = all_select_vocab

final_result_lasso_inner_v2 = []
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
        'best parameter': best_alpha,
        'training auc': train_auc,
        'test_auc': test_auc,
    }

    final_result_lasso_inner_v2.append(result_dict)
    # final_result_lasso_ridge - use intersection of lasso and t-test, then run ridge
    # final_result_lasso - use top 1000 lasso variable, then run model
    # final_result_lasso_top_1000

    # pd.DataFrame.from_dict(final_result_lasso_ridge)






