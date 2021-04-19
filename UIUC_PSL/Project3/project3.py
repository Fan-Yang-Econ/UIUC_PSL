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



    # use first data to get number of words
i = 1
data = 'train_' + str(i) + '.csv'
df_train = pd.read_csv(os.path.join(FOLDER, data))
num_reviews = df_train['review'].size

clean_train_reviews = []
# Loop over each review; create an index i that goes from 0 to the length of the movie review list
for j in range( num_reviews ):
    # Call our function for each one, and add the result to the list of clean reviews
    clean_train_reviews.append( review_to_words( df_train["review"][j] ))

# Initialize the "CountVectorizer" object
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None  \
                             # max_features = 3000
)

train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

vocab = vectorizer.get_feature_names()
dist = np.sum(train_data_features, axis=0)
features = pd.DataFrame(zip(vocab, dist), columns=['features', 'count']).sort_values(by=['count'])


# run lasso model to select words
X = train_data_features
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
model = LogisticRegression(penalty='l1', solver='liblinear', C = best_alpha)
model.fit(X , y)
df_model_coef = pd.DataFrame(model.coef_.reshape(-1,), columns=['coef']).sort_values('coef', ascending=False)
lasso_var = df_model_coef[abs(df_model_coef['coef'])>0].index.tolist()
myvocab = features.loc[lasso_var,:]['features'].tolist()


#use selected word to train models
list_result = []
for i in range(5):
    data = 'train_' + str(i) + '.csv'
    df_train = pd.read_csv(os.path.join(FOLDER, data))
    num_reviews = df_train['review'].size

    clean_train_reviews = []
# Loop over each review; create an index i that goes from 0 to the length of the movie review list
    for j in range( num_reviews ):
        # Call our function for each one, and add the result to the list of clean reviews
        clean_train_reviews.append( review_to_words( df_train["review"][j] ))


    vectorizer = CountVectorizer(analyzer = "word", vocabulary = myvocab)

    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()

    # run ridge model to select words
    X = train_data_features
    y = df_train['sentiment']

    ridge = LogisticRegression(penalty='l2', solver='liblinear')
    alphas = np.logspace(-2, 0, 10)
    tuned_parameters = [{'C': alphas}]
    n_folds = 5
    clf = GridSearchCV(ridge, tuned_parameters, cv=n_folds, refit=False, scoring='roc_auc')
    clf.fit(X, y)
    # clf.cv_results_['mean_test_score']
    best_alpha = clf.best_params_['C']
    print('for data_'+str(i) + ': best parameter is ' + str(best_alpha))
    model = LogisticRegression(penalty='l2', solver='liblinear', C=best_alpha)
    model.fit(X, y)
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

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    pred_test_y = model.predict_proba(test_data_features)[:,1]
    test_auc = roc_auc_score(df_test_y['sentiment'],pred_test_y)
    print('for data_'+str(i) + ': test auc is ' + str(test_auc))

    result_dict = {
        'data': i,
        'best parameter': best_alpha,
        'training auc': train_auc,
        'test_auc': test_auc,
    }

    list_result.append(result_dict)
