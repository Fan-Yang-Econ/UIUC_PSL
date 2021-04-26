import os
import re
import argparse

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# FOLDER = '/Users/fanyang/Dropbox/uiuc/cs598/UIUC_SPL/UIUC_PSL/Project3/'; TEST_FILE_NAME = 'test_0.csv'; TRAIN_FILE_NAME = 'train_0.csv'
# FOLDER = '/Users/yafa/Dropbox/Library/UIUC_PSL/UIUC_PSL/Project3/'; TEST_FILE_NAME = 'test_0.csv'; TRAIN_FILE_NAME = 'train_0.csv'

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 40)

parser = argparse.ArgumentParser(
    description="run_project_3.py")

parser.add_argument('--folder',
                    help='folder for data',
                    default='')
parser.add_argument('--test_file_name',
                    default='test.csv')
parser.add_argument('--train_file_name',
                    default='train.csv')

parsed_args = parser.parse_args()

FOLDER = parsed_args.folder
TEST_FILE_NAME = parsed_args.test_file_name
TRAIN_FILE_NAME = parsed_args.train_file_name

# ###############################
# load the vocab file
##################################

with open(os.path.join(FOLDER, 'myvocab.txt')) as f:
    content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
final_vocab = [x.strip() for x in content]

# ###############################
# load the training data
##################################

df_train = pd.read_csv(os.path.join(FOLDER, TRAIN_FILE_NAME))


#########################################
# train model
#######################################

# prepare data
# convert review to words; clean data, remove non-letters, stop words
def review_to_words(raw_review):
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
    return (" ".join(meaningful_words))


num_reviews = df_train['review'].size
clean_train_reviews = []
for j in range(num_reviews):
    clean_train_reviews.append(review_to_words(df_train["review"][j]))
vectorizer = CountVectorizer(analyzer="word", vocabulary=final_vocab)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()
X = train_data_features
y = df_train['sentiment']

################################################################
# run ridge model
ridge = LogisticRegression(penalty='l2', solver='liblinear')
alphas = np.logspace(-2, 0, 10)
tuned_parameters = [{'C': alphas}]
n_folds = 5
clf = GridSearchCV(ridge, tuned_parameters, cv=n_folds, refit=False, scoring='roc_auc')
clf.fit(X, y)
best_alpha = clf.best_params_['C']

model = LogisticRegression(penalty='l2', solver='liblinear', C=best_alpha)
model.fit(X, y)

########################################
# load test data
#######################################

df_test = pd.read_csv(os.path.join(FOLDER, TEST_FILE_NAME))

#####################################
# Compute prediction
# Store your prediction for test data in a data frame
# "output": col 1 is test$id
#           col 2 is the predited probabilities
#####################################
num_reviews = len(df_test["review"])
clean_test_reviews = []
for k in range(num_reviews):
    clean_review = review_to_words(df_test["review"][k])
    clean_test_reviews.append(clean_review)
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# test auc
pred_test_y = model.predict_proba(test_data_features)[:, 1]
# df_y_test = pd.read_csv(os.path.join(FOLDER, 'test_y_0.csv'))
# print(metrics.roc_auc_score(df_y_test['sentiment'], pred_test_y))


df_test['prob'] = pred_test_y
df_test[['id', 'prob']].to_csv(os.path.join(FOLDER, 'mysubmission.txt'), columns=['id', 'prob'], index=None, sep='\t')
