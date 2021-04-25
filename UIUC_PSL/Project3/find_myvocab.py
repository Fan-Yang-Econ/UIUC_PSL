
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
FOLDER = '/Users/fanyang/Dropbox/uiuc/cs598/UIUC_SPL/UIUC_PSL/Project3/'

#read data
data = 'alldata' + '.csv'
df_train = pd.read_csv(os.path.join(FOLDER, data))



# convert review to words
# remove html
# remove non-letters
# convert to lower case
# remove stop word

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


num_reviews = df_train['review'].size
ini_clean_train_reviews = []
for j in range( num_reviews ):
    ini_clean_train_reviews.append(review_to_words( df_train["review"][j] ))



# convert word to vectorizer use ngram 1-4
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

lasso_t_inner = list(set(word_id) & set(lasso_var))
all_select_vocab = features.loc[lasso_t_inner,:]['features'].tolist()
with open(os.path.join(FOLDER,'myvocab.txt'), '+w') as f:
    f.write('\n'.join(all_select_vocab))

lasso_rank = df_tresult.loc[lasso_var].sort_values('abs_value', ascending=False)
lasso_top_1000 = lasso_rank.iloc[:1000].index.tolist()
