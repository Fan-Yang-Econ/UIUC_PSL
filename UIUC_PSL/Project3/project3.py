import pandas as pd
import os
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
import ssl
from sklearn.feature_extraction.text import CountVectorizer

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
#
# nltk.download()


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

# load first set
FOLDER = '/Users/fanyang/Dropbox/uiuc/cs598/UIUC_SPL/UIUC_PSL/Project3/'

i = 1
data = 'train_' + str(i) + '.csv'
df_train = pd.read_csv(os.path.join(FOLDER, data))
num_reviews = df_train['review'].size

# For example, Porter Stemming and Lemmatizing (both available in NLTK) would allow us to treat "messages", "message", and "messaging" as the same word, which could certainly be useful.
# myvocab = pd.read_csv(os.path.join(FOLDER, 'df_confusion_matrix.csv'))
def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))

clean_train_reviews = []
# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list
for i in range( num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_reviews.append( review_to_words( df_train["review"][i] ))
