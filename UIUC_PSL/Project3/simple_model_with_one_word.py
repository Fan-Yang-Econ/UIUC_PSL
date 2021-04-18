import logging
import os
import ast

import spacy
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

FOLDER = '/Users/yafa/Dropbox/Library/UIUC_PSL/UIUC_PSL/Project3/'

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 40)


def set_logging(level=10,
                path=None):
    format = '%(levelname)s-%(name)s-%(funcName)s:\n %(message)s'
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    if path:
        logging.basicConfig(level=level, format=format, filename=path)
    else:
        logging.basicConfig(level=level, format=format)


set_logging(20)

df = pd.read_csv(os.path.join(FOLDER, 'alldata.tsv'), sep='\t')

df_confusion_matrix = pd.read_csv(os.path.join(FOLDER, 'df_confusion_matrix.csv'))
df_confusion_matrix = df_confusion_matrix[0:2000]

N = len(df)

nlp = spacy.load("en_core_web_sm")


# add review ids to each lemma
DICT_LEMMA = {}
for index_i, row_i in df_confusion_matrix.iterrows():
    _df = pd.DataFrame(list(ast.literal_eval(row_i['word_set'])))
    _df.columns = ['word']
    _df['lemma'] = row_i['lemma']
    DICT_LEMMA[row_i['lemma']] = {
        'reviews': set()
    }

list_reviews_with_lemma = {}
for count_i, row_i in df.iterrows():
    logging.info(f'process review {count_i}')
    
    set_lemma = set()
    for token in nlp(row_i['review']):
        if token.lemma_ in DICT_LEMMA:
            DICT_LEMMA[token.lemma_]['reviews'].add(row_i['id'])
df_lemma = pd.DataFrame(DICT_LEMMA)

# one-hot encoding
for count_i, lemma in enumerate(DICT_LEMMA):
    logging.info(f'Processing lemma count {count_i}')
    df['lemma-' + lemma] = df['id'].apply(lambda x: 1 if x in DICT_LEMMA[lemma]['reviews'] else 0)

# Run the Logistic model with Lasso
lasso_model = LogisticRegression(penalty='l1', solver='liblinear')
x_variables = [i for i in df.columns if 'lemma-' in i]
lasso_model.fit(X=df[x_variables], y=df['sentiment'])
prediction = lasso_model.predict_proba(X=df[x_variables])

# AUC
metrics.roc_auc_score(df['sentiment'], prediction[:, 1])
