import logging
import os
import re
import ast

import spacy
import pandas as pd
import numpy as np

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

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv(os.path.join(FOLDER, 'alldata.tsv'), sep='\t')
N = len(df)

list_nlp = []
for count_i, review in enumerate(df['review'].tolist()):
    logging.info(f'process review {count_i}')
    list_nlp.append(nlp(review))

import pickle
with open(os.path.abspath(f'/Users/yafa/Downloads/list_nlp.spacy'), '+wb') as f:
    if len(list_nlp) == N:
        pickle.dump(list_nlp, f)

#
#
import pickle
list_nlp = []
with open(f'/Users/yafa/Downloads/list_nlp.spacy', '+rb') as f:
    list_nlp2 = pickle.load(f)

len(list_nlp2)

assert len(list_nlp) == N

MEANINGFUL_2_GRAM_set = {'ADJ', 'ADV', 'NOUN', 'PART'}
DICT_VALID_WORDS = {}

for count_review, review in enumerate(list_nlp):
    list_two_gram_i = []

    last_token = None
    for count_i, token in enumerate(review):

        if token.ent_type_ == '' and not token.is_stop and token.is_alpha and token.pos_ != 'PUNCT' and len(token.text) > 1:
            lemma_key = (token.lemma_, token.pos_)
            if lemma_key not in DICT_VALID_WORDS:
                DICT_VALID_WORDS[lemma_key] = []

            DICT_VALID_WORDS[lemma_key].append(count_review)

        if count_i > 0:

            if (last_token.is_alpha and token.is_alpha) and \
                    (not last_token.is_stop and not token.is_stop) and \
                    {last_token.pos_, token.pos_}.issubset(MEANINGFUL_2_GRAM_set):
                two_gram = f'{last_token.lemma_} {token.lemma_}'

                if last_token.pos_ == 'NOUN' and token.pos_ == 'ADV':
                    print(count_review, two_gram)
                else:
                    if two_gram not in DICT_VALID_WORDS:
                        DICT_VALID_WORDS[two_gram] = []
                    DICT_VALID_WORDS[two_gram].append(count_review)

        last_token = token

len(DICT_VALID_WORDS)

list_only_one_frq = []
DICT_VALID_WORD_WITH_SENTIMENT = {}

for i in list(DICT_VALID_WORDS):
    freq = len(DICT_VALID_WORDS[i])
    if freq == 1:
        list_only_one_frq.append(i)
        del DICT_VALID_WORDS[i]
        continue

    sentiment_series = df['sentiment'][DICT_VALID_WORDS[i]]
    DICT_VALID_WORD_WITH_SENTIMENT[i] = {
        1: (sentiment_series == 1).sum(),
        0: (sentiment_series == 0).sum()
    }

for i in list(DICT_VALID_WORDS):
    if DICT_VALID_WORD_WITH_SENTIMENT[i][1] == DICT_VALID_WORD_WITH_SENTIMENT[i][0]:
        del DICT_VALID_WORDS[i]

    count_ = (DICT_VALID_WORD_WITH_SENTIMENT[i][1] + DICT_VALID_WORD_WITH_SENTIMENT[i][0])
    DICT_VALID_WORD_WITH_SENTIMENT[i]['diff'] = max(DICT_VALID_WORD_WITH_SENTIMENT[i][1], DICT_VALID_WORD_WITH_SENTIMENT[i][0]) / count_
    DICT_VALID_WORD_WITH_SENTIMENT[i]['count'] = count_

list_diff = []
list_count_word = []
valid_words = list(DICT_VALID_WORDS)
for i in valid_words:
    list_diff.append(DICT_VALID_WORD_WITH_SENTIMENT[i]['diff'])
    list_count_word.append(DICT_VALID_WORD_WITH_SENTIMENT[i]['count'])

df_diff = pd.DataFrame({'diff': list_diff, 'word': valid_words, 'count': list_count_word})
df_diff = df_diff.sort_values('diff', ascending=False)

# not (
#             re.compile('ing$').findall(x[0]) + re.compile('ness$').findall(x[0]) + re.compile('y$').findall(x[0]) + re.compile('y$').findall(x[0])
#     )


words_with_same_reviews_id = []

for count_i, df_diff_i in df_diff.groupby('count'):
    list_sets = []
    for word in df_diff_i['word'].tolist():
        new_set = set(DICT_VALID_WORDS[word])
        if new_set in list_sets:
            words_with_same_reviews_id.append(word)
        else:
            list_sets.append(new_set)

df_diff = df_diff[~df_diff['word'].isin(words_with_same_reviews_id)]

print(len(df_diff))
print((df_diff['count'] < 10).sum())

df_diff_small = df_diff[df_diff['count'] < 10]
df_diff_big = df_diff[df_diff['count'] >= 8]

df_features = pd.read_csv('/Users/yafa/Dropbox/Library/UIUC_PSL/UIUC_PSL/Project3/df_features.csv')
df_features = df_features.sort_values('importance', ascending=False)
df_features_1000 = df_features[0:2000]
df_features_1000['col_name'] = df_features_1000['col_name'].apply(lambda x: ast.literal_eval(x) if '(' in x else x)

list_features = df_features_1000['col_name'].tolist()
len(list_features)

def create_feature_df(list_features):
    DICT_VALID_WORDS_AS_FEATURE = {}
    for i in list_features:
        zero_list = np.repeat(0, N)
        for review_id in DICT_VALID_WORDS[i]:
            zero_list[review_id] += 1
        DICT_VALID_WORDS_AS_FEATURE[i] = zero_list

    return pd.DataFrame(DICT_VALID_WORDS_AS_FEATURE)

df_feature = create_feature_df(list_features)

len(df_feature.columns)


len(df_diff_big)

DICT_REVIEW_ID_REPRESENTED = {}
for word in list_features:
    for review_id in DICT_VALID_WORDS[word]:
        DICT_REVIEW_ID_REPRESENTED[review_id] = DICT_REVIEW_ID_REPRESENTED.get(review_id, 0) + 1

len(DICT_REVIEW_ID_REPRESENTED)





del globals()['clf']



len(df_diff[df_diff['count'] >= 1000])
len(df_diff[df_diff['count'] >= 500])
len(df_diff[df_diff['count'] >= 100])
len(df_diff[df_diff['count'] >= 50])
len(df_diff[df_diff['count'] >= 10])
len(df_diff[df_diff['count'] < 10])

len(words_with_same_reviews_id)

single_nouns = df_diff[df_diff['word'].apply(
    lambda x:
    type(x) is tuple and x[1] == 'NOUN'
)]['word'].apply(lambda x: x[0]).tolist()

len(single_nouns)
two_grams = df_diff[df_diff['word'].apply(lambda x: type(x) is str)]['word'].tolist()
two_grams_list = set(' '.join(two_grams).split(' '))


df_feature.to_csv(os.path.join(FOLDER, 'feature.csv'))
df_corr = df_feature.corr(method='pearson')
