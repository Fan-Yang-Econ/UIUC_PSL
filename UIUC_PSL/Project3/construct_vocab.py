import logging
import os

import spacy
import pandas as pd

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

DICT_VALID_WORDS = {}
for review in df['review'].tolist():
    for token in nlp(review):
        if token.ent_type_ == '' and not token.is_stop and token.is_alpha and token.pos_ != 'PUNCT' and len(token.text) > 1:
            DICT_VALID_WORDS[token.text] = {
                'lemma': token.lemma_,
                'token.pos_': token.pos_
            }

for i in list(DICT_VALID_WORDS.keys()):
    if DICT_VALID_WORDS[i]['token.pos_'] == 'PROPN':
        del DICT_VALID_WORDS[i]

DICT_LEMMA = {}
for word in list(DICT_VALID_WORDS.keys()):
    lemma = DICT_VALID_WORDS[word]['lemma']
    if lemma not in DICT_LEMMA:
        DICT_LEMMA[lemma] = {'text_set': set()}
    DICT_LEMMA[lemma]['text_set'].add(word)

len(DICT_LEMMA)


def _in_sentence(set_words, sentence):
    for word in set_words:
        if word in sentence:
            return True
    else:
        return False


for word in list(DICT_VALID_WORDS.keys()):
    logging.info(f'Evaluate correlation for word `{word}`')
    lemma = DICT_VALID_WORDS[word]['lemma']
    if 'correlation' in DICT_LEMMA[lemma]:
        continue
    series_has_lemma = df['review'].apply(lambda sentence: _in_sentence(set_words=DICT_LEMMA[lemma]['text_set'], sentence=sentence))
    if series_has_lemma.empty:
        raise ValueError(f'Cannot find word {word}')
    
    DICT_LEMMA[lemma]['correlation'] = {
        'sentiment-1 : word-1': (series_has_lemma & df['sentiment']).sum(),
        'sentiment-0 : word-0': (~series_has_lemma & ~df['sentiment']).sum(),
        'sentiment-1 : word-0': (~series_has_lemma & df['sentiment']).sum(),
    }


list_confusion_matrix = []
list_lemma = []
list_words = []
for lemma in DICT_LEMMA:
    list_lemma.append(lemma)
    list_confusion_matrix.append(DICT_LEMMA[lemma]['correlation'])
    list_words.append(DICT_LEMMA[lemma]['text_set'])
    
df_confusion_matrix = pd.DataFrame(list_confusion_matrix)
df_confusion_matrix['lemma'] = list_lemma
df_confusion_matrix['word_set'] = list_words

df_confusion_matrix['accuracy'] = (df_confusion_matrix['sentiment-1 : word-1'] + df_confusion_matrix['sentiment-0 : word-0']).apply(lambda x: max(x, N-x))


df_confusion_matrix = df_confusion_matrix.sort_values('accuracy', ascending=False)
df_confusion_matrix = df_confusion_matrix[0:5000]

df_confusion_matrix['word_set'] = df_confusion_matrix['word_set'].apply(lambda word_set: set([i.lower() for i in word_set]))
df_confusion_matrix['word_set'].apply(lambda x: len(x) ).sum()

df_confusion_matrix.to_csv(os.path.join(FOLDER, 'df_confusion_matrix.csv'), index=False)

