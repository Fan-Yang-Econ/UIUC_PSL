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

d_one_fold_train = pd.read_csv(os.path.join(FOLDER, 'train_0.csv'))

df_confusion_matrix = pd.read_csv(os.path.join(FOLDER, 'df_confusion_matrix.csv'))
# df_confusion_matrix = df_confusion_matrix[0:2000]

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

##====

N = 25000
model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
model.fit(X=df_feature.loc[0:N, selected_list_var], y=df.loc[0:N, 'sentiment'])
prediction = model.predict_proba(X=df_feature.loc[0:N, selected_list_var])
metrics.roc_auc_score(df.loc[0:N, 'sentiment'], prediction[:, 1])

df_model_coef = pd.DataFrame(model.coef_.reshape(-1, ), columns=['coef'])
lasso_var = df_model_coef[abs(df_model_coef['coef']) > 0].index.tolist()
len(lasso_var)

selected_list_var = pd.Series(df_feature.columns)[lasso_var].tolist()
len(selected_list_var)

len(df_feature.columns)

# select best features
myvocab = features.loc[lasso_var, :]['features'].tolist()

len(myvocab)

# AUC
metrics.roc_auc_score(df['sentiment'], prediction[:, 1])

df_available = df_feature.loc[df_missing.index, selected_list_var].sum(axis=1)

df_missing = df[df['sentiment'] != pd.Series(prediction[:, 1] > 0.5).apply(lambda x: 1 if x else 0)]

SELECT_FEATURE_IND = False


def two_step_predict(df_x_train, training_y, df_test_x, y_test):
    """

    :param df_x_train:
        df_x_train = df_feature

        df_x_train = df_feature[list_features]
        df_x_train = df_feature[df_features_500['col_name']]

        df_x_train = df_feature_small[df_feature_small.index.isin(review_id_with_wrong_prediciton)]

        len(df_feature.column)

    :param training_y:
        training_y = df['sentiment']

        training_y = df[df['id'].isin(review_id_with_wrong_prediciton)]['sentiment']
    :param df_test_x:
    :param y_test:
    :return:
    """

    from sklearn.ensemble import GradientBoostingClassifier
    SELECT_FEATURE_IND = 0

    for iteration in [180, 250, 300, 400, 500, 800]:
        for max_depth in [10, 20]:

            learning_rate = 0.1
            iteration = 1000
            max_depth = 20
            SELECT_FEATURE_IND = 1

            clf = GradientBoostingClassifier(
                n_estimators=iteration,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=0.5 if SELECT_FEATURE_IND else 1,
                verbose=True,
                random_state=0).fit(
                df_x_train,
                training_y)

            prediction_test = clf.predict_proba(X=df_test_x)
            print(f'test error for iteration {iteration} and max_depth {max_depth}')
            print(metrics.roc_auc_score(y_test, prediction_test[:, 1]))

            prediction_training = clf.predict_proba(X=df_x_train)
            print('training_error')
            print(metrics.roc_auc_score(training_y, prediction_training[:, 1]))


    clf = LogisticRegression(fit_intercept=False, penalty='l1', solver='liblinear')
    clf.fit(X=df_x_train, y=training_y)

    print(f'test error for iteration in logistic')
    prediction_test = clf.predict_proba(X=df_test_x)
    print(metrics.roc_auc_score(y_test, prediction_test[:, 1]))

    df_model_coef = pd.DataFrame(clf.coef_.reshape(-1, ), columns=['coef'])
    lasso_var = df_model_coef[abs(pd.DataFrame(clf.coef_.reshape(-1, ), columns=['coef'])['coef']) > 0].index.tolist()
    selected_list_var = pd.Series(df_x_train.columns)[lasso_var].tolist()

    print(len(selected_list_var))



    if SELECT_FEATURE_IND:
        df_features = pd.DataFrame({'col_name': df_x_train.columns, 'importance': clf.feature_importances_})
        df_features = df_features.sort_values(['importance'], ascending=False)
        df_features.to_csv('/Users/yafa/Dropbox/Library/UIUC_PSL/UIUC_PSL/Project3/df_features.csv')

        df_features = df_features[df_features['importance'] > 0]
        df_features.iloc[10]


        from pprint import pprint
        pprint(df_features[400:500]['col_name'].tolist())

        df_features = pd.read_csv('/tmp/df_features.csv')
        df_features['col_name'] = df_features['col_name'].apply(lambda x: ast.literal_eval(x) if '(' in x  else x)

        pprint(df_features[100:1000][df_features[100:1000]['col_name'].apply(lambda x: type(x) == tuple and x[1] in ['ADJ', 'ADV'])]['col_name'].tolist())

        list_to_not_include = df_features[100:1000][df_features[100:1000]['col_name'].apply(lambda x: type(x) == tuple and x[1] not in ['ADJ', 'ADV'])]['col_name'].tolist()
        len(list_to_not_include)

        df_features_500 = df_features[0:500][~df_features[0:500]['col_name'].isin(list_to_not_include)]

        pprint(df_features[100:1000][df_features[100:1000]['col_name'].apply(lambda x: type(x) != tuple )]['col_name'].tolist())



        (df_diff[df_diff['word'].isin(feasures_1000)]['count'] <= 50).sum()


        feasures_1000 = df_features[0:2000]['col_name'].tolist()
        len(feasures_1000)



        prediction_training

        boolean_wrong = (training_y != pd.Series(prediction_training[:, 1] ).apply(lambda x: 1 if x else 0)).tolist()
        df_missing = training_y[boolean_wrong]

        review_id_with_wrong_prediciton = set(df_missing.index.tolist())

        review_id_with_wrong_prediciton

        list_word_in_wrong_reviews = []
        for i in df_diff_small['word'].tolist():
            if set(DICT_VALID_WORDS[i]).intersection(review_id_with_wrong_prediciton):
                list_word_in_wrong_reviews.append(i)

        list_word_in_wrong_reviews[1]

        len(list_word_in_wrong_reviews)
        len(df_diff_small)
        len(review_id_with_wrong_prediciton)

        df_diff_small_wrong = df_diff_small[df_diff_small['word'].isin(list_word_in_wrong_reviews)]
        df_diff_small_wrong = df_diff_small_wrong[df_diff_small_wrong['diff'] == 1]

        df_diff_small_wrong = df_diff_small_wrong.sort_values('count', ascending=False)

        final_features = df_features[0:(1000 - 133)]['col_name'].tolist() + df_diff_small_wrong[df_diff_small_wrong['count'] >= 8]['word'].tolist()
        len(final_features)

    # feature_importances_ndarray of shape (n_features,)
    # The impurity-based feature importances.
    #
    # oob_improvement_ndarray of shape (n_estimators,)

    # clf = LogisticRegression(solver='liblinear')
    # clf.fit(X=df_x_train, y=training_y)
    prediction_training = clf.predict_proba(X=df_x_train)
    print(metrics.roc_auc_score(training_y, prediction_training[:, 1]))

    # Run the Logistic model with Lasso


    prediction = clf.predict_proba(X=df_x_train)
    print(metrics.roc_auc_score(training_y, prediction[:, 1]))

    boolean_wrong = (training_y != pd.Series(prediction[:, 1] > 0.5).apply(lambda x: 1 if x else 0)).tolist()
    df_missing = training_y[boolean_wrong]
    df_feature2 = df_x_train.loc[boolean_wrong]

    model_2 = LogisticRegression(solver='liblinear')
    model_2.fit(X=df_feature2, y=df_missing)
    df_model_coef = pd.DataFrame(model_2.coef_.reshape(-1, ), columns=['coef'])
    lasso_var = df_model_coef[abs(pd.DataFrame(model_2.coef_.reshape(-1, ), columns=['coef'])['coef']) > 0].index.tolist()
    selected_list_var_2 = pd.Series(df_x_train.columns)[lasso_var].tolist()

    len(pd.Series(selected_list_var_2 + selected_list_var).unique().tolist())

    LIST_VARS = pd.Series(selected_list_var_2 + selected_list_var).unique().tolist()

    prediction_2 = model_2.predict_proba(X=df_feature2)

    prediction_series = deepcopy(prediction[:, 1])
    prediction_series[df_missing['sentiment'].index] = prediction_2[:, 1]

    metrics.roc_auc_score(y_test, prediction_series)


from copy import deepcopy

df_one_fold_train = pd.read_csv(os.path.join(FOLDER, 'train_1.csv'))
df_one_fold_test = pd.read_csv(os.path.join(FOLDER, 'test_1.csv'))
df_one_fold_test_y = pd.read_csv(os.path.join(FOLDER, 'test_y_1.csv'))

df_x_train = df_feature[df['id'].isin(df_one_fold_train['id'])][list_features]
training_y = df_one_fold_train['sentiment']
assert len(df_x_train) == len(training_y) == len(df_one_fold_train)

len(df_x_train.columns)

df_test_x = df_feature[df['id'].isin(df_one_fold_test_y['id'])][list_features]
assert len(df_test_x) == len(df_one_fold_test_y)
y_test = pd.Series(df_one_fold_test_y['sentiment'])

two_step_predict(df_x_train, training_y, df_test_x, y_test)

test = df_test_x.iloc[0:4]
test.loc[pd.Series([True, False, False, False])]
