import itertools
from pprint import pprint

from sklearn import linear_model
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv('~/Dropbox/Library/UIUC_PSL/quiz3/prostate.csv')
Y = 'lpsa'

list_x = 'lcavol + lweight + age + lbph + svi + lcp + gleason + pgg45'.split('+')
list_x = [i.strip() for i in list_x]

df_train = df[df['train']]
df_test = df[~df['train']]

dict_results = {}

for n_var in range(1, 9):
    for tuple_var in list(itertools.combinations(list_x, n_var)):
        formula_ = f"{Y} ~ {'+'.join(tuple_var)}"
        mod = smf.ols(formula=formula_, data=df_train)
        res = mod.fit()
        res.summary()
        dict_results[tuple_var] = {'aic': res.aic, 'bic': res.bic, 'res': res}

pprint(dict_results)

min_aic = min([dict_result['aic'] for dict_result in dict_results.values()])
model_aic = [model for model, dict_result in dict_results.items() if dict_result['aic'] == min_aic][0]

min_bic = min([dict_result['bic'] for dict_result in dict_results.values()])
model_bic = [model for model, dict_result in dict_results.items() if dict_result['bic'] == min_bic][0]


def get_mse(df_test, y_var, model_result):
    predictions = model_result.predict(df_test)
    rss = (df_test[y_var] - predictions).apply(lambda x: x ** 2).sum()
    return rss
    
get_mse(df_test, Y, model_result=dict_results[tuple(list_x)]['res'])
get_mse(df_test=df[~df['train']], y_var=Y, model_result=dict_results[model_bic]['res'])
get_mse(df_test=df[~df['train']], y_var=Y, model_result=dict_results[model_aic]['res'])


mod = smf.gls(formula='lpsa ~ lcavol + lweight + age + lbph + svi + lcp + gleason + pgg45', data=df[df['train']])

for l1_wt in [0.1, 0.5, 0.01]:
    clf = linear_model.Lasso(
        alpha=l1_wt,
        fit_intercept=True, normalize=True)
    
    clf.fit([i.tolist() for row_i, i in df_train[list_x].iterrows()], df_train[Y].tolist())
    predictions = clf.predict([i.tolist() for row_i, i in df_test[list_x].iterrows()])
    rss = (df_test[Y] - predictions).apply(lambda x: x ** 2).sum()
    print(rss)