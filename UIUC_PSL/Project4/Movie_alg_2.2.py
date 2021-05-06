"""
Reference:

https://surprise.readthedocs.io/en/stable/knn_inspired.html?highlight=KNNWithMeans#surprise.prediction_algorithms.knns.KNNWithMeans

Notation:

https://surprise.readthedocs.io/en/stable/notation_standards.html

"""
import os

from surprise import Dataset
from surprise import Reader
import pandas as pd
from surprise import KNNWithMeans
from surprise.model_selection import KFold, GridSearchCV

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 40)

# load data
# FOLDER = '/Users/yafa/Dropbox/Library/UIUC_PSL/UIUC_PSL/Project4/'
FOLDER = '/Users/fanyang/Dropbox/uiuc/cs598/UIUC_SPL/UIUC_PSL/Project4/'
df_movie = pd.read_csv(os.path.join(FOLDER, 'MovieAppBackendServer/MovieAppBackendServer/data/movies.csv'))
len(df_movie)

df_rating = pd.read_csv(os.path.join(FOLDER, 'ratings.csv'))

df_rating_avg = df_rating.groupby(['MovieID'], as_index=False)['Rating'].agg(['sum', 'count'])
df_rating_avg['MovieID'] = df_rating_avg.index
df_rating_avg['avg_rating'] = df_rating_avg['sum'] / df_rating_avg['count']
df_rating_avg = df_rating_avg.reset_index(drop=True)
df_rating_avg = df_rating_avg.sort_values(['avg_rating', 'count'], ascending=False)

## 6040 users
df_users = pd.read_csv(os.path.join(FOLDER, 'users.csv'))

len(df_rating['MovieID'].unique())

# use Surprise
reader = Reader(rating_scale=(1, 5))
rating_data = Dataset.load_from_df(df_rating[['UserID', 'MovieID', 'Rating']], reader)
anti_set = rating_data.build_full_trainset().build_anti_testset()
full_train = rating_data.build_full_trainset()

# 10 iterations
fold = 10
kf = KFold(n_splits=fold)

i = 1
for trainset, testset in kf.split(rating_data):
    # use first fold to do parameter tuning
    if i <= 1:
        param_grid = {'k': [10, 50, 100],
                      'sim_options': {'name': ['cosine'],
                                      'user_based': [False]}
                      }
        gs = GridSearchCV(KNNWithMeans, param_grid, measures=['rmse'], cv=3)
        gs.fit(rating_data)
        i += 1

df_cv = pd.DataFrame.from_dict(gs.cv_results)

best_k = gs.best_params['rmse']['k']

sim_options = {'name': 'cosine',
               'user_based': False  # compute  similarities between items
               }

# best_k = 50
algo = KNNWithMeans(k=best_k, sim_options=sim_options)
algo.fit(full_train)
# Compute and print Root Mean Squared Error
movieID_list = df_rating['MovieID'].unique().tolist()
df_sim = pd.DataFrame(algo.sim)

# find movie (key) and its neighbours
dic_50NN = {}
for movie in movieID_list:
    movie_inner_iid = full_train.to_inner_iid(movie)
    k_nn = algo.get_neighbors(movie_inner_iid, k=best_k)
    dic_50NN[movie_inner_iid] = k_nn

# find movie and see if it is in others neighbours
dic_i_in_j_NN = {}
for inner_id_i in dic_50NN.keys():
    for inner_id_j, movie_list in dic_50NN.items():
        if inner_id_i in movie_list:  # if n
            dic_i_in_j_NN[inner_id_i] = dic_i_in_j_NN.get(inner_id_i, []) + [inner_id_j]

# transform from the inner iD to movie ID
record_list = []
for i in dic_i_in_j_NN.keys():
    i_raw_id = full_train.to_raw_iid(i)
    for j in dic_i_in_j_NN[i]:
        sim = algo.sim[i, j]
        j_raw_id = full_train.to_raw_iid(j)
        record = {'MovieID': i_raw_id,
                  'Whose_NN': j_raw_id,
                  'similarity': sim}
        record_list.append(record)

df_nn = pd.DataFrame(record_list)

# output data
df_nn.to_csv(os.path.join(FOLDER, 'MovieAppBackendServer/MovieAppBackendServer/data/df_nn.csv'), index=False)
df_rating_avg.to_csv(os.path.join(FOLDER, 'MovieAppBackendServer/MovieAppBackendServer/data/df_rating_avg.csv'), index=False)
