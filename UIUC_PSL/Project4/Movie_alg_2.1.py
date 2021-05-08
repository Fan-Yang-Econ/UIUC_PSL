import os

from surprise import SVD
from surprise import accuracy
from surprise.model_selection import KFold, RandomizedSearchCV
from surprise import Dataset
from surprise import Reader
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 40)

# load data
# FOLDER = '/Users/yafa/Dropbox/Library/UIUC_PSL/UIUC_PSL/Project4/'
FOLDER = '/Users/fanyang/Dropbox/uiuc/cs598/UIUC_SPL/UIUC_PSL/Project4/'
df_movie = pd.read_csv(os.path.join(FOLDER, 'MovieAppBackendServer/MovieAppBackendServer/data/movies.csv'))
df_rating = pd.read_csv(os.path.join(FOLDER, 'ratings.csv'))
## 6040 users
df_users = pd.read_csv(os.path.join(FOLDER, 'users.csv'))

# use Surprise
reader = Reader(rating_scale=(1, 5))
rating_data = Dataset.load_from_df(df_rating[['UserID', 'MovieID', 'Rating']], reader)
anti_set = rating_data.build_full_trainset().build_anti_testset()

# 10 iterations
fold = 10
kf = KFold(n_splits=fold)

i = 1
for trainset, testset in kf.split(rating_data):
    # use first fold to do parameter tuning
    if i <= 1:
        # baseline, no parameter tuning - rmse = 0.8681
        # algo = SVD()
        # algo.fit(trainset)
        # predictions = algo.test(testset)
        # accuracy.rmse(predictions, verbose=True)
        
        param_grid = {'n_factors': [50, 100],
                      'lr_all': [0.002, 0.005, 0.2],
                      'reg_all': [0.02, 0.2],
                      'n_epochs': [10, 20]
                      }
        gs = RandomizedSearchCV(SVD, param_grid, n_iter=10, measures=['rmse'], cv=3)
        gs.fit(rating_data)
        i += 1

df_cv_result = pd.DataFrame.from_dict(gs.cv_results)

best_factor = gs.best_params['rmse']['n_factors']
best_lr = gs.best_params['rmse']['lr_all']
best_reg = gs.best_params['rmse']['reg_all']
best_epoch = gs.best_params['rmse']['n_epochs']

algo = SVD(n_factors=best_factor, lr_all=best_lr, reg_all=best_reg, n_epochs=best_epoch)
SVD_accuracy_list = []
for trainset, testset in kf.split(rating_data):
    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)
    # Compute and print Root Mean Squared Error
    current_acc = accuracy.rmse(predictions)
    SVD_accuracy_list.append(current_acc)

# new user rating data
rating_movie = [2905, 2019, 858, 1198, 260]
rating_result = [1, 1, 1, 1, 1]
n = len(rating_result)
user_list = [0] * n

ratings_dict = {'UserID': user_list,
                'MovieID': rating_movie,
                'Rating': rating_result}
df_new = pd.DataFrame(ratings_dict)

# add new user data to existing data, for quick processing, only use 30% of the data
original_data = df_rating[['UserID', 'MovieID', 'Rating']].sample(frac=0.3)
df_new_rating = df_new.append(original_data[['UserID', 'MovieID', 'Rating']], ignore_index=True)
reader = Reader(rating_scale=(1, 5))
new_data = Dataset.load_from_df(df_new_rating[['UserID', 'MovieID', 'Rating']], reader)
new_train = new_data.build_full_trainset()
# run algo
algo = SVD(n_factors=best_factor, lr_all=best_lr, reg_all=best_reg, n_epochs=best_epoch)
algo.fit(new_train)
# prediction
movie_list = [movie for movie in original_data['MovieID'].unique().tolist() if movie not in rating_movie]
uid = 0
SVD_pred_list = []
for movie in movie_list:
    iid = movie
    pred = algo.predict(uid, iid)[3]
    SVD_pred_list.append(pred)

top_5_idx = np.argsort(SVD_pred_list)[-5:]
top_5_pred_value = [SVD_pred_list[i] for i in top_5_idx]
top5_movie_id = [movie_list[i] for i in top_5_idx]

df_movie[df_movie['MovieID'].isin(top5_movie_id)]
df_movie[df_movie['MovieID'].isin(rating_movie)]
