import pandas as pd
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 40)


FOLDER = '/Users/fanyang/Dropbox/uiuc/cs598/UIUC_SPL/UIUC_PSL/Project4/'
data = 'movies.csv'
df_movie = pd.read_csv(os.path.join(FOLDER, data))
df_rating = pd.read_csv(os.path.join(FOLDER, 'ratings.csv'))

grp_rating = df_rating.groupby('MovieID').agg({'Rating': ['count', 'mean'] })
grp_rating.columns = grp_rating.columns.droplevel(0)
grp_rating = grp_rating.reset_index()
grp_rating = grp_rating.rename(columns = {'count': 'rating_ct', 'mean': 'avg_rating'})

df_movie_rating = df_movie.merge(grp_rating, 'left', on = 'MovieID')

genre_list = ["Action", "Adventure", "Animation",
               "Children's", "Comedy", "Crime",
               "Documentary", "Drama", "Fantasy",
               "Film-Noir", "Horror", "Musical",
               "Mystery", "Romance", "Sci-Fi",
               "Thriller", "War", "Western"]
genre_summary_list = []
for genre in genre_list:
    df_movie_rating[genre] = df_movie_rating['Genres'].apply(lambda x: 1 if genre in x else 0)
    top5_rating_ct_id = df_movie_rating[df_movie_rating[genre] == 1].sort_values('rating_ct', ascending=False)['MovieID'].iloc[:5,].tolist()
    top5_rating_id = df_movie_rating[df_movie_rating[genre] == 1].sort_values('avg_rating', ascending=False)['MovieID'].iloc[:5,].tolist()
    top5_rating_ct_name = df_movie_rating[df_movie_rating[genre] == 1].sort_values('rating_ct', ascending=False)['Title'].iloc[:5,].tolist()
    top5_rating_name = df_movie_rating[df_movie_rating[genre] == 1].sort_values('avg_rating', ascending=False)['Title'].iloc[:5,].tolist()
    summary = {
        'Genres': genre,
        'Top5_most_rating_id': top5_rating_ct_id,
        'Top5_highest_rating_id': top5_rating_id,
        'Top5_most_rating_name': top5_rating_ct_name,
        'Top5_highest_rating_name': top5_rating_name
    }
    genre_summary_list.append(summary)

df_top5 = pd.DataFrame.from_dict(genre_summary_list)
