# The Algorithm Run in the App.

We implement the [`KNNWithMeans`](https://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithMeans) algorithm in Python package `surprise`.

In the training stage, for each movie, it finds the `K` most similar movies using the KNN algorithm. 
The similarity between two movies is defined as the `Cosine` similarity of the ratings of the two movies.


In the prediction phase, we need to predict the user $u$'s rating for each movie $i$. The intuition is that: the predicted rating for movie ($j$) is 1) the average rating of this movie by all the other users, and 2) adjusted by the limited set of movies ($j$) that the user $u$ already rate ($r_{uj}$) if the rated movie $j$ is a neighbour for movie $i$.


# How to Test the Movie Rating App in Local?

### Start the Backend Server

We require `Python 3.7+` to run the backend server.
```
cd MovieAppBackendServer
pip3 install -r requirements.txt
python3 manage.py runserver 7000 --settings MovieAppBackendServer.settings_dev
```

### Start the Frontend Server

We require `npm` installed to run the frontend server.

```
cd MovieAppFrontend
npm install -g npx
npm install
npx next dev
```

Then you can go to http://localhost:3000/ in your browser to play with the app.

# Public Links

Where is this code: https://github.com/Fan-Yang-Econ/UIUC_PSL/tree/main/UIUC_PSL/Project4
Public website: http://nasty-cat-movie.us-east-1.elasticbeanstalk.com

