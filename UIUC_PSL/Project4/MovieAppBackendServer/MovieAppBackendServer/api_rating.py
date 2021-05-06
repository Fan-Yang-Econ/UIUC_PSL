"""
"""
import re
import logging
import json
from pprint import pprint
from collections import OrderedDict
import math
import random

import numpy as np
import pandas as pd
from django.http import HttpResponse
from MovieAppBackendServer.data.load_data import df_nn, df_rating_avg

NUMBER_MOVIES_TO_RECOM = 10
MIN_VOTE = 10


def is_empty(x):
    if x is None:
        return True
    elif x is np.nan or (type(x) is float and math.isnan(x)):
        return True
    else:
        try:
            if not len(x):
                return True
        except TypeError:
            pass

    return False


def delete_empty_v_in_obj(dict_like_obj):
    keys_to_delete = []
    if type(dict_like_obj) in [dict, OrderedDict]:
        for k, v in dict_like_obj.items():
            if is_empty(v):
                keys_to_delete.append(k)
            elif type(v) in [dict, OrderedDict, list]:
                dict_like_obj[k] = delete_empty_v_in_obj(v)

        for k in keys_to_delete:
            del dict_like_obj[k]

        return dict_like_obj

    elif type(dict_like_obj) in [list]:
        new_list = []
        for v_i in dict_like_obj:
            v_i = delete_empty_v_in_obj(v_i)
            if not is_empty(v_i):
                new_list.append(v_i)
        return new_list

    return dict_like_obj


def recommend_movie(dict_new_rating, df_nn, df_rating_avg):
    """

    :param dict_new_rating:
    rating_movie = {2905: 4, 2019: 5, 858: 4, 1198: 5, 260: 5}
    :param df_nn:
    :param df_rating_avg:
    :return:
    """

    df_nn_ = df_nn[df_nn['MovieID'].isin(dict_new_rating.keys())]
    # find movies with the neighbours that include the `rating_movie`
    df_nn_['rating'] = df_nn_['MovieID'].apply(lambda x: dict_new_rating[x])
    df_nn_.duplicated(['MovieID', 'Whose_NN']).sum()

    LIST_NEW_RATING = []
    for Whose_NN, df_nn_rated in df_nn_.groupby(['Whose_NN']):
        avg_rating = df_rating_avg[df_rating_avg['MovieID'] == Whose_NN]['avg_rating'].iloc[0]
        rating_adj = (df_nn_rated['similarity'] * (df_nn_rated['rating'] - avg_rating)).sum() / df_nn_rated['similarity'].sum()
        new_rating = avg_rating + rating_adj
        LIST_NEW_RATING.append({
            'MovieID': Whose_NN,
            'avg_rating': new_rating,
            'populous_rating': avg_rating,
            'rating_adj': rating_adj,
            'rated_movies': df_nn_rated['MovieID'].unique().tolist()
        })

    df_new_rating = pd.DataFrame(LIST_NEW_RATING)
    # df_new_rating['avg_rating'].unique()
    df_new_rating['rating_source'] = 'new'

    df_rating_avg_populous = df_rating_avg[df_rating_avg['count'] > MIN_VOTE]
    df_rating_avg_populous = df_rating_avg_populous[~df_rating_avg_populous['MovieID'].isin(df_new_rating['MovieID'])]
    df_rating_avg_populous = df_rating_avg_populous[~df_rating_avg_populous['MovieID'].isin(dict_new_rating.keys())]
    df_rating_avg_populous = df_rating_avg_populous.iloc[[random.randint(0, 200) for i in range(NUMBER_MOVIES_TO_RECOM)],][
        ['MovieID', 'avg_rating']]
    df_rating_avg_populous['rating_source'] = 'old'
    df_rating_mix = pd.concat([
        df_new_rating,
        df_rating_avg_populous
    ])
    print('\n New Ratings ')
    print(df_new_rating)

    df_rating_mix = df_rating_mix.sort_values('avg_rating', ascending=False)

    return delete_empty_v_in_obj(df_rating_mix.iloc[0:NUMBER_MOVIES_TO_RECOM].to_dict('records'))

def api_rating(request):
    """

    from pprint import pprint
    from PyHelpers import set_logging; set_logging(10)

    import os
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "MovieAppBackendServer.settings")

    from django.test import RequestFactory
    from django.conf import settings

    settings.configure()
    request_factory = RequestFactory()

    rating1 = 1
    movie1 = 213
    request = request_factory.get(f'http://127.0.0.1:7000/api_genre?rating1={rating1}&movie1={movie1}')

    :param request:
    :return:
    """
    logging.info(request.GET)

    DICT_REQUEST = {}
    for i in request.GET:
        key_ = re.compile('\d+$').sub('', i)
        order_id = re.compile('\d+$').findall(i)[0]
        if order_id not in DICT_REQUEST:
            DICT_REQUEST[order_id] = {}

        DICT_REQUEST[order_id][key_] = request.GET[i]

    list_movie_ratings = list(DICT_REQUEST.values())
    dict_new_rating = {}
    for i in list_movie_ratings:
        dict_new_rating[int(i['movie'])] = int(i['rating'])

    list_recom = recommend_movie(dict_new_rating, df_nn, df_rating_avg)

    pprint(list_recom)

    response = HttpResponse(
        json.dumps(list_recom),
        content_type="text/plain")
    response['Access-Control-Allow-Origin'] = '*'

    return response
