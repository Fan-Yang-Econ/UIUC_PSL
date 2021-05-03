import logging
import json
from django.http import HttpResponse

GENRE_SUMMARY_DICT = {'Action': [260, 1196, 1210, 480, 2028], 'Adventure': [260, 1196, 1210, 480, 1580], 'Animation': [1, 2987, 2355, 3114, 588],
                      "Children's": [1097, 1, 34, 919, 2355], 'Comedy': [2858, 1270, 1580, 2396, 1197], 'Crime': [608, 1617, 858, 296, 50],
                      'Documentary': [2064, 246, 162, 3007, 1147], 'Drama': [2858, 1196, 2028, 593, 608], 'Fantasy': [260, 1097, 2628, 2174, 2797],
                      'Film-Noir': [1617, 541, 2987, 1252, 913], 'Horror': [2716, 1214, 1387, 1219, 2710], 'Musical': [919, 588, 1220, 2657, 364],
                      'Mystery': [1617, 924, 648, 3176, 1252], 'Romance': [1210, 2396, 1197, 1265, 356], 'Sci-Fi': [260, 1196, 1210, 480, 589],
                      'Thriller': [589, 2571, 593, 608, 2762], 'War': [1196, 1210, 2028, 110, 527], 'Western': [590, 1304, 2012, 3671, 1266]}


def api_genre(request):
	"""

	from pprint import pprint
	from PyHelpers import set_logging; set_logging(10)

	import os
	os.environ.setdefault("DJANGO_SETTINGS_MODULE", "MovieAppBackendServer.settings")

	from django.test import RequestFactory
	from django.conf import settings

	settings.configure()
	request_factory = RequestFactory()

	genre_name = 'Action'
	page = 1
	request = request_factory.get(f'http://127.0.0.1:7000/api_genre?genre_name={genre_name}')

	:param request:
	:return:
	"""
	logging.info(request.GET)

	genre_name = request.GET.get('genre_name')

	response = HttpResponse(
		json.dumps(GENRE_SUMMARY_DICT[genre_name]),
		content_type="text/plain")
	response['Access-Control-Allow-Origin'] = '*'

	return response