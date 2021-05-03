"""
"""
import re
import logging


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

	DICT_RATINGS = {}
	for i in request.GET:
		key_ = re.compile('\d+$').sub('', i)
		order_id = re.compile('\d+$').findall(i)[0]
		if order_id not in DICT_RATINGS:
			DICT_RATINGS[order_id] = {}

		DICT_RATINGS[order_id][key_] = request.GET[i]

	list_movie_ratings = list(DICT_RATINGS.values())
