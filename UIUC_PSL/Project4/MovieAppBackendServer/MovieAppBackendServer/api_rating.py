"""
"""

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

	genre_name = 'Action'
	page = 1
	request = request_factory.get(f'http://127.0.0.1:7000/api_genre?genre_name={genre_name}')

	:param request:
	:return:
	"""
	logging.info(request.GET)

	genre = request.GET.get('genre')
