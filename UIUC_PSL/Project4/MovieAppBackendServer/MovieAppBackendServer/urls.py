from django.contrib import admin
from django.conf.urls import url

from MovieAppBackendServer.api_rating import api_rating
from MovieAppBackendServer.api_genre import api_genre

urlpatterns = [
    url('admin/', admin.site.urls),
    url(rf'^{api_rating.__name__}?', api_rating, name='api_rating'),
    url(rf'^{api_genre.__name__}?', api_genre, name='api_genre'),
]
