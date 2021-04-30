"""WarrensDjango URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.conf.urls import url
from MovieAppBackendServer.api_rating import api_rating
from MovieAppBackendServer.api_genre import api_genre

#
# sys.path.extend(['/Users/yafa/Dropbox/WarrensBackend/WarrensDataAccess',
#                  '/Users/yafa/Dropbox/WarrensBackend/PyHelpers'])


urlpatterns = [
	url('admin/', admin.site.urls),
	url(rf'^{api_rating.__name__}?', api_rating, name='api_rating'),
	url(rf'^{api_genre.__name__}?', api_genre, name='suggest'),
	# url(r'^p?', return_one_page, name='return_one_page'),
]
