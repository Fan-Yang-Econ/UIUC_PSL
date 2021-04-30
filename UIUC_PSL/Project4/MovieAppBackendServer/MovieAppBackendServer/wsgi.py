"""
WSGI config for WarrensDjango project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.0/howto/deployment/wsgi/
"""

import os
import sys

from django.core.wsgi import get_wsgi_application
sys.path.extend(['/Users/yafa/Dropbox/WarrensBackend/WarrensDataAccess','/Users/yafa/Dropbox/WarrensBackend/PyHelpers'])

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "WarrensDjango.settings")

application = get_wsgi_application()
