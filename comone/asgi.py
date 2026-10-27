"""
ASGI config for comone project.
Supports HTTP and real-time streaming for Asterisk AudioSocket / WebSockets.
"""
import os
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'comone.settings')
application = get_asgi_application()
