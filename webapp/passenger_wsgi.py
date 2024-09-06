import sys
import os
import logging

logging.basicConfig(stream=sys.stderr)
sys.path.insert(0, '/home/evulotyh/mood-sleuth')

from app import app as application