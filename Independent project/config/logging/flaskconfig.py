import os
import logging
from os import path

# logging configurations
logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %I:%M:%S %p',
                    level=logging.INFO)

DEBUG = True
PORT = 5000
APP_NAME = "LCG App"
HOST = "0.0.0.0"

LOGGING_CONFIG = 'config/logging/logging.conf'

