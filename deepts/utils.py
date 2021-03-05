
import os
import numpy as np


def set_logging():
    import logging
    logging.basicConfig(level = logging.INFO,
        format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')
    return logging
