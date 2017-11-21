# flake8: noqa
import logging

__version__ = '0.0.1'

def _init_logger():
    # logfmt = '%(message)s'
    logfmt = '%(levelname)s %(name)s(%(lineno)d): %(message)s'
    level = logging.DEBUG
    try:
        import coloredlogs
        # The colorscheme can be controlled by several environment variables
        # https://coloredlogs.readthedocs.io/en/latest/#environment-variables
        coloredlogs.install(level=level, fmt=logfmt)
    except ImportError:
        logging.basicConfig(format=logfmt, level=level)

_init_logger()

def getLogger(name):
    logger = logging.getLogger(name)
    logger.debug('initializing module: {}'.format(name))
    return logger

logger = getLogger(__name__)

logging.getLogger('PIL').setLevel(logging.INFO)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.INFO)
logging.getLogger('parse').setLevel(logging.INFO)
