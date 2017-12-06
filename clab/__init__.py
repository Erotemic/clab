# flake8: noqa
import logging

__version__ = '0.0.1'


_this_srcfile = __file__


class CustomLogger(logging.getLoggerClass()):
    def __init__(self, name, level=logging.NOTSET):
        super(CustomLogger, self).__init__(name, level)

    def findCaller(self, stack_info=False):
        """
        Find the stack frame of the caller so that we can note the source
        file name, line number and function name.

        This function comes straight from the original python one
        """
        import os
        f = logging.currentframe()
        #On some versions of IronPython, currentframe() returns None if
        #IronPython isn't run with -X:Frames.
        if f is not None:
            f = f.f_back
        rv = "(unknown file)", 0, "(unknown function)", None
        while hasattr(f, "f_code"):
            co = f.f_code
            filename = os.path.normcase(co.co_filename)
            # Hack to disable logging in one specific util/misc func
            if f.f_code.co_name in ['protected_print']:
                f = f.f_back
                continue
            if filename == [logging._srcfile, _this_srcfile]:
                f = f.f_back
                continue
            sinfo = None
            if stack_info:
                sio = io.StringIO()
                sio.write('Stack (most recent call last):\n')
                traceback.print_stack(f, file=sio)
                sinfo = sio.getvalue()
                if sinfo[-1] == '\n':
                    sinfo = sinfo[:-1]
                sio.close()
            rv = (co.co_filename, f.f_lineno, co.co_name, sinfo)
            break
        return rv

logging.setLoggerClass(CustomLogger)


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
