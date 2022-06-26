from functools import wraps
import inspect
import logging
import sys
import traceback

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)


def log_it(func):
    @wraps(func)
    def exceptions(*args, **kwargs):
        frame = inspect.currentframe()
        _, _, _, values = inspect.getargvalues(frame)
        values.pop("frame")
        logger.info(f"Starting {func.__name__}")
        logger.debug(f"Starting {func.__name__} with\n{values}")
        try:
            wrapped = func(*args, **kwargs)
            logger.info(f"Completed {func.__name__}")
            logger.debug(f"Completed {func.__name__}")
            return wrapped
        except Exception as e:
            logger.error(f"{func.__name__} error")
            logger.error(traceback.format_exc())
    return exceptions
