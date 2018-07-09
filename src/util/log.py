import logging


def log_exception(name):
    """Record Exceptions

    :param name: logging.getLogger(name)
    :return:
    """
    def decorator(func):
        logger = logging.getLogger(name)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(e)
        return wrapper
    return decorator
