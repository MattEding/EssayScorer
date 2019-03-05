import functools
import logging
import pathlib


def get_logger(filename, name=None, **kwargs):
    """Return logger instance.

    Parameters
    ----------
    filename : str
        Stem of filename of log.
    name : str, optional
        Name of logger if specified, otherwise use root logger.
    **kwargs
        See logging.basicConfig

    Returns
    -------
    logger : Logger
        Logger with specified name or root logger.
    """

    logs = pathlib.Path.cwd() / 'data' / 'logs'
    log_file = logs / f'{filename}.log'
    log_file.touch()

    fmt = '{name} - {asctime} - {levelname} - Message: {message}'
    logging.basicConfig(filename=log_file, style='{', format=fmt, **kwargs)
    logger = logging.getLogger(name)
    return logger


def log_index(logger, fmt='{func} @ Index {i}'):
    """Decorator for enumerated function.

    Parameters
    ----------
    logger : Logger
        Logger instance to log index of enumerated function call.
    fmt : f-string
        Log message to be formatted with function name and index of enumeration.

    Returns
    -------
    func : function
        Decorated function. Expects enumerated inputs for function args.
    """

    def outer(func):
        @functools.wraps(func)
        def inner(i_args):
            print(i_args)
            i, *args = i_args
            try:
                result = func(*args)
            except BaseException as exc:
                logger.exception(exc)
                raise
            msg = fmt.format(func=func.__name__, i=i)
            logger.info(msg)
            return result
        return inner
    return outer


if __name__ == '__main__':
    import multiprocessing
    import pickle

    # TODO: figure out how to pickle decorated log func to pass into multiprocessing pool.

    # import dill
    # pickle.dump = dill.dump
    # pickle.dumps = dill.dumps
    # pickle.load = dill.load
    # pickle.loads = dill.loads

    logger = get_logger('TEST', __name__, filemode='w')

    chr = log_index(chr)

    with multiprocessing.Pool() as pool:
        results = pool.map(chr, enumerate(range(33, 127)))
        print(results)
