import collections
import functools
import multiprocessing
import pathlib
import reprlib

import numpy as np
import pandas as pd
import textstat

import utils


NAME = input('Input data name: ').strip().lower()
logger = utils.get_logger(f'{NAME}_difficulty_level', __name__)
reprlib.aRepr.maxstring = 70

#: Directory Paths
data = pathlib.Path.cwd() / 'data'
npys = data / 'npys'
pkls = data / 'pkls'

#: Load Corrections Array
npy_corrs = npys / f'{NAME}_corrections.npy'
corrs = np.load(npy_corrs)


def difficulty_level(text):
    """Return the estimated school grade level required to understand the text.

    Parameters
    ----------
    i_text : (int, str)
        Pair containing index and string.
    func : function, optional
        Function used to assign a grade level score.

    Returns
    -------
    i_difficulty_level : (int, float)
        Pair of ith text estimated grade level to understand.
    """

    try:
        difficulty_level = [f(text) for f in nlp_util.diff_funcs]
        difficulty_level = nlp_util.DifficultyLevel(*difficulty_level)
    except BaseException as exc:
        logger.exception(exc)
        raise
    else:
        logger.info(f'{NAME} : {reprlib.repr(text)}')
    return difficulty_level._asdict()


def difficulty_level_range(start=0, stop=None):
    """Save estimated grade levels of all essays in the given range from start to stop.

    Parameters
    ----------
    start : int, optional
        Slice start index.

    stop : int, optional
        Slice stop index.

    func : function, optional
        Function used to assign a grade level score.
    """

    logger.info(f'Start Index: {NAME} @ {start}')

    with multiprocessing.Pool() as pool:
        difficulty_levels = pool.map(difficulty_level, corrs[start:stop])

    df_difficulty_levels = pd.DataFrame(difficulty_levels)
    df_difficulty_levels.to_pickle(pkls / f'{NAME}_difficulty_level.pkl')

    logger.info(f'Stop Index: {NAME} @ {stop}')


if __name__ == '__main__':
    difficulty_level_range()
