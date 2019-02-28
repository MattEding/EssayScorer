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
logger = utils.get_logger(f'{NAME}_grade_level', __name__)
reprlib.aRepr.maxstring = 70

#: Directory Paths
data = pathlib.Path.cwd() / 'data'
npys = data / 'npys'
pkls = data / 'pkls'

#: Load Corrections Array
npy_corrs = npys / f'{NAME}_corrections.npy'
corrs = np.load(npy_corrs)

#: Setup Grade Level Funcs
_funcs = [textstat.flesch_reading_ease, textstat.smog_index,
          textstat.flesch_kincaid_grade, textstat.coleman_liau_index,
          textstat.automated_readability_index, textstat.dale_chall_readability_score,
          textstat.linsear_write_formula, textstat.gunning_fog]

GradeLevel = collections.namedtuple('GradeLevel', [f.__name__ for f in _funcs])


def grade_level(text):
    """Return the estimated school grade level required to understand the text.

    Parameters
    ----------
    i_text : (int, str)
        Pair containing index and string.
    func : function, optional
        Function used to assign a grade level score.

    Returns
    -------
    i_grade_level : (int, float)
        Pair of ith text estimated grade level to understand.
    """

    try:
        grade_level = [f(text) for f in _funcs]
        grade_level = GradeLevel(*grade_level)
    except BaseException as exc:
        logger.exception(exc)
        raise
    else:
        logger.info(f'{NAME} : {reprlib.repr(text)}')
    return grade_level._asdict()


def grade_level_range(start=0, stop=None):
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
        grade_levels = pool.map(grade_level, corrs[start:stop])

    df_grade_levels = pd.DataFrame(grade_levels)
    df_grade_levels.to_pickle(pkls / f'{NAME}_grade_level.pkl')

    logger.info(f'Stop Index: {NAME} @ {stop}')


if __name__ == '__main__':
    grade_level_range()
