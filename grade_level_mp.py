import functools
import multiprocessing
import pathlib

import numpy as np
import textstat

import utils


#: User Input Prompts
NAME = input('Input data name: ').strip().lower()

text_standard = functools.partial(textstat.text_standard, float_output=True)
text_standard.__name__ = textstat.text_standard.__name__

funcs = [textstat.flesch_reading_ease, textstat.smog_index,
         textstat.flesch_kincaid_grade, textstat.coleman_liau_index,
         textstat.automated_readability_index, textstat.dale_chall_readability_score,
         textstat.linsear_write_formula, textstat.gunning_fog, text_standard]
choices = list(enumerate(f.__name__ for f in funcs))
print(choices)
idx = int(input('Enter index of function: '))
FUNC = funcs[idx]


logger = utils.get_logger(f'{NAME}_grade_level_{FUNC.__name__}', __name__)


#: Directory Paths
data = pathlib.Path.cwd() / 'data'
npys = data / 'npys'


#: Load Corrections Array
npy_corrs = npys / f'{NAME}_corrections.npy'
corrs = np.load(npy_corrs)


#: Load Prompt Similarity Arrays
npy_grade = npys / f'{NAME}_grade_level_{FUNC.__name__}.npy'
if npy_grade.exists():
    arr_grade = np.load(npy_grade)
else:
    arr_grade = np.empty(len(corrs))
    arr_grade.fill(np.nan)


def grade_level(i_text, func=text_standard):
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

    i, text = i_text
    try:
        grade_level = func(text)
    except BaseException as exc:
        logger.exception(exc)
        raise
    else:
        logger.info(f'Estimated Grade Level Index: {NAME} @ {i}')
    return i, grade_level


def grade_level_range(start=0, stop=None, func=text_standard):
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

    grade_level_partial = functools.partial(grade_level, func=func)
    with multiprocessing.Pool() as pool:
        i_grade_levels = pool.map(grade_level_partial,
                                  enumerate(corrs[start:stop], start=start))

    idxs, grade_levels = map(list, zip(*i_grade_levels))

    arr_grade[idxs] = grade_levels
    np.save(npy_grade, arr_grade)

    logger.info(f'Stop Index: {NAME} @ {stop}')


if __name__ == '__main__':
    grade_level_range(0, None, FUNC)
