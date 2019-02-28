import functools
import multiprocessing
import operator
import pathlib

import numpy as np

import nlp_util
import utils


#: User Input Prompts
NAME = input('Input data name: ').strip().lower()

logger = utils.get_logger(f'{NAME}_pos', __name__)

#: Directory Paths
data = pathlib.Path.cwd() / 'data'
npys = data / 'npys'

#: Load Corrections Array
npy_corrs = npys / f'{NAME}_corrections.npy'
corrs = np.load(npy_corrs)

#: Load Prompt Similarity Arrays
npy_pos = npys / f'{NAME}_polarity.npy'
if npy_pos.exists():
    arr_pos = np.load(npy_pos)
else:
    arr_pos = np.empty(len(corrs), dtype=object)


def pos_range(start=0, stop=None):
    """TODO
    """

    logger.info(f'Start Index: {NAME} @ {start}')

    with multiprocessing.Pool() as pool:
        pos_counters = pool.map(nlp_util.parts_of_speech, corrs[start:stop])

    arr_polarity[:] = pos_counters
    np.save(npy_pos, arr_pos)

    pos_keys = (c.keys() for c in pos_counters)
    all_pos = functools.reduce(operator.or_, pos_keys, set())
    print(all_pos)

    logger.info(f'Stop Index: {NAME} @ {stop}')


if __name__ == '__main__':
    pos_range()
