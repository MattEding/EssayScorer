import collections
import functools
import multiprocessing
import operator
import pathlib

import numpy as np
import pandas as pd

import nlp_util
import utils


#: User Input Prompts
NAME = input('Input data name: ').strip().lower()

logger = utils.get_logger(f'{NAME}_pos', __name__)

#: Directory Paths
data = pathlib.Path.cwd() / 'data'
npys = data / 'npys'
pkls = data / 'pkls'

#: Load Corrections Array
npy_corrs = npys / f'{NAME}_corrections.npy'
corrs = np.load(npy_corrs)

#: Load Prompt Similarity Arrays
# npy_pos = npys / f'{NAME}_pos.npy'
# if npy_pos.exists():
#     arr_pos = np.load(npy_pos)
# else:
#     arr_pos = np.empty(len(corrs), dtype=object)


def pos_range(start=0, stop=None):
    """TODO
    """

    logger.info(f'Start Index: {NAME} @ {start}')

    with multiprocessing.Pool() as pool:
        pos_counters = pool.map(nlp_util.parts_of_speech, corrs[start:stop])

    df_pos = pd.DataFrame(pos_counters)
    df_pos.to_pickle(pkls / f'{NAME}_pos.pkl')
    # arr_pos[:] = pos_counters
    # np.save(npy_pos, arr_pos)

    logger.info(f'Stop Index: {NAME} @ {stop}')


if __name__ == '__main__':
    pos_range()
