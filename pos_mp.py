import multiprocessing
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


def pos_range(start=0, stop=None):
    """TODO
    """

    logger.info(f'Start Index: {NAME} @ {start}')

    with multiprocessing.Pool() as pool:
        pos_counters = pool.map(nlp_util.parts_of_speech, corrs[start:stop])

    df_pos = pd.DataFrame(pos_counters)
    df_pos.to_pickle(pkls / f'{NAME}_pos.pkl')

    logger.info(f'Stop Index: {NAME} @ {stop}')


if __name__ == '__main__':
    pos_range()
