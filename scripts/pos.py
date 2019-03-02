import multiprocessing
import pathlib

import numpy as np
import pandas as pd

import utils.feature
import utils.log


def pos_range(name, start=0, stop=None):
    """TODO
    """

    data = pathlib.Path.cwd() / 'data'
    npys = data / 'npys'
    pkls = data / 'pkls'

    corrections_npy = npys / f'{name}_corrections.npy'
    sentiment_pkl = pkls / f'{name}_sentiment.pkl'

    corrections_arr = np.load(corrections_npy)[start:stop]

    logger = utils.log.get_logger(f'{name}_sentiment', __name__)
    logger.info(f'Start Index: {name} @ {start}')

    with multiprocessing.Pool() as pool:
        pos_counters = pool.map(utils.feature.parts_of_speech, corrections_arr)

    df_pos = pd.DataFrame(pos_counters)
    df_pos.to_pickle(pkls / f'{NAME}_pos.pkl')

    logger.info(f'Stop Index: {NAME} @ {stop}')


if __name__ == '__main__':
    pos_range()
