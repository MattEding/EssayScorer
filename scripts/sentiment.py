import multiprocessing
import pathlib

import numpy as np
import pandas as pd

import utils.feature
import utils.log


def sentiment_range(name, start=0, stop=None):
    """Save sentiment of all essays in the given range from start to stop.

    Parameters
    ----------
    start : int, optional
        Slice start index.

    stop : int, optional
        Slice stop index.
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
        sentiments_dicts = pool.map(utils.feature.sentiment, corrections_arr)

    df_sentiment = pd.DataFrame([sentiments_dicts])
    df_sentiment.to_pickle(pkls / f'{NAME}_sentiment.pkl')

    logger.info(f'Stop Index: {NAME} @ {stop}')


if __name__ == '__main__':
    name = input('Input data name: ').strip().lower()
    sentiment_range(name)
