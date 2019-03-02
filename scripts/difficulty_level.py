import collections
import multiprocessing
import pathlib

import numpy as np
import pandas as pd

import utils.feature
import utils.log


def difficulty_level_range(name, start=0, stop=None):
    """Save estimated grade levels of all essays in the given range from start to stop.

    Parameters
    ----------
    start : int, optional
        Slice start index.

    stop : int, optional
        Slice stop index.

    func : function, optional
        Function used to assign a grade level score.

    Returns
    -------
    None
    """

    data = pathlib.Path.cwd() / 'data'
    npys = data / 'npys'
    pkls = data / 'pkls'

    corrections_npy = npys / f'{name}_corrections.npy'
    difficulty_level_pkl = pkls / f'{name}_difficulty_level.pkl'

    corrections_arr = np.load(corrections_npy)[start:stop]

    logger = utils.log.get_logger(f'{name}_difficulty_level', __name__)
    logger.info(f'Start Index: {name} @ {start}')

    with multiprocessing.Pool() as pool:
        difficulty_dicts = pool.map(utils.feature.difficulty_level, corrections_arr)

    difficulty_level_df = pd.DataFrame(collections.ChainMap(difficulty_dicts))
    difficulty_level_df.to_pickle(difficulty_level_pkl)

    logger.info(f'Stop Index: {name} @ {stop}')


if __name__ == '__main__':
    name = input('Input data name: ').strip().lower()
    difficulty_level_range(name)
