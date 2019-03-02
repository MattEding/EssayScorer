import collections
import multiprocessing
import pathlib

import numpy as np
import pandas as pd

import utils.feature
import utils.log


def main():
    data = pathlib.Path.cwd() / 'data'
    npys = data / 'npys'
    pkls = data / 'pkls'

    name = input('Input data name: ').strip().lower()
    corrections_npy = npys / f'{name}_corrections.npy'
    corrections_arr = np.load(corrections_npy)

    features = [utils.feature.difficulty_level, utils.feature.error_ratio,
             utils.feature.pos, utils.feature.sentiment, utils.feature.similarity]
    print({i: func.__name__ for i, func in enumerate(features)})
    idx = input('Choose feature function: ')
    feature = features[int(idx)]

    logger = utils.log.get_logger(f'{name}_{feature.__name__}', __name__)
    logger.info(f'Start Index: {name} @ {start}')

    with multiprocessing.Pool() as pool:
        feature_dicts = pool.map(feature, corrections_arr)

    feature_chain_map = collections.ChainMap(*feature_dicts)
    feature_pkl = pkls / f'{name}_{feature.__name__}.pkl'
    feature_df = pd.DataFrame(feature_chain_map)
    feature_df.to_pickle(feature_pkl)

    logger.info(f'Stop Index: {NAME} @ {stop}')


if __name__ == '__main__':
    main()
