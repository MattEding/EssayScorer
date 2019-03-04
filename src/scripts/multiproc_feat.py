import collections
import logging
import multiprocessing
import pathlib

import numpy as np
import pandas as pd

import src.utils.feature
import src.utils.log


__all__ = []


name = input('Input data name: ').strip().lower()

data = pathlib.Path.cwd() / 'data'
npys = data / 'npys'
pkls = data / 'pkls'

descr_pkl = pkls / f'descr.pkl'
prompts = pd.read_pickle(descr_pkl)['prompt']

dataset_pkl = pkls / f'{name}.pkl'
dataset_df = pd.read_pickle(dataset_pkl)

essay_originals = dataset_df['essay']
essay_set_counts = dataset_df['essay_set'].value_counts()

essay_set_idxs = np.array([0] + [essay_set_counts[i] for i in range(1, 9)]).cumsum()
essay_set_ranges = tuple(range(start, stop) for start, stop in zip(essay_set_idxs, essay_set_idxs[1:]))
clean_prompts = [src.utils.feature._clean(prompt) for prompt in prompts]


def find_prompt(index, ranges):
    for i, range_ in enumerate(ranges):
        if index in range_:
            return i
    raise ValueError('index not in any range')


def prompt_similarity(i_correction):
    i, correction = i_correction
    prompt = clean_prompts[find_prompt(i, essay_set_ranges)]
    similarity_dict = src.utils.feature.similarity(prompt, correction)
    return similarity_dict


def error_ratio(i_correction):
    i, correction = i_correction
    original = essay_originals.iloc[i]
    error_ratio_dict = src.utils.feature.error_ratio(original, correction)
    return error_ratio_dict


def main():
    corrections_npy = npys / f'{name}_corrections.npy'
    corrections_arr = np.load(corrections_npy)

    features = [src.utils.feature.difficulty_level, error_ratio, src.utils.feature.words,
                src.utils.feature.pos, src.utils.feature.sentiment, prompt_similarity]
    print({i: func.__name__ for i, func in enumerate(features)})
    idx = input('Choose feature function: ')
    feature = features[int(idx)]

    logger = src.utils.log.get_logger(f'{name}_{feature.__name__}', __name__, level=logging.INFO)
    logger.info(f'STARTED')

    if (feature is prompt_similarity) or (feature is error_ratio):
        corrections_arr = enumerate(corrections_arr)

    with multiprocessing.Pool() as pool:
        feature_dicts = pool.map(feature, corrections_arr)

    feature_pkl = pkls / f'{name}_{feature.__name__}.pkl'
    feature_df = pd.DataFrame(feature_dicts)
    feature_df.to_pickle(feature_pkl)

    logger.info(f'FINISHED')


main()
