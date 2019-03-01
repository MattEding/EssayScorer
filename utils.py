import io
import itertools
import logging
import pathlib
import pickle

import docx
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.distance import cosine


data = pathlib.Path.cwd() / 'data'
npys = data / 'npys'
pkls = data / 'pkls'


def get_logger(filename, module_name):
    logs = pathlib.Path.cwd() / 'data' / 'logs'

    log_file = logs / f'{filename}.log'
    log_file.touch()
    fmt = '{name} - {asctime} - {levelname} - Message: {message}'
    logging.basicConfig(filename=log_file,
                        level=logging.INFO,
                        style='{',
                        format=fmt)
    logger = logging.getLogger(module_name)
    return logger


def public(obj):
    """Return list of public attributes of obj.

    Parameters
    ----------
    obj : object
        Object to inspect public attributes of.

    Returns
    -------
    attrs : list of strings
        List of public attributes of object
    """
    return [attr for attr in dir(obj) if not attr.startswith('_')]


def stream_extract(zipfile, *file_indices):
    """Extract files without saving them to disk.

    Parameters
    ----------
    zipfile : ZipFile
        ZipFile object to extract info from.
    *file_indices : int
        Indices of the files to extract from zipfile.

    Returns
    -------
    files: generator of files
        Generator yielding extracted files.
    """

    files = (zipfile.filelist[idx] for idx in file_indices)
    files = (zipfile.read(file) for file in files)
    files = (io.BytesIO(file) for file in files)
    return files


def get_prompt(readme):
    """Return prompt of essay from a docx file.

    Parameters
    ----------
    readme : docx file
        Readme file of an essay.

    Returns
    -------
    prompt : str
        String of the essay prompt.
    """

    doc = docx.Document(readme)
    pred_drop = lambda s: 'Prompt' not in s
    pred_take = lambda s: 'Rubric' not in s
    prompts = (p.text for p in doc.paragraphs)
    prompts = itertools.dropwhile(pred_drop, prompts)
    next(prompts)
    prompts = itertools.takewhile(pred_take, prompts)
    return '\n'.join(prompts)


def interpolate(df, col_min, col_max, new_min=0, new_max=100):
    maps = []
    for _, series in df.iterrows():
        old_interval = series[[col_min, col_max]]
        new_interval = [new_min, new_max]
        map_ = interp1d(old_interval, new_interval)
        maps.append(map_)
    return maps


def cosine_similarity(u, v, w=None):
    return 1 - cosine(u, v, w)


def merge_features(name):
    """Merge all serialized features into a single dataframe.

    Parameters
    ----------
    name : str
        'train' or 'valid'

    Returns
    -------
    df : DataFrame
        DataFrame with all the features combined.
    """

    essay_df = pd.read_pickle(pkls / f'{name}.pkl')
    descr_df = pd.read_pickle(pkls / 'descr.pkl')[['essay_set', 'grade_level']]
    essay_to_grade_level = descr_df.set_index('essay_set').to_dict()['grade_level']

    grade_level_arr = essay_df['essay_set'].map(essay_to_grade_level).values
    grade_level_df = pd.DataFrame(grade_level_arr, columns=['grade_level'])

    promt_count_arr = np.load(npys / f'{name}_prompt_count.npy')
    promt_count_df = pd.DataFrame(promt_count_arr, columns=['prompt_count'])

    promt_tfidf_arr = np.load(npys / f'{name}_prompt_tfidf.npy')
    promt_tfidf_df = pd.DataFrame(promt_tfidf_arr, columns=['prompt_tfidf'])

    pos_df = pd.read_pickle(pkls / f'{name}_pos.pkl')
    pos_df = pos_df.div(pos_df.sum(axis=1), axis=0)

    percent_df = essay_df[['domain1_percent', 'domain2_percent']]
    sentiment_df = pd.read_pickle(pkls / f'{name}_sentiment.pkl')
    diff_level_df = pd.read_pickle(pkls / f'{name}_grade_level.pkl')

    dfs = [grade_level_df, percent_df, promt_count_df, promt_tfidf_df,
           diff_level_df, sentiment_df, pos_df]
    return pd.concat(dfs, axis=1)
