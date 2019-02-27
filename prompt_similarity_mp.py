import multiprocessing
import pathlib

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import nlp_util
import utils


#: User Input Prompts
NAME = input('Input data name: ')
START = int(input('Input start index: '))
try:
    STOP = int(input('Input stop index: '))
except Exception:
    STOP = None

    
logger = utils.get_logger(f'{NAME}_prompt', __name__)


#: Directory Paths
data = pathlib.Path.cwd() / 'data'
pkls = data / 'pkls'
npys = data / 'npys'


#: Load Prompts
pkl = pkls / f'descr.pkl'
prompts = pd.read_pickle(pkl)['prompt']


#: Load Essay Set Ranges
pkl = pkls / f'{NAME}.pkl'
essay_set_counts = pd.read_pickle(pkl)['essay_set'].value_counts()
essay_set_idxs = np.array([0] + [essay_set_counts[i] for i in range(1, 9)]).cumsum()
essay_set_ranges = tuple(range(start, stop) for start, stop in zip(essay_set_idxs, essay_set_idxs[1:]))


#: Load Corrections Array
npy_corrs = npys / f'{NAME}_corrections.npy'
corrs = np.load(npy_corrs)


#: Load Prompt Similarity Arrays
npy_count = npys / f'{NAME}_prompt_count.npy'
if npy_count.exists():
    arr_count = np.load(npy_count)
else:
    arr_count = np.empty(len(corrs))
    arr_count.fill(np.nan)
    
npy_tfidf = npys / f'{NAME}_prompt_tfidf.npy'
if npy_tfidf.exists():
    arr_tfidf = np.load(npy_tfidf)
else:
    arr_tfidf = np.empty(len(corrs))
    arr_tfidf.fill(np.nan)


def clean(text):
    """Return cleaned text with stopwords removed, no punctuation, and lemmatized.
    
    Parameters
    ----------
    text : str
        Text string.
    
    Returns
    -------
    cleaned : str
        Cleaned string.
    """
    
    lower_proper = nlp_util.lower_with_proper(text)
    lemmas = nlp_util.lemmatize(lower_proper)
    cleaned = nlp_util.clean_stopwords_punctuation(lemmas)
    return cleaned


clean_prompts = [clean(prompt) for prompt in prompts]


def find_prompt(index, ranges):
    """Return i of range in ranges that contains provided index.
    
    Parameters
    ----------
    index : int
        Index of essay.
    ranges : container of ranges
        Container of ranges to search index inside.
    
    Returns
    -------
    i : int
        Index of range containing index of essay.
    
    Raises
    ------
    ValueError
        If provided index is not in any of the ranges.
    """
    
    for i, range_ in enumerate(ranges):
        if index in range_:
            return i
    raise ValueError('index not in any range')
    
    
def assign_similarity(i_essay):
    """Return cosine similarities of the ith essay to its prompt.
    
    Parameters
    ----------
    i_essay : (int, str)
        Pair containing index and string.
    
    Returns
    -------
    i_count_tfidf : (i, float, float)
        Triple tuple of ith essay cosine similarity measures using CountVectorizer and TfidfVectorizer. 
    """
    
    i, essay = i_essay
    clean_prompt = clean_prompts[find_prompt(i, essay_set_ranges)]
    clean_essay = clean(essay)
    
    try:
        count_meas = nlp_util.prompt_similarity(clean_prompt, clean_essay, vectorizer=CountVectorizer)
        tfidt_meas = nlp_util.prompt_similarity(clean_prompt, clean_essay, vectorizer=TfidfVectorizer)
    except BaseException as exc:
        logger.exception(exc)
        raise
    else:
        logger.info(f'Measured Cosine Similarity Index: {NAME} @ {i}')
    return i, count_meas, tfidt_meas


def assign_similarities_range(start=0, stop=None):
    """Save cosine similarities of all essays in the given range from start to stop.
    
    Parameters
    ----------
    start : int, optional
        Slice start index.
    
    stop : int, optional
        Slice stop index.
    """
    
    logger.info(f'Start Index: {NAME} @ {start}')
    
    with multiprocessing.Pool() as pool:
        i_count_tfidf = pool.map(assign_similarity, enumerate(corrs[START:STOP], start=start))
    
    idxs, counts, tfidfs = map(list, zip(*i_count_tfidf))
    
    arr_count[idxs] = counts
    np.save(npy_count, arr_count)

    arr_tfidf[idxs] = tfidfs
    np.save(npy_tfidf, arr_tfidf)
    
    logger.info(f'Stop Index: {NAME} @ {stop}')
    
    
if __name__ == '__main__':
    assign_similarities_range(START, STOP)
