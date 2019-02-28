import functools
import multiprocessing
import pathlib

import numpy as np
import textblob

import nlp_util
import utils


#: User Input Prompts
NAME = input('Input data name: ').strip().lower()

logger = utils.get_logger(f'{NAME}_sentiment', __name__)

#: Directory Paths
data = pathlib.Path.cwd() / 'data'
pkls = data / 'pkls'
npys = data / 'npys'

#: Load Corrections Array
npy_corrs = npys / f'{NAME}_corrections.npy'
corrs = np.load(npy_corrs)

#: Load Prompt Similarity Arrays
npy_polarity = npys / f'{NAME}_polarity.npy'
if npy_polarity.exists():
    arr_polarity = np.load(npy_polarity)
else:
    arr_polarity = np.empty(len(corrs))
    arr_polarity.fill(np.nan)

npy_subjectivity = npys / f'{NAME}_subjectivity.npy'
if npy_subjectivity.exists():
    arr_subjectivity = np.load(npy_subjectivity)
else:
    arr_subjectivity = np.empty(len(corrs))
    arr_subjectivity.fill(np.nan)


def sentiment(i_text):
    """Return the estimated school grade level required to understand the text.

    Parameters
    ----------
    i_text : (int, str)
        Pair containing index and string.

    Returns
    -------
    i_polarity_subjectivity : (int, float, float)
        Triplet of ith text polarity and subjectivity.
    """

    i, text = i_text
    try:
        polarity, subjectivity = nlp_util.blobify(text).sentiment
    except BaseException as exc:
        logger.exception(exc)
        raise
    else:
        logger.info(f'Sentiment Index: {NAME} @ {i}')
    return i, polarity, subjectivity


def sentiment_range(start=0, stop=None):
    """Save sentiment of all essays in the given range from start to stop.

    Parameters
    ----------
    start : int, optional
        Slice start index.

    stop : int, optional
        Slice stop index.
    """

    logger.info(f'Start Index: {NAME} @ {start}')

    with multiprocessing.Pool() as pool:
        i_polarity_subjectivity = pool.map(sentiment, enumerate(corrs[start:stop], start=start))

    idxs, polarities, subjectivities = map(list, zip(*i_polarity_subjectivity))

    arr_polarity[idxs] = polarities
    np.save(npy_polarity, arr_polarity)

    arr_subjectivity[idxs] = subjectivities
    np.save(npy_subjectivity, arr_subjectivity)

    logger.info(f'Stop Index: {NAME} @ {stop}')


if __name__ == '__main__':
    sentiment_range(0, None)
