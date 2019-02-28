import multiprocessing
import pathlib
import reprlib

import numpy as np
import pandas as pd
import textblob

import nlp_util
import utils


#: User Input Prompts
NAME = input('Input data name: ').strip().lower()

reprlib.aRepr.maxstring = 70
logger = utils.get_logger(f'{NAME}_sentiment', __name__)

#: Directory Paths
data = pathlib.Path.cwd() / 'data'
npys = data / 'npys'
pkls = data / 'pkls'

#: Load Corrections Array
npy_corrs = npys / f'{NAME}_corrections.npy'
corrs = np.load(npy_corrs)


def sentiment(text):
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
    text = text.strip()
    try:
        sentiment = nlp_util.blobify(text).sentiment
    except BaseException as exc:
        logger.exception(exc)
        raise
    else:
        logger.info(f'{NAME} : {reprlib.repr(text)}')
    return sentiment._asdict()


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
        sentiments = pool.map(sentiment, corrs[start:stop])

    df_sentiment = pd.DataFrame(sentiments)
    df_sentiment.to_pickle(pkls / f'{NAME}_sentiment.pkl')

    logger.info(f'Stop Index: {NAME} @ {stop}')


if __name__ == '__main__':
    sentiment_range(0, None)
