import logging
import multiprocessing
import pathlib

import numpy as np
import pandas as pd
from textblob import TextBlob


#: User Input Prompts
NAME = input('Input data name: ')
START = int(input('Input start index: '))
try:
    STOP = int(input('Input stop index: '))
except Exception:
    STOP = None


#: Directory Paths
data = pathlib.Path.cwd() / 'data'
pkls = data / 'pkls'
npys = data / 'npys'
logs = data / 'logs'


#: Launch Logger
log_file = logs / f'{NAME}_corrections.log'
log_file.touch()
fmt = '{name} - {asctime} - {levelname} - Message: {message}'
logging.basicConfig(filename=log_file, 
                    level=logging.INFO, 
                    style='{', 
                    format=fmt)
logger = logging.getLogger(__name__)


#: Load Essay Array
pkl = pkls / f'{NAME}.pkl'
df = pd.read_pickle(pkl)        
essays = df['essay'][START:STOP]


#: Load Correction Array
npy = npys / f'{NAME}_corrections.npy'
if npy.exists():
    arr = np.load(npy)
else:
    arr = np.empty(len(df['essay']), dtype=object)


def correct(i_essay):
    """Correct the ith essay spelling.
    
    Parameters
    ----------
    i_essay : (int, str)
        Pair containing index and string.
    
    Returns
    -------
    i_corr : (i, str)
        Pair of ith string of corrections. 
    """
    
    i, essay = i_essay
    try:
        corr = str(TextBlob(essay).correct())
    except BaseException as exc:
        logger.exception(exc)
        raise
    else:
        logger.info(f'Corrected Index: {NAME} @ {i}')
    return i, corr


def correct_range(start=0, stop=None):
    """Correct all essays in the given range from start to stop.
    
    Parameters
    ----------
    start : int, optional
        Slice start index.
    
    stop : int, optional
        Slice stop index.
    """
    
    logger.info(f'Start Index: {NAME} @ {start}')
    
    with multiprocessing.Pool() as pool:
        i_corrs = pool.map(correct, enumerate(essays, start=start))
    
    idxs, corrs = map(list, zip(*i_corrs))
    arr[idxs] = corrs
    np.save(npy, arr)
    logger.info(f'Stop Index: {NAME} @ {stop}')

    
if __name__ == '__main__':
    correct_range(START, STOP)
