import itertools
import logging
import multiprocessing
import pathlib

import numpy as np
import pandas as pd
from textblob import TextBlob


name = input('Input data name: ')
start = int(input('Input start index: '))


data = pathlib.Path.cwd() / 'data'
pkls = data / 'pkls'
npys = data / 'npys'


log_file = data / 'logs' / f'{name}_corrections.log'
log_file.touch()
fmt = '{name} - {asctime} - {levelname} - Message: {message}'
logging.basicConfig(filename=log_file, level=logging.INFO, style='{', format=fmt)
logger = logging.getLogger(__name__)


npy = npys / f'{name}_corrections.npy'
if npy.exists():
    arr = np.load(npy)
else:
    arr = np.empty(len(essays), dtype=object)


cycle = itertools.cycle(range(50))
next(cycle) # don't np.save immediately


def correct(i_essay, *, _cycle=cycle):
    i, essay = i_essay
    corr = str(TextBlob(essay).correct())
    try:
        arr[i] = corr
        logger.info(f'Corrected Index: {name} @ {i}')
    except BaseException as exc:
        logger.exception(exc)
        raise
    finally:
        if not next(_cycle):
            np.save(npy, arr)
            logger.info(f'Saved Corrections: {name} @ {i-1}...(mp)')


def correct_range(name, start=0, stop=None):
    pkl = pkls / f'{name}.pkl'

    logger.info(f'Start Index: {name} @ {start}')
    
    df = pd.read_pickle(pkl)        
    essays = df['essay'][start:stop]

    with multiprocessing.Pool() as pool:
        corrs = pool.map(correct, enumerate(essays, start=start))
    
    logger.info(f'Stop Index: {name} @ {stop}')
    

if __name__ == '__main__':
    correct_range(name, start)
