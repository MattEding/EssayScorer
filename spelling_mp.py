import itertools
import logging
import multiprocessing
import pathlib

import numpy as np
import pandas as pd
from textblob import TextBlob


name = input('Input data name: ')
start = int(input('Input start index: '))
stop = input('Input stop index: ')
if not stop:
    stop = None
else:
    stop = int(stop)


data = pathlib.Path.cwd() / 'data'
pkls = data / 'pkls'
npys = data / 'npys'


#: Launch Logger
log_file = data / 'logs' / f'{name}_corrections.log'
log_file.touch()
fmt = '{name} - {asctime} - {levelname} - Message: {message}'
logging.basicConfig(filename=log_file, level=logging.INFO, style='{', format=fmt)
logger = logging.getLogger(__name__)

#: Load essay array
pkl = pkls / f'{name}.pkl'
df = pd.read_pickle(pkl)        
essays = df['essay'][start:stop]

#: Load correction array
npy = npys / f'{name}_corrections.npy'
if npy.exists():
    arr = np.load(npy)
else:
    arr = np.empty(len(df['essay']), dtype=object)


cycle = itertools.cycle(range(10))
next(cycle) # don't np.save immediately


def correct(i_essay, *, _cycle=cycle):
    i, essay = i_essay
    try:
        corr = str(TextBlob(essay).correct())
    except BaseException as exc:
        logger.exception(exc)
        raise
    else:
        logger.info(f'Corrected Index: {name} @ {i}')
    return i, corr


def correct_range(name, start=0, stop=None):
    logger.info(f'Start Index: {name} @ {start}')
    
    with multiprocessing.Pool() as pool:
        i_corrs = pool.map(correct, enumerate(essays, start=start))
    
    idxs, corrs = map(list, zip(*i_corrs))
    arr[idxs] = corrs
    np.save(npy, arr)
    logger.info(f'Stop Index: {name} @ {stop}')

    
if __name__ == '__main__':
    correct_range(name, start)
