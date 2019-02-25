import logging
import multiprocessing as mp
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





def correct_range(name, start=0, stop=None):
    pkl = pkls / f'{name}.pkl'
    npy = npys / f'{name}_corrections.npy'

    
    if npy.exists():
        arr = np.load(npy)
    else:
        arr = np.empty(len(essays), dtype=object)

    logger.info(f'Start Index: {name} @ {start}')
    
    df = pd.read_pickle(pkl)        
    essays = df['essay'][start:stop]

    
    def correct(i_essay):
        i, essay = i_essay
        corr = str(TextBlob(essay).correct())
        try:
            arr[i] = corr
            logger.info(f'Corrected Index: {name} @ {i}')
        except BaseException as exc:
            logger.exception(exc)
            raise
        finally:
            np.save(npy, arr)
            logger.info(f'Saved Corrections: {name} @ {i-1}...(mp)')
        

    with mp.Pool() as pool:
        corrs = pool.map(correct, enumerate(essays, start=start))


#    corrs = (str(TextBlob(essay).correct()) for essay in essays)
#    try:
#        for i, corr in enumerate(corrs, start=start):
#            arr[i] = corr
#            logger.info(f'Corrected Index: {name} @ {i}')
#            if (start + i) % 50 == 0:
#                np.save(npy, arr)
#                logger.info(f'Saved Corrections: {name} @ {i}')
#    except BaseException as exc:
#        logger.exception(exc)
#        raise
#    finally:
#        np.save(npy, arr)
#        logger.info(f'Saved Corrections: {name} @ {i-1}')
    
    logger.info(f'Stop Index: {name} @ {stop}')
    

if __name__ == '__main__':
    correct_range(name, start)
