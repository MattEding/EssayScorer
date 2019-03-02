import multiprocessing
import pathlib

import pandas as pd

import feature_util


pkls = pathlib.Path.cwd() / 'data' / 'pkls'

error_ratio_pkl = pkls / 'error_ratio_pkl'
if not error_ratio_pkl.exists():
    error_ratio_pkl.touch()
