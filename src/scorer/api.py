import pathlib
import pickle

import numpy as np
import pandas as pd

import utils.feature

# TODO: need to redo model with error ratio -- had accidentally not used prior
model_pkl = pathlib.Path.cwd() / 'data' / 'models' / 'rand_forest.pkl'
with open(model_pkl, 'rb') as fp:
    model = pickle.load(fp)


def score_essay(features):
    pred = model.predict(features)
    score = np.asscalar(pred.round(1))
    if score < 0:
        score = 0.0
    elif score > 100:
        score = 100.0
    return f'{score}%'


def process_args(args):
    essay = args.get('essay', '')
    prompt = args.get('prompt', '')
    grade_level = args.get('grade_level')
    if grade_level is not None:
        grade_level = int(grade_level)

    score = ''
    if all(arg for arg in [essay, prompt, grade_level]):
        try:
            features_chain_map = utils.feature.all_features(essay, prompt, grade_level)
            features_df = pd.DataFrame([features_chain_map])
            score = score_essay(features_df)
        except Exception:
            pass

    return essay, prompt, grade_level, score
