import pathlib
import pickle

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import textblob

import nlp_util


model_pkl = pathlib.Path.cwd() / 'data' / 'models' / 'rand_forest.pkl'
with open(model_pkl, 'rb') as fp:
    model = pickle.load(fp)


def correct(essay):
    corr = str(textblob.TextBlob(essay).correct()).strip()
    return corr


def similarity(corrected, prompt):
    clean_prompt = nlp_util.clean(prompt)
    clean_essay = nlp_util.clean(corrected)
    count_meas = nlp_util.prompt_similarity(clean_prompt, clean_essay,
                                            vectorizer=CountVectorizer)
    tfidt_meas = nlp_util.prompt_similarity(clean_prompt, clean_essay,
                                            vectorizer=TfidfVectorizer)
    dct = {'prompt_count': count_meas, 'prompt_tfidf': tfidt_meas}
    df = pd.DataFrame([dct])
    return df


def pos(corrected):
    pos = nlp_util.parts_of_speech(corrected)
    df = pd.DataFrame([pos])
    df = df.div(df.sum(axis=1), axis=0)
    return df


def difficulty(corrected):
    diff_level = [f(corrected) for f in nlp_util.diff_funcs]
    diff_level = nlp_util.DifficultyLevel(*diff_level)._asdict()
    df = pd.DataFrame([diff_level])
    return df


def sentiment(corrected):
    sent = nlp_util.blobify(corrected).sentiment._asdict()
    df = pd.DataFrame([sent])
    return df


def all_features(essay, prompt, grade_level):
    corr = correct(essay)
    grade_df = pd.DataFrame([grade_level], columns=['grade_level'])
    sim_df = similarity(corr, prompt)
    pos_df = pos(corr)
    diff_df = difficulty(corr)
    sent_df = sentiment(corr)
    dfs = [grade_df, sim_df, diff_df, sent_df, pos_df]
    features = pd.concat(dfs, axis=1)
    return features


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
            features = api.all_features(essay, prompt, grade_level)
            score = api.score_essay(features)
        except Exception:
            pass

    return essay, prompt, grade_level, score