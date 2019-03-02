import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import nlp_util


def _clean(text):
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


def error_ratio(original, corrected):
    """Return the error ratio of original text in comparison to corrected text.

    Parameters
    ----------
    original : str, TextBlob
        Original essay.
    corrected : str, TextBlob
        Corrected essay.

    Returns
    -------
    error_ratio_df : DataFrame shape=(1, 1)
        Ratio of errors to the number of words in original.
    """

    original = npl_util.blobify(original)
    corrected = npl_util.blobify(corrected)
    error_ratio = sum(not word in corrected.tokenize() for word in original.tokenize()) / len(original)
    error_ratio_dict = {'error_ratio': error_ratio}
    error_ratio_df = pd.DataFrame([error_ratio_dict])
    return error_ratio_df


def similarity(text1, text2):
    """Return dataframe of the cosine similarity between both texts.

    Parameters
    ----------
    text1 : str
        One of the texts to be compared.
    text2 : str
        Other text to be compared.

    Returns
    -------
    similarity_df : DataFrame shape=(1, 2)
        DataFrame with columns: count, tfidf
    """

    clean1 = _clean(text1)
    clean2 = _clean(text2)
    count_meas = nlp_util.prompt_similarity(clean1, clean2, vectorizer=CountVectorizer)
    tfidt_meas = nlp_util.prompt_similarity(clean1, clean2, vectorizer=TfidfVectorizer)
    similarity_dict = {'count': count_meas, 'tfidf': tfidt_meas}
    similarity_df = pd.DataFrame([similarity_dict])
    return similarity_df


def pos(text):
    """Return dataframe of the ratio of each POS to length of the text.

    Parameters
    ----------
    text : str
        Text to analyize.

    Returns
    -------
    pos_df : DataFrame shape=(1, 36)
        DataFrame with columns for each POS defined by NLTK library.
    """

    pos_counter = nlp_util.parts_of_speech(text)
    pos_df = pd.DataFrame([pos_counter])
    pos_df = pos_df.div(pos_df.sum(axis=1), axis=0)
    return pos_df


def sentiment(text):
    """Return dataframe of the sentiment of the text.

    Parameters
    ----------
    text : str
        Text to analyize.

    Returns
    -------
    sentiment_df : DataFrame shape=(1, 2)
        DataFrame with columns: polarity, subjectivity.
    """

    sentiment_dict = nlp_util.blobify(text).sentiment._asdict()
    sentiment_df = pd.DataFrame([sentiment_dict])
    return sentiment_df


def difficulty(text):
    """Return dataframe of the various difficulty levels of the text.

    Parameters
    ----------
    text : str
        Text to analyize.

    Returns
    -------
    difficulty_df : DataFrame shape=(1, 8)
        DataFrame with columns for each difficulty level from TextStat library.
    """
    difficulty_levels = [func(text) for func in nlp_util.DIFFICULTY_FUNCS]
    difficulty_level_dict = nlp_util.DifficultyLevel(*difficulty_levels)._asdict()
    difficulty_df = pd.DataFrame([difficulty_level_dict])
    return difficulty_df


def all_features(essay, prompt, grade_level):
    """Return dataframe of all the features in this module.

    Parameters
    ----------
    essay : str
        Essay to analyize.
    prompt : str
        Prompt to be used to compare similarity with essay.

    Returns
    -------
    features_df : DataFrame shape=(1, ...)
        DataFrame with columns including grade level, similarity, POS,
        difficulty level, sentiment.
    """

    correction = nlp_util.correct(essay)
    grade_df = pd.DataFrame([grade_level], columns=['grade_level'])
    similarity_df = similarity(correction, prompt)
    pos_df = pos(correction)
    difficulty_df = difficulty(correction)
    sentiment_df = sentiment(correction)
    dfs = [grade_df, similarity_df, difficulty_df, sentiment_df, pos_df]
    features_df = pd.concat(dfs, axis=1)
    return features_df
