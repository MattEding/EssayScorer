import collections

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import utils.nlp


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

    lower_proper = utils.nlp.lower_with_proper(text)
    lemmas = utils.nlp.lemmatize(lower_proper)
    cleaned = utils.nlp.clean_stopwords_punctuation(lemmas)
    return cleaned


def difficulty_level(text):
    """Return dataframe of the various difficulty levels of the text.

    Parameters
    ----------
    text : str
        Text to analyize.

    Returns
    -------
    difficulty_dict : dict
        Mapping with columns for each difficulty level from TextStat library.
    """
    difficulty_levels = [func(text) for func in utils.nlp.DIFFICULTY_FUNCS]
    difficulty_level_dict = utils.nlp.DifficultyLevel(*difficulty_levels)._asdict()
    return difficulty_level_dict


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
    error_ratio_dict : dict
        Mappingf errors to the number of words in original.
    """

    original = utils.nlp.blobify(original)
    corrected = utils.nlp.blobify(corrected)
    error_ratio = sum(not word in corrected.tokenize() for word in original.tokenize()) / len(original)
    error_ratio_dict = {'error_ratio': error_ratio}
    return error_ratio_dict


def pos(text):
    """Return dataframe of the ratio of each POS to length of the text.

    Parameters
    ----------
    text : str
        Text to analyize.

    Returns
    -------
    pos_dict : dict
        Mapping with columns for each POS defined by NLTK library.
    """

    pos_counter = utils.nlp.parts_of_speech(text)
    total_count = sum(pos_counter.values())
    pos_dict = {pos: count / total_count for pos, count in pos_counter.items()}
    return pos_dict


def sentiment(text):
    """Return dataframe of the sentiment of the text.

    Parameters
    ----------
    text : str
        Text to analyize.

    Returns
    -------
    sentiment_dict : dict
        Mapping with columns: polarity, subjectivity.
    """

    sentiment_dict = utils.nlp.blobify(text).sentiment._asdict()
    return sentiment_dict


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
    similarity_dict : dict
        Mapping with columns: count, tfidf
    """

    clean1 = _clean(text1)
    clean2 = _clean(text2)
    count_meas = utils.nlp.prompt_similarity(clean1, clean2, vectorizer=CountVectorizer)
    tfidt_meas = utils.nlp.prompt_similarity(clean1, clean2, vectorizer=TfidfVectorizer)
    similarity_dict = {'count': count_meas, 'tfidf': tfidt_meas}
    return similarity_dict


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
    features_chain_map : ChainMap
        DataFrame with columns including grade level, similarity, POS,
        difficulty level, sentiment.
    """

    correction = utils.nlp.correct(essay)
    grade_level_dict = {'grade_level': grade_level}
    similarity_dict = similarity(correction, prompt)
    error_ratio_dict = error_ratio(essay, correction)
    pos_dict = pos(correction)
    difficulty_level_dict = difficulty_level(correction)
    sentiment_dict = sentiment(correction)
    chain_map = collections.ChainMap(grade_level_dict, similarity_dict, error_ratio_dict,
                                     difficulty_level_dict, sentiment_dict, pos_dict)
    return chain_map
