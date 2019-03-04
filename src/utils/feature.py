import collections

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import src.utils.nlp


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

    lower_proper = src.utils.nlp.lower_with_proper(text)
    lemmas = src.utils.nlp.lemmatize(lower_proper)
    cleaned = src.utils.nlp.clean_stopwords_punctuation(lemmas)
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
    difficulty_levels = [func(text) for func in src.utils.nlp.DIFFICULTY_FUNCS]
    difficulty_level_dict = src.utils.nlp.DifficultyLevel(*difficulty_levels)._asdict()
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

    original = src.utils.nlp.blobify(original)
    corrected = src.utils.nlp.blobify(corrected)
    error_ratio = sum(not word in corrected.tokenize() for word in original.tokenize()) / len(original)
    error_ratio_dict = {'error_ratio': error_ratio}
    return error_ratio_dict


def pos(text):
    """Return dataframe of the ratio of each POS to length of the text.

    Parameters
    ----------
    text : str, TextBlob
        Text to analyize.

    Returns
    -------
    pos_dict : dict
        Mapping with columns for each POS defined by NLTK library.
    """

    pos_counter = src.utils.nlp.parts_of_speech(text)
    total_count = sum(pos_counter.values())
    pos_dict = {pos: count / total_count for pos, count in pos_counter.items()}
    return pos_dict


def sentiment(text):
    """Return dataframe of the sentiment of the text.

    Parameters
    ----------
    text : str, TextBlob
        Text to analyize.

    Returns
    -------
    sentiment_dict : dict
        Mapping with columns: polarity, subjectivity.
    """

    sentiment_dict = src.utils.nlp.blobify(text).sentiment._asdict()
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
    count_meas = src.utils.nlp.prompt_similarity(clean1, clean2, vectorizer=CountVectorizer)
    tfidt_meas = src.utils.nlp.prompt_similarity(clean1, clean2, vectorizer=TfidfVectorizer)
    similarity_dict = {'count': count_meas, 'tfidf': tfidt_meas}
    return similarity_dict


def words(text):
    """Return dataframe of the cosine similarity between both texts.

    Parameters
    ----------
    text : str
        One of the texts to be compared.

    Returns
    -------
    similarity_dict : dict
        Mapping with columns: count, tfidf
    """
    clean = src.utils.nlp.blobify(_clean(text))
    sentence_count = len(clean.sentences)
    words = clean.tokenize()
    word_count = len(words)
    avg_len = np.mean([len(word) for word in words])
    words_dict = {'sentence_count': sentence_count, 'word_count': word_count,
                 'avg_len': avg_len}
    return words_dict


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

    correction = src.utils.nlp.correct(essay)

    difficulty_level_dict = difficulty_level(correction)
    error_ratio_dict = error_ratio(essay, correction)
    grade_level_dict = {'grade_level': grade_level}
    pos_dict = pos(correction)
    sentiment_dict = sentiment(correction)
    similarity_dict = similarity(correction, prompt)
    words_dict = words(correction)

    chain_map = collections.ChainMap(difficulty_level_dict, error_ratio_dict,
                                     grade_level_dict, pos_dict, sentiment_dict,
                                     similarity_dict, words_dict)
    return chain_map
