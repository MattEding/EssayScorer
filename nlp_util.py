import collections
import string

import numpy as np

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from textblob import TextBlob


PUNCTUATION = string.punctuation
STOPWORDS = stopwords.words('english')

NOUN_MAP = {
    'common': frozenset('EMAIL MONEY NUM PERCENT TIME'.split()),
    'proper': frozenset('CAPS CITY DATE DR LOCATION MONTH PERSON ORGANIZATION STATE'.split()),
}


def blobify(text):
    """Coerce text to be a TextBlob.
    
    Parameters
    ----------
    text : str, TextBlob
        Text to be converted.
    
    Returns
    -------
    textblob : TextBlob
        TextBlob of the text.
        
    Raises
    ------
    TypeError
        If the text is not either an instance of str or TextBlob.
    """
    
    try:
        text = TextBlob(text)
    except TypeError:
        if not isinstance(text, TextBlob):
            raise
    return text


#: lemmatize prompt and essay before checking similarity
def prompt_similarity(prompt, essay, metric=cosine_similarity, vectorizer=TfidfVectorizer):
    """Return the metric measurement between the prompt and essay.
    
    Parameters
    ----------
    prompt : str
        Essay prompt.
    essay : str
        Essay written.
    metric : pairwise metric
        Metric to compare the two documents.
    vectorizer : class with VectorizerMixin
        Vectorizer to fit transform the documents.
        
    Returns
    -------
    measure : float
        Measurement between the vectorized documents.
    """
    
    vec = vectorizer()
    X = vec.fit_transform([corrected, original])
    measure = np.asscalar(metric(X[0], X[1]))
    return measure


def spelling_error_rate(original, corrected):
    """Return the spelling error rate of original text in comparison to corrected text.
    
    Parameters
    ----------
    original : str, TextBlob
        Original essay.
    corrected : str, TextBlob
        Corrected essay.
    
    Returns
    -------
    rate : float
        Ratio of spelling errors to the number of words in original.
    """
    
    original = blobify(original)
    corrected = blobify(corrected)
    rate = sum(not word in corrected.tokenize() for word in original.tokenize()) / len(original)
    return rate


def lower_with_proper(text):
    """Return text with all non-proper nouns lowercased regardless of punctuation.
    
    Parameters
    ----------
    text : str
        Text to lowercase.
    
    Returns
    -------
    lower_proper : str
        Lowercased text except for proper nouns.
    """
    
    proper_flags = NOUN_MAP['proper']
    lower_proper = []
    for word in map(str.lower, text.split()):
        #: flag for special words
        if word.startswith('@'):
            word = word[1:]
            if any(flag.lower() in word for flag in proper_flags):
                word = word.upper()
        lower_proper.append(word)
    return ' '.join(lower_proper)


# need to lower all words first
def lemmatize(text):
    """Lemmatize text.
    
    Parameters
    ----------
    text : str, TextBlob
        Text to lemmatize.
    
    Returns
    -------
    lemmas : WordList
        List of tokenized words that have been lemmatized.
    """
    
    text = blobify(text)
    lemmas = text.tokenize().lemmatize()
    return lemmas


def parts_of_speech(text):
    """Return counts of each POS.
    
    Parameters
    ----------
    text : str, TextBlob
        Text to process.
    
    Returns
    -------
    pos : Counter
        Counter of all part of speech occurances in text.
    """
    
    text = blobify(text)
    _, pos = zip(*text.pos_tags)
    return collections.Counter(pos)