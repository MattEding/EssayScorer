import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from textblob import TextBlob


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


def spelling_error_rate(original, corrected, *, textblob=True):
    """Return the spelling error rate of original text in comparison to corrected text.
    
    Parameters
    ----------
    original : str, TextBlob
        Original essay.
    corrected : str, TextBlob
        Corrected essay.
    textblob : bool
        Flag for whether or not to cast strings as TextBlobs.
    
    Returns
    -------
    rate : float
        Ratio of spelling errors to the number of words in original.
    """
    
    if textblob:
        original = TextBlob(original)
        corrected = TextBlob(corrected)
        
    rate = sum(not word in corrected.tokenize() for word in original.tokenize()) / len(original)
    return rate


def lemmatize(string, *, textblob=True):
    """Lemmatize a string.
    
    Parameters
    ----------
    string : str, TextBlob
        String to lemmatize.
    textblob : bool
        Flag for whether or not to cast strings as TextBlobs.
    
    Returns
    -------
    lemmas : WordList
        List of tokenized words that have been lemmatized.
    """
    
    if textblob:
        corrected = TextBlob(corrected)
    lemmas = corrected.tokenize().lemmatize()
    return lemmas