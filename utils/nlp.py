import collections
import string

import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob, WordList
import textstat


DifficultyLevel = collections.namedtuple('DifficultyLevel', [func.__name__ for func in DIFFICULTY_FUNCS])

DIFFICULTY_FUNCS = (textstat.flesch_reading_ease, textstat.smog_index,
                    textstat.flesch_kincaid_grade, textstat.coleman_liau_index,
                    textstat.automated_readability_index, textstat.dale_chall_readability_score,
                    textstat.linsear_write_formula, textstat.gunning_fog)

NOUN_MAP = {
    'common': 'EMAIL MONEY NUM PERCENT TIME'.split(),
    'proper': 'CAPS CITY DATE DR LOCATION MONTH PERSON ORGANIZATION STATE'.split(),
}

POS = ('CC CD DT EX FW IN JJ JJR JJS LS MD NN NNP NNPS NNS PDT POS PRP PRP$ '
       'RB RBR RBS RP SYM TO UH VB VBD VBG VBN VBP VBZ WDT WP WP$ WRB').split()

_POS_COUNTER = collections.Counter()
for pos in POS:
    _POS_COUNTER[pos] = 0


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


def clean_stopwords_punctuation(wordlist, punctuation=string.punctuation,
                                stopwords=stopwords.words('english')):
    """Return cleaned text without stopwords or punctuation.

    Parameters
    ----------
    wordlist : WordList
        WordList of words to clean.
    punctuation : container, optional
        Set of punctuation characters to ignore.
    stopwords : container, optional
        Set of strings of words to ignore.

    Returns
    -------
    cleaned : str
        String of cleaned text joined.
    """

    cleaned = WordList(w for w in wordlist if w not in punctuation and w not in stopwords)
    cleaned = ' '.join(cleaned)
    return cleaned


def correct(text):
    """Return corrected text.

    Parameters
    ----------
    text : str, TextBlob
        Text to be corrected.

    Returns
    -------
    correction : str
        Corrected text string.
    """

    correction = str(blobify(text).correct()).strip()
    return correction


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
        #: '@' is a flag for special words
        if word.startswith('@'):
            word = word[1:]
            if any(flag.lower() in word for flag in proper_flags):
                word = word.upper()
        lower_proper.append(word)
    lower_proper = ' '.join(lower_proper)
    return lower_proper


def parts_of_speech(text):
    """Return counts of each POS.

    Parameters
    ----------
    text : str, TextBlob
        Text to process.

    Returns
    -------
    pos_counter : Counter
        Counter of all part of speech occurances in text.
    """

    text = blobify(text)
    _, pos = zip(*text.pos_tags)
    pos_counter = _POS_COUNTER.copy()
    pos_counter.update(pos)
    return pos_counter


def prompt_similarity(prompt, essay, metric=cosine_similarity, vectorizer=TfidfVectorizer):
    """Return the metric measurement between the prompt and essay.

    Parameters
    ----------
    prompt : str
        Essay prompt.
    essay : str
        Essay written.
    metric : pairwise metric, optional
        Metric to compare the two documents.
    vectorizer : VectorizerMixin, optional
        Vectorizer to fit transform the documents.

    Returns
    -------
    measure : float
        Measurement between the vectorized documents.
    """

    vec = vectorizer()
    X = vec.fit_transform([essay, prompt])
    measure = np.asscalar(metric(X[0], X[1]))
    return measure
