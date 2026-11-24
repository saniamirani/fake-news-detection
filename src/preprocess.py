import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

NEGATION_KEEP = {
    'no', 'not', 'never', 'neither',
    'nor', 'nobody', 'nothing', 'nowhere'
}


def ensure_nltk_resources():
    for resource in [
        'punkt',
        'punkt_tab',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng'
    ]:
        nltk.download(resource, quiet=True)


def get_stop_words():
    stop_words = set(stopwords.words('english'))
    return stop_words - NEGATION_KEEP


def preprocess_for_tfidf(text, stop_words=None, lemmatizer=None):
    if stop_words is None:
        stop_words = get_stop_words()
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()

    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and (token not in stop_words or token in NEGATION_KEEP)
    ]
    return ' '.join(tokens)