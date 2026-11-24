import numpy as np
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

LING_FEATURES = [
    'superlative_ratio',
    'proper_noun_ratio',
    'first_person_ratio',
    'pron_noun_ratio',
    'modal_ratio',
    'exclamation_den',
    'question_den',
    'avg_sent_len',
    'type_token_ratio',
    'uppercase_ratio'
]


def get_pos_tags(text):
    tokens = word_tokenize(str(text)[:3000])
    return pos_tag(tokens)


def superlative_ratio(tags):
    total = len(tags)
    return sum(1 for _, tag in tags if tag in ('JJS', 'RBS')) / total if total else 0.0


def proper_noun_ratio(tags):
    total = len(tags)
    return sum(1 for _, tag in tags if tag in ('NNP', 'NNPS')) / total if total else 0.0


def first_person_ratio(tags):
    total = len(tags)
    if total == 0:
        return 0.0
    first_person = {
        'i', 'me', 'my', 'myself',
        'we', 'us', 'our', 'ours', 'ourselves'
    }
    return sum(
        1 for word, tag in tags
        if tag.startswith('PRP') and word.lower() in first_person
    ) / total


def pronoun_to_noun_ratio(tags):
    pron = sum(1 for _, tag in tags if tag.startswith('PRP'))
    nouns = sum(1 for _, tag in tags if tag.startswith('NN'))
    denom = pron + nouns
    return pron / denom if denom else 0.0


def modal_verb_ratio(tags):
    total = len(tags)
    return sum(1 for _, tag in tags if tag == 'MD') / total if total else 0.0


def exclamation_density(text):
    sentences = sent_tokenize(str(text))
    return str(text).count('!') / max(len(sentences), 1)


def question_density(text):
    sentences = sent_tokenize(str(text))
    return str(text).count('?') / max(len(sentences), 1)


def avg_sentence_length(text):
    sentences = sent_tokenize(str(text))
    if not sentences:
        return 0.0
    return np.mean([len(word_tokenize(s)) for s in sentences])


def type_token_ratio(text):
    tokens = [t for t in word_tokenize(str(text).lower()) if t.isalpha()]
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def uppercase_word_ratio(text):
    words = str(text).split()
    if not words:
        return 0.0
    return sum(1 for w in words if w.isupper() and len(w) > 1) / len(words)


def extract_linguistic_features(text):
    tags = get_pos_tags(text)
    return [
        superlative_ratio(tags),
        proper_noun_ratio(tags),
        first_person_ratio(tags),
        pronoun_to_noun_ratio(tags),
        modal_verb_ratio(tags),
        exclamation_density(text),
        question_density(text),
        avg_sentence_length(text),
        type_token_ratio(text),
        uppercase_word_ratio(text)
    ]


class LinguisticExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(
            [extract_linguistic_features(text) for text in X],
            dtype=float
        )