from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.features import LinguisticExtractor
from src.preprocess import preprocess_for_tfidf


def build_pipeline(model, combined=True, nb=False):
    tfidf = TfidfVectorizer(
        preprocessor=preprocess_for_tfidf,
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
        min_df=3,
        max_df=0.90
    )

    if combined:
        ling_pipeline = Pipeline([
            ('extract', LinguisticExtractor()),
            ('scale', MinMaxScaler() if nb else StandardScaler())
        ])
        features = FeatureUnion([
            ('tfidf', tfidf),
            ('linguistic', ling_pipeline)
        ])
    else:
        features = tfidf

    return Pipeline([
        ('features', features),
        ('classifier', model)
    ])