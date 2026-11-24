import warnings
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.features import LinguisticExtractor
from src.lstm_model import train_and_eval_lstm
from src.pipelines import build_pipeline
from src.preprocess import ensure_nltk_resources
from src.train_eval import build_ablation_df, build_summary_df, evaluate_model, get_cv, tune_model

warnings.filterwarnings('ignore')


def load_data(fake_path='data/Fake.csv', real_path='data/True.csv', n_per_class=5000):
    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)
    fake['label'], real['label'] = 0, 1

    df = pd.concat([
        fake.sample(min(n_per_class, len(fake)), random_state=42),
        real.sample(min(n_per_class, len(real)), random_state=42)
    ], ignore_index=True)

    df = df[['text', 'label']].dropna(subset=['text'])
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'] != ''].drop_duplicates(subset=['text']).reset_index(drop=True)
    return df


def main(run_lstm=False):
    ensure_nltk_resources()
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].values, df['label'].values, test_size=0.2, random_state=42, stratify=df['label'].values
    )
    cv = get_cv()
    results = {}

    base_specs = [
        ('LR', lambda: LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42), False),
        ('CNB', lambda: ComplementNB(alpha=0.1), True),
        ('SVM', lambda: LinearSVC(C=0.5, max_iter=2000, class_weight='balanced', random_state=42), False),
        ('RF', lambda: RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=1), False),
    ]

    for tag, ctor, is_nb in base_specs:
        evaluate_model(f'{tag} (TF-IDF)', build_pipeline(ctor(), combined=False), X_train, y_train, X_test, y_test, cv, results)
        evaluate_model(f'{tag} (Combined)', build_pipeline(ctor(), combined=True, nb=is_nb), X_train, y_train, X_test, y_test, cv, results)

    tune_model(
        'LR Tuned (TF-IDF)',
        build_pipeline(LogisticRegression(max_iter=1500, class_weight='balanced', random_state=42), combined=False),
        {'classifier__C': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0], 'classifier__penalty': ['l1', 'l2']},
        10, X_train, y_train, X_test, y_test, cv, results
    )
    tune_model(
        'SVM Tuned (Combined)',
        build_pipeline(LinearSVC(max_iter=2000, class_weight='balanced', random_state=42), combined=True),
        {'classifier__C': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]},
        7, X_train, y_train, X_test, y_test, cv, results
    )
    evaluate_model(
        'RF (Linguistic Only)',
        Pipeline([
            ('extract', LinguisticExtractor()),
            ('scale', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=1))
        ]),
        X_train, y_train, X_test, y_test, cv, results
    )

    if run_lstm:
        train_and_eval_lstm(X_train, y_train, X_test, y_test, results)

    print("\nSUMMARY:\n", build_summary_df(results).round(4).to_string())
    print("\nABLATION:\n", build_ablation_df(results).round(4).to_string())


if __name__ == '__main__':
    main(run_lstm=False)