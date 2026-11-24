import time
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score


def get_cv(n_splits=5, random_state=42):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def evaluate_model(name, model, X_train, y_train, X_test, y_test, cv, results_store):
    cv_start = time.time()
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1
    )
    cv_time = time.time() - cv_start

    fit_start = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - fit_start

    y_pred = model.predict(X_test)

    res = {
        'name': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        'cv_time_s': cv_time,
        'train_time_s': fit_time,
        'y_pred': y_pred,
        'model': model
    }
    results_store[name] = res

    print(f"\n{name} | Acc: {res['accuracy']:.4f} | Test F1: {res['f1_weighted']:.4f} | "
          f"CV F1: {res['cv_f1_mean']:.4f} +/- {res['cv_f1_std']:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    return res


def tune_model(name, pipeline, param_dist, n_iter, X_train, y_train, X_test, y_test, cv, results_store):
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    search.fit(X_train, y_train)
    print(f"Best {name} params: {search.best_params_} | Best CV F1: {search.best_score_:.4f}")
    return evaluate_model(name, search.best_estimator_, X_train, y_train, X_test, y_test, cv, results_store)


def build_summary_df(all_results):
    rows = [
        {
            'Model': k,
            'Accuracy': v['accuracy'],
            'F1 (weighted)': v['f1_weighted'],
            'CV F1 mean': v['cv_f1_mean'],
            'CV F1 std': v['cv_f1_std'],
            'Train time (s)': v['train_time_s'],
            'CV time (s)': v['cv_time_s']
        }
        for k, v in all_results.items()
    ]
    return pd.DataFrame(rows).set_index('Model').sort_values('F1 (weighted)', ascending=False)


def build_ablation_df(all_results):
    pairs = [
        ('Logistic Regression', 'LR (TF-IDF)', 'LR (Combined)'),
        ('Naive Bayes', 'CNB (TF-IDF)', 'CNB (Combined)'),
        ('Linear SVM', 'SVM (TF-IDF)', 'SVM (Combined)'),
        ('Random Forest', 'RF (TF-IDF)', 'RF (Combined)')
    ]
    rows = []
    for label, k_tfidf, k_comb in pairs:
        f1_t = all_results[k_tfidf]['f1_weighted']
        f1_c = all_results[k_comb]['f1_weighted']
        delta = f1_c - f1_t
        rows.append({
            'Model': label,
            'TF-IDF only F1': f1_t,
            'TF-IDF + Linguistic F1': f1_c,
            'Delta': delta,
            'Improvement': 'Yes' if delta > 0 else 'No'
        })
    return pd.DataFrame(rows).set_index('Model')