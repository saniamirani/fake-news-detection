import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score


def train_and_eval_lstm(X_train, y_train, X_test, y_test, results_store):
    try:
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, SpatialDropout1D
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.preprocessing.text import Tokenizer
    except ImportError:
        print("TensorFlow not installed; skipping LSTM.")
        return None

    max_words, max_len = 30000, 300
    tok = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tok.fit_on_texts(X_train)

    X_tr = pad_sequences(tok.texts_to_sequences(X_train), maxlen=max_len, padding='post', truncating='post')
    X_te = pad_sequences(tok.texts_to_sequences(X_test), maxlen=max_len, padding='post', truncating='post')

    model = Sequential([
        Embedding(max_words, 64, input_length=max_len),
        SpatialDropout1D(0.3),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
        LSTM(32, dropout=0.2),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        X_tr, y_train, epochs=15, batch_size=64, validation_split=0.1, callbacks=[early_stop], verbose=1
    )

    y_pred = (model.predict(X_te).flatten() >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"\nLSTM | Acc: {acc:.4f} | Test F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

    results_store['LSTM'] = {
        'name': 'LSTM',
        'accuracy': acc,
        'f1_weighted': f1,
        'cv_f1_mean': np.nan,
        'cv_f1_std': np.nan,
        'train_time_s': np.nan,
        'cv_time_s': np.nan,
        'y_pred': y_pred,
        'model': model,
        'history': history.history
    }
    return results_store['LSTM']