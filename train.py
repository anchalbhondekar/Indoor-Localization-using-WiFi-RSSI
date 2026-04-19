import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    LSTM, Bidirectional, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


print("Loading data...")
train_df = pd.read_csv('trainingData.csv')
val_df   = pd.read_csv('validationData.csv')
df = pd.concat([train_df, val_df], ignore_index=True)

wap_cols = [c for c in df.columns if c.startswith('WAP')]
df[wap_cols] = df[wap_cols].replace(100, -110)

X = df[wap_cols].values
y = df['FLOOR'].values
n_classes = len(np.unique(y))

print(f"Samples: {len(df)}  |  Features: {len(wap_cols)}  |  Floors: {np.unique(y)}")

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15/0.85, stratify=y_temp, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]}  |  Val: {X_val.shape[0]}  |  Test: {X_test.shape[0]}")


def get_metrics(name, pred, train_time, color):
    return {
        'model':      name,
        'accuracy':   round(accuracy_score(y_test, pred) * 100, 2),
        'f1':         round(f1_score(y_test, pred, average='macro') * 100, 2),
        'precision':  round(precision_score(y_test, pred, average='macro') * 100, 2),
        'recall':     round(recall_score(y_test, pred, average='macro') * 100, 2),
        'train_time': round(train_time, 1),
        'color':      color
    }


results = []


print("\nTraining Random Forest...")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_metrics = get_metrics('Random Forest', rf.predict(X_test), time.time() - t0, '#2e7d32')
results.append(rf_metrics)
print(f"Accuracy: {rf_metrics['accuracy']}%  |  F1: {rf_metrics['f1']}%")


print("\nTraining SVM...")
t0 = time.time()
svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm.fit(X_train, y_train)
svm_metrics = get_metrics('SVM', svm.predict(X_test), time.time() - t0, '#1565c0')
results.append(svm_metrics)
print(f"Accuracy: {svm_metrics['accuracy']}%  |  F1: {svm_metrics['f1']}%")


print("\nTraining CNN...")
X_tr_cnn = X_train.reshape(-1, X_train.shape[1], 1)
X_va_cnn = X_val.reshape(-1, X_val.shape[1], 1)
X_te_cnn = X_test.reshape(-1, X_test.shape[1], 1)
y_tr_cat = to_categorical(y_train, n_classes)
y_va_cat = to_categorical(y_val, n_classes)

cnn = Sequential([
    Conv1D(64,  5, activation='relu', padding='same', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(), MaxPooling1D(2), Dropout(0.3),
    Conv1D(128, 5, activation='relu', padding='same'),
    BatchNormalization(), MaxPooling1D(2), Dropout(0.3),
    Conv1D(256, 3, activation='relu', padding='same'),
    GlobalAveragePooling1D(), Dropout(0.4),
    Dense(128, activation='relu'), Dropout(0.3),
    Dense(n_classes, activation='softmax')
])
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

t0 = time.time()
cnn.fit(X_tr_cnn, y_tr_cat, validation_data=(X_va_cnn, y_va_cat),
        epochs=50, batch_size=64,
        callbacks=[EarlyStopping(patience=8, restore_best_weights=True)], verbose=1)
cnn_metrics = get_metrics('CNN', np.argmax(cnn.predict(X_te_cnn, verbose=0), axis=1), time.time() - t0, '#c0392b')
results.append(cnn_metrics)
print(f"Accuracy: {cnn_metrics['accuracy']}%  |  F1: {cnn_metrics['f1']}%")


print("\nTraining LSTM...")
X_tr_lstm = X_train.reshape(-1, 1, X_train.shape[1])
X_va_lstm = X_val.reshape(-1, 1, X_val.shape[1])
X_te_lstm = X_test.reshape(-1, 1, X_test.shape[1])

lstm_model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(1, X_train.shape[1])),
    Dropout(0.3),
    LSTM(64), Dropout(0.3),
    Dense(128, activation='relu'), Dropout(0.3),
    Dense(n_classes, activation='softmax')
])
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

t0 = time.time()
lstm_model.fit(X_tr_lstm, y_tr_cat, validation_data=(X_va_lstm, y_va_cat),
               epochs=50, batch_size=64,
               callbacks=[EarlyStopping(patience=8, restore_best_weights=True)], verbose=1)
lstm_metrics = get_metrics('LSTM', np.argmax(lstm_model.predict(X_te_lstm, verbose=0), axis=1), time.time() - t0, '#6a1b9a')
results.append(lstm_metrics)
print(f"Accuracy: {lstm_metrics['accuracy']}%  |  F1: {lstm_metrics['f1']}%")


with open('results.json', 'w') as f:
    json.dump(results, f)

print("\nDone! Results saved to results.json")
print("\n{:<16} {:>10} {:>10}".format('Model', 'Accuracy', 'F1'))
print("-" * 40)
for r in results:
    print("{:<16} {:>9}% {:>9}%".format(r['model'], r['accuracy'], r['f1']))
