# train/train_xgb.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from backend.model_utils import featurize_from_history, parse_api_list, number_to_size

HIST_PATH = "../backend/history.csv"
OUT_DIR = "../backend/models"
os.makedirs(OUT_DIR, exist_ok=True)

def load_history_as_parsed(n=None):
    df = pd.read_csv(HIST_PATH)
    # convert to list newest-first
    records = df.to_dict(orient="records")[::-1]
    return parse_api_list(records)[:n] if n else parse_api_list(records)

def build_dataset(parsed, window=8):
    X, y_size, y_color = [], [], []
    # parsed is newest-first; we need to produce samples where X uses last window and y is next period
    for i in range(window, len(parsed)):
        window_slice = parsed[i-window:i]
        target = parsed[i]
        feats = featurize_from_history(window_slice, window=window).ravel()
        X.append(feats)
        y_size.append(1 if target["number"]>=5 else 0)
        # color: choose 0:red,1:green,2:violet -> choose first color if multiple
        col = target["colors"][0] if target["colors"] else "red"
        col_idx = {"red":0,"green":1,"violet":2}.get(col.lower(),0)
        y_color.append(col_idx)
    return np.vstack(X), np.array(y_size), np.array(y_color)

def train():
    parsed = load_history_as_parsed()
    X, y_size, y_color = build_dataset(parsed, window=8)
    # simple TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    # size classifier
    size_clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=100)
    size_clf.fit(X, y_size)
    # color classifier
    color_clf = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", objective="multi:softprob", num_class=3, n_estimators=100)
    color_clf.fit(X, y_color)

    joblib.dump({"size": size_clf, "color": color_clf}, os.path.join(OUT_DIR, "model_ensemble.joblib"))
    print("Saved models to", OUT_DIR)

if __name__ == "__main__":
    train()
