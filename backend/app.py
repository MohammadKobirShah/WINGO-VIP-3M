# backend/app.py
import os
import json
from flask import Flask, request, jsonify
from model_utils import parse_api_list, heuristic_predict, featurize_from_history, load_model, predict_with_model
from storage import append_history, load_history_csv

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "backend/model.joblib")
model = load_model(MODEL_PATH)  # may return None if no model file

@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({"status":"ok", "model_loaded": model is not None})

@app.route("/api/ingest", methods=["POST"])
def ingest():
    """
    Accepts JSON { "list": [...] } in same format as API or CSV upload.
    Stores to local CSV (append).
    """
    data = request.json
    if not data:
        return jsonify({"error":"no json provided"}), 400

    items = data.get("list") or data.get("data", {}).get("list")
    if not items:
        return jsonify({"error":"no list found"}), 400

    appended = append_history(items)
    return jsonify({"ingested": appended})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Body expected:
    {
      "source": "api" or "storage",
      "take": 8,
      "use_model": false
    }
    If source=="api", also allowed: {"api_url": "<url>"} (server-side fetch not implemented here)
    """
    payload = request.json or {}
    source = payload.get("source", "storage")
    take = int(payload.get("take", 8))
    use_model = bool(payload.get("use_model", False))

    # load history
    if source == "storage":
        history = load_history_csv(take=take)
    else:
        # if you wanted to pass the records directly:
        history = payload.get("records", [])

    if not history:
        return jsonify({"error":"no history available"}), 400

    parsed = parse_api_list(history)  # returns most-recent-first list of dicts

    # choose approach
    if use_model and model is not None:
        X = featurize_from_history(parsed)
        pred = predict_with_model(model, X)
        pred["method"] = "xgboost"
    else:
        pred = heuristic_predict(parsed)
        pred["method"] = "heuristic"

    return jsonify({"prediction": pred, "recent": parsed})

@app.route("/api/backtest", methods=["POST"])
def backtest():
    """
    Very basic backtest: feed historical storage and simulate naive heuristic predictions.
    Accept params: window, take, strategy.
    """
    payload = request.json or {}
    window = payload.get("window", 8)
    history = load_history_csv()
    # history expected oldest-first, we will simulate sliding window
    if len(history) < window + 1:
        return jsonify({"error":"not enough history for backtest", "have": len(history)}), 400

    parsed = parse_api_list(history[::-1])  # reverse to newest-first then we'll slide
    hits = 0
    total = 0
    results = []
    # simulate: for i from window to len(parsed)-1, predict using parsed[i-window:i], compare to parsed[i]
    for i in range(window, len(parsed)):
        window_slice = parsed[i-window:i]  # most-recent-first in our parse_api_list semantics
        actual = parsed[i]
        pred = heuristic_predict(window_slice)
        hit = (pred["size"].lower() == ("big" if actual["number"]>=5 else "small"))
        hits += int(hit)
        total += 1
        results.append({"index": i, "pred": pred, "actual": actual, "hit": hit})
    accuracy = hits/total if total else None
    return jsonify({"accuracy": accuracy, "total": total, "hits": hits, "results_sample": results[-10:]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
