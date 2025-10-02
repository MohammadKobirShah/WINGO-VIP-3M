import os
import requests
from flask import Flask, jsonify
from collections import Counter
from model_utils import number_to_size, heuristic_predict, parse_api_list

app = Flask(__name__)

API = "https://draw.ar-lottery01.com/WinGo/WinGo_3M/GetHistoryIssuePage.json"
PORT = int(os.environ.get("PORT", 5000))


def fetch_history():
    """Fetch last 8 results from API"""
    r = requests.get(API, timeout=6)
    r.raise_for_status()
    data = r.json()
    return data["data"]["list"][:8]


@app.route("/")
def home():
    return jsonify({"status": "ok", "msg": "Wingo VIP API running"})


@app.route("/predict")
def predict():
    history = fetch_history()
    parsed = parse_api_list(history)
    pred = heuristic_predict(parsed)
    return jsonify({"prediction": pred, "recent": parsed})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
