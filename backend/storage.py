import csv, os
from datetime import datetime

CSV_PATH = os.path.join(os.path.dirname(__file__), "history.csv")
FIELDNAMES = ["issueNumber","number","color","ts"]

def append_history(items):
    wrote = 0
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        for it in items:
            writer.writerow({
                "issueNumber": it.get("issueNumber"),
                "number": it.get("number"),
                "color": it.get("color"),
                "ts": datetime.utcnow().isoformat()
            })
            wrote += 1
    return wrote

def load_history_csv(take=100):
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH, newline="", encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    return rows[::-1][:take]
