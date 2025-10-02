# backend/storage.py
import csv, os
from datetime import datetime

CSV_PATH = os.path.join(os.path.dirname(__file__), "history.csv")

FIELDNAMES = ["issueNumber","number","color","ts"]

def append_history(items):
    """
    items: list of dicts containing issueNumber, number, color
    Appends each to CSV with timestamp
    """
    wrote = 0
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        for it in items:
            try:
                writer.writerow({
                    "issueNumber": it.get("issueNumber") or it.get("issue") or "",
                    "number": it.get("number"),
                    "color": it.get("color"),
                    "ts": datetime.utcnow().isoformat()
                })
                wrote += 1
            except Exception:
                continue
    return wrote

def load_history_csv(take=100):
    """
    Returns list of dicts, newest-first
    """
    if not os.path.exists(CSV_PATH):
        return []
    rows = []
    with open(CSV_PATH, newline="", encoding="utf8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    # convert to same dict shape used by parse_api_list; assume file is oldest-first, so reverse
    rows = rows[::-1]
    return rows[:take]
