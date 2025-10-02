# backend/model_utils.py
import joblib
from collections import Counter
import numpy as np

def load_model(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

def parse_api_list(items):
    """
    Accepts list of items from your API: newest-first or oldest-first.
    Standardize to newest-first list of dicts: {"number": int, "colors": ["red",...], "issue": str}
    """
    parsed = []
    for it in items:
        # flexible keys
        num = it.get("number") or it.get("num") or it.get("Number")
        try:
            n = int(num)
        except Exception:
            # sometimes in the form {"number":"8"}; try direct cast
            n = int(str(num).strip()) if num is not None else None
        color_raw = it.get("color") or it.get("colors") or ""
        if isinstance(color_raw, str):
            colors = [c.strip() for c in color_raw.split(",") if c.strip()]
        elif isinstance(color_raw, list):
            colors = color_raw
        else:
            colors = []
        parsed.append({"number": n, "colors": [c.lower() for c in colors], "issue": it.get("issueNumber") or it.get("issue") or it.get("issueNumber")})
    # We assume the list provided is newest-first; if you pass oldest-first change caller.
    return parsed

def number_to_size(n):
    return "Big" if n >= 5 else "Small"

def heuristic_predict(parsed):
    """
    parsed: newest-first list of {number, colors}
    returns: {"size":..., "color":..., "numbers":[top3]}
    """
    if not parsed:
        return {"size":"Small", "color":"Red", "numbers":[1,2,3]}

    nums = [p["number"] for p in parsed if p["number"] is not None]
    sizes = [number_to_size(n) for n in nums]
    colors_flat = [c for p in parsed for c in p["colors"]]

    # Size rule: if last two are Big -> Small; if last two Small -> Big; else majority
    size = None
    if len(sizes) >= 2:
        if sizes[0] == sizes[1] == "Big":
            size = "Small"
        elif sizes[0] == sizes[1] == "Small":
            size = "Big"
    if not size:
        size = Counter(sizes).most_common(1)[0][0] if sizes else "Small"

    # Color rule: majority color in window; if last was red and last two were Big -> favor Green reversion
    color = None
    if colors_flat:
        top_color = Counter(colors_flat).most_common(1)[0][0]
        last_colors = parsed[0]["colors"]
        if "red" in last_colors and len(sizes) >= 2 and sizes[0]==sizes[1]=="Big":
            color = "Green"
        else:
            color = top_color.capitalize()
    else:
        color = "Red"

    # Numbers suggestion
    if size == "Small":
        number_candidates = [1,2,3]
    else:
        number_candidates = [6,7,8]

    return {"size": size, "color": color, "numbers": number_candidates}

def featurize_from_history(parsed, window=8):
    """
    Build a flat feature vector for model usage from parsed newest-first.
    Example features: last N numbers, lastN sizes, red_count, green_count, last_number, streak_len
    """
    nums = [p["number"] for p in parsed][:window]
    # pad with -1
    nums_padded = nums + [-1]*(window-len(nums))
    sizes = [1 if n>=5 else 0 for n in nums_padded]  # Big=1 Small=0
    red_count = sum(1 for p in parsed[:window] for c in p["colors"] if c=="red")
    green_count = sum(1 for p in parsed[:window] for c in p["colors"] if c=="green")
    violet_count = sum(1 for p in parsed[:window] for c in p["colors"] if c=="violet")
    last_num = nums[0] if nums else -1
    streak = 1
    for i in range(1, len(nums)):
        if number_to_size(nums[i]) == number_to_size(nums[0]):
            streak += 1
        else:
            break
    feats = nums_padded + sizes + [red_count, green_count, violet_count, last_num, streak]
    return np.array(feats).reshape(1, -1)

def predict_with_model(model, X):
    """
    model expected to output (size_prob, color_probs, topk numbers)
    This is a wrapper; actual model training must create such outputs or we adapt predict methods.
    For simplicity: model.predict_proba(X) for size & color; and model_num for numbers.
    """
    out = {}
    try:
        # Example: model is a dict of {'size':clf1,'color':clf2,'num':clf3}
        if isinstance(model, dict):
            size_clf = model.get("size")
            color_clf = model.get("color")
            num_clf = model.get("num")
            if size_clf:
                p = size_clf.predict_proba(X)[0]
                out["size"] = {"label": "Big" if p[1] > 0.5 else "Small", "prob": float(p[1])}
            if color_clf:
                p2 = color_clf.predict_proba(X)[0]
                # assume order [red,green,violet]
                out["color"] = [{"label":lbl, "prob": float(prob)} for lbl,prob in zip(["Red","Green","Violet"], p2)]
            if num_clf:
                probs = num_clf.predict_proba(X)[0]  # 10-way
                topk_idx = probs.argsort()[-3:][::-1]
                out["top_numbers"] = [{"n": int(i), "prob": float(probs[i])} for i in topk_idx]
        else:
            # fallback simple: single classifier for size only
            p = model.predict_proba(X)[0]
            out["size"] = {"label":"Big" if p[1]>0.5 else "Small", "prob": float(p[1])}
    except Exception as e:
        out = {"error": str(e)}
    return out
