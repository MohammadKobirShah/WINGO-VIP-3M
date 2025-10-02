from collections import Counter

def number_to_size(n):
    return "Big" if n >= 5 else "Small"

def parse_api_list(items):
    parsed = []
    for it in items:
        n = int(it.get("number"))
        color_raw = it.get("color") or ""
        colors = [c.strip().lower() for c in color_raw.split(",") if c.strip()]
        parsed.append({"number": n, "colors": colors, "issue": it.get("issueNumber")})
    return parsed

def heuristic_predict(parsed):
    nums = [p["number"] for p in parsed]
    sizes = [number_to_size(n) for n in nums]
    colors_flat = [c for p in parsed for c in p["colors"]]

    # Size rule
    if sizes[0] == sizes[1] == "Big":
        size = "Small"
    elif sizes[0] == sizes[1] == "Small":
        size = "Big"
    else:
        size = Counter(sizes).most_common(1)[0][0]

    # Color rule
    if "red" in parsed[0]["colors"] and sizes[0] == sizes[1] == "Big":
        color = "Green"
    else:
        color = Counter(colors_flat).most_common(1)[0][0].capitalize()

    # Numbers
    numbers = [1, 2, 3] if size == "Small" else [6, 7, 8]

    return {"size": size, "color": color, "numbers": numbers}
