import unicodedata
import json
from repository.repository import search, hybrid_search


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")

data = json.load(open("evals/dataset.json", encoding="utf-8"))

def run_eval(search_fn, label, data, k=4, verbose=False):
    ranks, hit_d, miss_d = [], [], []

    for item in data:
        r = search_fn(item["question"], k)
        docs = r["documents"][0]
        dists = (r.get("distances") or [None])[0]
        needle = item["must_contain"]

        if needle is None:
            if dists:
                miss_d.append(dists[0])
            continue

        rank = next((i for i, d in enumerate(docs, 1) if norm(needle) in norm(d)), None)
        ranks.append(rank)
        if rank and dists:
            hit_d.append(dists[rank - 1])

        if verbose and rank is None:
            pass
        
    n = len(ranks)
        
    return {
        "label": label,
        "recall": sum(r is not None for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
        "worst_hit": max(hit_d) if hit_d else None,
        "best_miss": min(miss_d) if miss_d else None,
    }


if __name__ == "__main__":
    runs = [run_eval(search, "semantic 250/50", data)]
    for a in (0, 0.25, 0.5, 0.75, 1):
        runs.append(run_eval(lambda q, k, a=a: hybrid_search(q, k, alpha=a), f"hybrid a={a}", data))

    for x in runs:
        print(f"{x['label']:<20} recall {x['recall']:.2f}  MRR {x['mrr']:.2f}")

