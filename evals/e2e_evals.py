import json
from service.query import answer_hybrid, NOT_FOUND
from schema import QueryRequest

data = json.load(open("evals/dataset.json", encoding="utf-8"))
neg_ok = pos_rechazadas = 0

for d in data:
    res = answer_hybrid(QueryRequest(question=d["question"]))
    rechazo = res.answer == NOT_FOUND
    if d["must_contain"] is None:
        neg_ok += rechazo
        if not rechazo:
            print("INVENTÓ     |", d["question"], "|", res.answer[:90])
    elif rechazo:
        pos_rechazadas += 1
        print("RECHAZÓ MAL |", d["question"])

print(f"negativas rechazadas {neg_ok}/11 | positivas rechazadas por error {pos_rechazadas}/25")