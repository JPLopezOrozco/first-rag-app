from pathlib import Path
from service.ingest import process_document

for p in sorted(Path("samples").glob("banfield*.txt")):
    n = process_document(p.read_bytes(), p.name, "text/plain")
    print(p.name, n)
