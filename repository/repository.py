from dotenv import load_dotenv
from psycopg.types.json import Jsonb
import os
import voyageai
import psycopg

load_dotenv()

DSN = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag")
EMBED_MODEL = "voyage-4-lite"
EMBED_BATCH = 128

vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

def embed_texts(texts: list[str]) -> list[list[float]]:
    resultado = []
    for i in range(0, len(texts), EMBED_BATCH):
        lote = texts[i:i + EMBED_BATCH]
        resultado.extend(vo.embed(texts=lote, model=EMBED_MODEL).embeddings)
    return resultado

def to_vector_literal(vector: list[float]) -> str:
    return f"{vector}".replace(" ", "")


def add_chunks(filename: str, chunks: list[str], metadata: list[dict], ids: list[str], content_type: str) -> None:
    if not (len(chunks) == len(metadata) == len(ids)):
        raise ValueError("chunks, metadata e ids deben tener el mismo largo")

    embeddings = embed_texts(chunks)

    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE filename = %s;", (filename,))
            cur.execute(
                "INSERT INTO documents (filename, content_type) VALUES (%s, %s) RETURNING id;",
                (filename, content_type),
            )
            document_id = cur.fetchone()[0]
            for i, chunk in enumerate(chunks):
                cur.execute(
                    "INSERT INTO chunks (id, document_id, chunk, content, metadata, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s::vector)",
                    (ids[i], document_id, i, chunk, Jsonb(metadata[i]), to_vector_literal(embeddings[i])),
                )
def vector_rows(query_embedding: list[float], n: int) -> list[tuple]:
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, content, metadata, embedding <=> %s::vector AS distance
                FROM chunks
                ORDER BY distance
                LIMIT %s;
                """,
                (to_vector_literal(query_embedding), n),
            )
            return cur.fetchall()


def search(query: str, n_results: int = 4) -> dict:
    rows = vector_rows(embed_texts([query])[0], n_results)
    return {
        "ids": [[r[0] for r in rows]],
        "documents": [[r[1] for r in rows]],
        "metadatas": [[r[2] for r in rows]],
        "distances": [[r[3] for r in rows]],
    }

def lexical_rows(query: str, n: int) -> list[tuple]:
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH q AS (
                    SELECT replace(websearch_to_tsquery('spanish', %s)::text, '&', '|')::tsquery AS query
                )
                SELECT id, content, metadata, ts_rank_cd(tsv, query) AS rank
                FROM chunks, q
                WHERE tsv @@ query
                ORDER BY rank DESC, id
                LIMIT %s;
                """,
                (query, n),
            )
            return cur.fetchall()

def candidate_rows(query: str, n_each: int = 10) -> list[tuple]:
    by_id = {}
    for row in vector_rows(embed_texts([query])[0], n_each) + lexical_rows(query, n_each):
        by_id.setdefault(row[0], row)
    return list(by_id.values())

def rerank_search(query: str, k: int = 4, n_each: int = 20) -> dict:
    rows = candidate_rows(query, n_each)
    if not rows:
        return {"documents": [[]], "metadatas": [[]], "scores": [[]]}

    rr = vo.rerank(query, [r[1] for r in rows], model="rerank-2.5", top_k=k)
    return {
        "documents": [[x.document for x in rr.results]],
        "metadatas": [[rows[x.index][2] for x in rr.results]],
        "scores": [[x.relevance_score for x in rr.results]],
    }

def hybrid_search(query: str, n_results: int = 4, alpha: float = 0.5, pool: int = 20) -> dict:
    vec = vector_rows(embed_texts([query])[0], pool)
    lex = lexical_rows(query, pool)
    fused = rrf([[r[0] for r in vec], [r[0] for r in lex]], [alpha, 1 - alpha])[:n_results]
    by_id = {r[0]: r for r in vec + lex}
    return {
        "ids": [fused],
        "documents": [[by_id[i][1] for i in fused]],
        "metadatas": [[by_id[i][2] for i in fused]],
    }


def rrf(rankings, weights, k=60):
    scores = {}
    for ranking, w in zip(rankings, weights):
        for pos, id_ in enumerate(ranking, 1):
            scores[id_] = scores.get(id_, 0) + w / (k + pos)
    return sorted(scores, key=scores.get, reverse=True)






    