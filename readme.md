# RAG sobre documentos, sin frameworks

Sistema de preguntas y respuestas sobre documentos (PDF, DOCX, TXT) con **citas a las fuentes**. Pipeline completo escrito a mano: ingesta, chunking, embeddings, búsqueda híbrida, reranking, generación y evaluación.

## Qué hace

1. Subís un documento (`POST /documents`): se parsea, se divide en fragmentos y se indexa.
2. Hacés una pregunta en lenguaje natural (`POST /question`).
3. El sistema recupera los fragmentos más relevantes y Claude responde **solo con ese contexto**, citando los fragmentos usados con `[1]`, `[2]`, etc.
4. Si la respuesta no está en los documentos, responde que no la encuentra en vez de inventar.

## Arquitectura

```mermaid
flowchart LR
  subgraph Ingesta["Ingesta - POST /documents"]
    A["Archivo PDF / DOCX / TXT"] --> B["Parser"]
    B --> C["Chunking por tokens<br/>500 tokens, overlap 100"]
    C --> D["Embeddings<br/>Voyage voyage-4-lite"]
    D --> E[("Postgres 17 + pgvector<br/>vector(1024), HNSW coseno<br/>tsvector español, GIN")]
    C --> E
  end
  subgraph Consulta["Consulta - POST /question"]
    Q["Pregunta"] --> R{"Router<br/>Claude Haiku"}
    R -- SMALLTALK --> S["Respuesta fija"]
    R -- SEARCH --> H["Candidatos<br/>unión: top 20 vectorial + top 20 palabras"]
    H --> RR["Reranker<br/>Voyage rerank-2.5, top 4"]
    RR -.->|"si falla"| FB["Fallback: búsqueda híbrida<br/>RRF, alpha 0.75"]
    RR --> P["Prompt con contexto numerado"]
    FB --> P
    P --> L["Claude Haiku 4.5"]
    L --> V{"¿Rechaza<br/>o no cita?"}
    V -- sí --> N["Respuesta de rechazo<br/>sin fuentes"]
    V -- no --> O["Respuesta + solo las fuentes citadas"]
  end
  E -.-> H
```

**Capas:** `main.py` (HTTP) → `service/` (lógica: ingesta y consulta) → `repository/` (acceso a Postgres, embeddings y reranker). Los servicios no conocen HTTP; el router traduce las excepciones de dominio a códigos de estado.

## Stack

Python 3.12 · FastAPI · Pydantic · Postgres 17 + pgvector · psycopg 3 · Voyage AI (embeddings y reranker) · Anthropic API (Claude Haiku 4.5) · tiktoken · pypdf · python-docx · pytest · Docker (para la base en desarrollo)

> Las dependencias `langchain-*` de `requirements.txt` las trae el SDK de Voyage; el código de este repo no usa LangChain ni LlamaIndex.

## Decisiones de diseño

| Decisión | Elegí | Por qué |
|---|---|---|
| Frameworks | Ninguno (código propio) | Entender y poder depurar cada paso. Costo: más código. |
| Chunking | Tamaño fijo por tokens, 500 con overlap de 100 | Baseline simple y medible. Se mide en tokens porque es lo que cuenta el modelo (`tiktoken` es una aproximación). Probado también 250/50 (ver resultados). |
| Embeddings | Voyage `voyage-4-lite`, pedidos explícitamente en lotes de 128 | Anthropic no ofrece API de embeddings; Voyage es el proveedor que su documentación sugiere. Con Postgres el embedding ya no lo hace la base de datos: lo pide el repositorio, antes de abrir la transacción. |
| Almacenamiento | Postgres + pgvector (HNSW, distancia coseno) | Un solo motor para vectores, texto y datos relacionales: transacciones, borrado en cascada y nada en memoria que sincronizar. Reemplazó a Chroma + un índice BM25 en memoria. Costo: hay que operar un servicio (Docker en desarrollo) y el keyword search deja de ser BM25. |
| Búsqueda por palabras | `tsvector` en español con índice GIN, consulta con **OR** | Postgres normaliza el texto al insertarlo (raíces, sin tildes ni palabras vacías). `websearch_to_tsquery` exige todos los términos (AND), y una pregunta como "¿Quién fundó el club?" se vuelve `quien & fund & club`, que ningún fragmento cumple. Se cambió a OR (ver hallazgos). Costo: `ts_rank_cd` no tiene IDF ni normaliza por largo, y su puntaje es grueso; los empates se desempatan por id para que las evals sean reproducibles. |
| Candidatos | **Unión** de los 20 primeros del vectorial y los 20 primeros de la búsqueda por palabras, sin repetidos | Cada rama cubre los puntos ciegos de la otra: el vectorial busca por significado, las palabras por términos exactos. No se ponderan ni se cortan: eso lo decide el reranker (ver hallazgos). |
| Reranker | Voyage `rerank-2.5` por API, se queda con los 4 mejores | Es un cross-encoder: lee la pregunta y cada fragmento juntos. Un cross-encoder multilingüe local pesa del orden de 2 GB y no entra en 512 MB de RAM. Costo: una llamada extra por consulta. |
| Fallback | Si el reranker falla, búsqueda híbrida con RRF (alpha 0.75) | La consulta degrada en calidad en vez de devolver un error. La falla queda registrada en el log. |
| Router | Llamada corta a Claude Haiku (`SMALLTALK` / `SEARCH`) | Evita buscar y generar ante saludos. Las preguntas ajenas al tema van a búsqueda y se rechazan, para no contestar con el conocimiento del modelo. Costo: una llamada extra por consulta. |
| Generación | Prompt que restringe al contexto + validación en código | Si la respuesta empieza con la frase de rechazo, o no cita ningún fragmento, se devuelve el rechazo sin fuentes. Las fuentes devueltas son solo las citadas. No se depende de que el modelo obedezca el prompt. |
| Ingesta | Idempotente y transaccional: en una sola transacción borra el documento (sus fragmentos caen en cascada) e inserta el nuevo | Re-subir un archivo lo reemplaza sin duplicar, y si la inserción falla queda la versión anterior intacta. Costo: el id interno del documento cambia en cada re-subida. |
| Conexiones | Pool de conexiones (`psycopg_pool`, hasta 10, configurable con `DB_POOL_MAX`), creado la primera vez que se usa | Evita abrir una conexión (y negociar TLS, si la base es remota) en cada consulta, y respeta el tope de conexiones de la base: si hay más pedidos que conexiones, esperan en cola hasta 10 segundos. `check_connection` descarta las que el servidor ya cerró. Costo: una dependencia más y un estado global que hay que cerrar al terminar. |
| Errores | Excepciones propias → 415 / 422 | Los errores del cliente se distinguen de los bugs, que salen como 500 y quedan en los logs. |
| Acceso | Clave de API en el header `X-API-Key`, límite de pedidos por clave (20 consultas y 5 subidas por minuto) y tope de subida de 10 MB, todo configurable por variables de entorno | Cada consulta gasta API de Claude y Voyage, y subir un archivo reemplaza el que tenga el mismo nombre: sin esto cualquiera podría gastar tu dinero o pisar documentos. Si falta `API_KEY` los endpoints devuelven 503 en vez de quedar abiertos. Costo: una sola clave compartida, y el contador de pedidos vive en la memoria de un proceso (se reinicia al redeployar y no se comparte entre réplicas). |

## Evaluación

**Dataset** (`evals/dataset.json`): 36 preguntas sobre el corpus, 25 con respuesta y 11 sin respuesta en los documentos (algunas ajenas al tema y otras del mismo tema). Cada pregunta con respuesta lleva una frase literal (`must_contain`) que tiene que aparecer en el fragmento recuperado; la comparación ignora tildes y mayúsculas. Corpus de prueba: 4 archivos (38 fragmentos con 500/100).

**Retrieval** (k = 4, `python3 -m evals.run_eval`), con **Postgres** (actual):

| Configuración | recall@4 | MRR |
|---|---|---|
| semantic (solo vectorial) | 0.88 | 0.73 |
| hybrid, alpha 0.75 | 0.92 | 0.79 |
| **rerank sobre la unión de candidatos (20+20)** | **1.00** | **1.00** |

Los mismos números con **Chroma + BM25 en memoria** (antes de la migración, commit `ee88a27`), donde además se barrió alpha y el tamaño de chunk:

| Configuración | recall@4 (500/100) | MRR (500/100) | recall@4 (250/50) | MRR (250/50) |
|---|---|---|---|---|
| semantic (solo vectorial) | 0.88 | 0.73 | 0.92 | 0.79 |
| hybrid, alpha 0 (solo BM25) | 0.80 | 0.76 | 0.80 | 0.70 |
| hybrid, alpha 0.25 | 0.88 | 0.76 | 0.80 | 0.73 |
| hybrid, alpha 0.5 | 0.92 | 0.81 | 0.84 | 0.72 |
| hybrid, alpha 0.75 | 0.96 | 0.80 | 0.92 | 0.73 |
| hybrid, alpha 1 (equivale a semantic) | 0.88 | 0.73 | 0.92 | 0.79 |
| **rerank sobre la unión de candidatos (20+20)** | **1.00** | **1.00** | no medido | no medido |

La migración mantiene el resultado del camino principal (rerank 1.00 / 1.00) y el vectorial idéntico. El híbrido, que ahora es solo el fallback, bajó de 0.96 a 0.92: una pregunta, dentro del ruido, pero la rama de palabras ya no es BM25 y no se puede afirmar que sean equivalentes.

Configuración por defecto: **reranker sobre la unión de candidatos, chunks de 500/100**, con hybrid (alpha 0.75) como fallback. Está pendiente de validar con preguntas que no se hayan usado para diseñarla.

**Extremo a extremo** (`python3 -m evals.e2e_evals`): con el reranker sobre Postgres, **11/11** preguntas sin respuesta rechazadas correctamente y **0/25** con respuesta rechazadas por error (una sola corrida). Antes de la migración, con búsqueda híbrida (alpha 0.75) sobre Chroma, eran 11/11 y 1/25 (la de "tabla histórica", ver hallazgos). Mide si el sistema responde o rechaza, **no** si la respuesta es correcta. Una corrida tarda unos 100 segundos y cuesta del orden de US$0,10 de API (estimado).

### Hallazgos

- **La fusión ponderada puede dejar afuera un acierto que solo encuentra una rama.** La pregunta "¿En qué puesto está en la tabla histórica?" fallaba con todos los métodos. Medí en qué posición (de 38) queda el fragmento correcto: **37° en el vectorial y 1° en BM25** (con Chroma). Con RRF ponderado y alpha 0.75, cada candidato del vectorial aporta como mínimo 0.75/80 ≈ 0.0094, mientras que un fragmento que solo BM25 encuentra (aunque sea 1°) aporta 0.25/61 ≈ 0.0041: queda siempre detrás de los 20 candidatos vectoriales. En una simulación con alpha 0.75 ese fragmento cae en la posición 21 y no llega ni al top 20 que se le pasaba al reranker; con alpha 0.5 entra segundo. Por qué el vectorial lo pone casi último no está verificado. La solución fue pasarle al reranker la **unión** de las dos ramas, sin ponderar. Con Postgres pasa lo mismo: el fragmento correcto no aparece en el top 5 del vectorial y la búsqueda por palabras lo devuelve primero.
- **`websearch_to_tsquery` exige todos los términos y las palabras interrogativas no se descartan.** "¿Quién fundó el club?" se convierte en `quien & fund & club`, y "¿Cuántos partidos invicto como local?" en `cuant & part & invict & local`: el fragmento que responde casi nunca contiene "quién" ni "cuántos", así que ambas, que sí tienen respuesta en el corpus, devolvieron cero resultados. Reemplazar `&` por `|` (OR) lo resuelve, a costa de traer también candidatos con una sola palabra en común, que el reranker descarta.
- **`ts_rank_cd` es un puntaje grueso y sin IDF.** Los valores de la prueba fueron 0.1, 0.3, 0.7, con muchos empates, y premia la repetición de términos: para "¿Quién fundó el club?" el primer resultado fue un fragmento que repite "club" y no el de la fundación. Con el reranker adelante esto importa poco (la rama solo tiene que *incluir* el fragmento correcto); como fallback, sí pesa.
- **Voyage no devuelve vectores idénticos entre llamadas.** La misma pregunta, embebida en dos corridas, dio distancias que difieren en el cuarto decimal (0.61573 y 0.61536 contra el mismo fragmento). Diferencias de ranking menores a ~1e-3 son ruido.
- **El umbral de distancia no sirve en este corpus.** Los aciertos tienen distancia coseno entre 0.33 y 0.72; las preguntas sin respuesta, entre 0.35 y 0.84, y las del mismo tema caen dentro del rango de los aciertos. La distancia mide cercanía de tema, no si el fragmento contiene la respuesta. Por eso el rechazo se delega al prompt y a la regla de citas. Falta probar si el score del reranker sí separa los dos grupos.
- **Híbrido ayudó con chunks de 500/100 y no con 250/50** (medido con Chroma y BM25). Con 250/50 la rama de palabras empeoró en todos los valores de alpha intermedios. No lo investigué; es una hipótesis pendiente.
- **Los chunks de 250/50 usan la mitad de contexto** (≈1000 contra ≈2000 tokens por consulta, estimado con 4 fragmentos), con un recall a una pregunta de distancia en semantic. Elegí calidad sobre costo con los números a la vista.

### Límites de la evaluación

Con 25 preguntas, una pregunta pesa 0.04 de recall: diferencias chicas son ruido. La unión de candidatos se diseñó después de analizar este mismo dataset, así que el 1.00 puede estar sobreajustado. El experimento es además generoso: hasta 40 candidatos (20 por rama) sobre un corpus de 38 fragmentos significa que el reranker ve casi todo el corpus, así que el 1.00 no demuestra que las dos ramas encuentren el fragmento correcto. Las preguntas se escribieron mirando el texto, lo que favorece a la búsqueda por palabras. La latencia y el costo del reranker todavía no se midieron. Falta un set de validación con preguntas parafraseadas.

## Cómo correrlo

Necesitás Docker (para Postgres) y Python 3.12.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Creá un `.env` con las claves. `API_KEY` es la clave que tienen que mandar los clientes de la API (generá una con `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`). Las demás variables son opcionales:

```
ANTHROPIC_API_KEY=...
VOYAGE_API_KEY=...
API_KEY=...
# DATABASE_URL=postgresql://rag:rag@localhost:5433/rag   (por defecto, el Postgres del docker-compose.yml)
# QUESTION_RATE_PER_MIN=20   consultas por minuto y por clave
# DOCUMENT_RATE_PER_MIN=5    subidas por minuto y por clave
# MAX_UPLOAD_MB=10           tamaño máximo de un archivo
# DB_POOL_MAX=10             conexiones máximas a Postgres
# LOG_LEVEL=INFO             nivel de logging
```

Levantá la base. La primera vez que se crea el volumen, `db/schema.sql` se aplica solo:

```bash
docker compose up -d
```

```bash
python3 -m scripts.reindex    # indexa los documentos de samples/
uvicorn main:app --reload
```

Subir un documento y preguntar:

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -H "X-API-Key: $API_KEY" \
  -F "file=@samples/banfield.txt;type=text/plain"

curl -X POST http://127.0.0.1:8000/question \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Quién fundó el club?", "n_results": 4}'
```

Estado de la app y de la base (no pide clave; responde 503 si la base no contesta):

```bash
curl http://127.0.0.1:8000/health    # {"status":"ok","database":"ok"}
```

Forma de la respuesta de `/question` (ejemplo ilustrativo, texto recortado):

```json
{
  "answer": "Los dos máximos ídolos del club son Julio César Falcioni y Garrafa Sánchez [1].",
  "sources": [
    { "source": "banfield.txt", "chunk": 1, "text": "…Sus dos máximos ídolos son «el emperador», Julio César Falcioni…" }
  ]
}
```

Tests y evaluaciones:

```bash
python3 -m pytest -v          # chunking, RRF y repositorio; los de integración necesitan Postgres y, si no está levantado, se saltean
python3 -m evals.run_eval     # retrieval: recall@4 y MRR (semantic, hybrid y rerank)
python3 -m evals.e2e_evals    # extremo a extremo: llama a la API de Claude
```

`run_eval` mide lo que esté indexado en ese momento; si cambiás el tamaño de chunk, reindexá los documentos antes de correrlo.

Si el volumen de Postgres ya existía, `schema.sql` no se vuelve a aplicar solo. Para aplicarlo a mano (es idempotente):

```bash
docker compose exec -T db psql -U rag -d rag < db/schema.sql
```

Para correr la API en un contenedor (con la base del compose levantada; `host.docker.internal` funciona en Docker Desktop):

```bash
docker build -t rag-api .
docker run --rm -p 8000:8000 --env-file .env \
  -e DATABASE_URL=postgresql://rag:rag@host.docker.internal:5433/rag rag-api
```

## Estructura

```
main.py                  endpoints FastAPI
schema.py                modelos Pydantic de entrada y salida
security.py              clave de API y límite de pedidos
Dockerfile               imagen de la API
service/ingest.py        parseo, chunking, ingesta
service/query.py         router, generación, citas
service/health.py        chequeo de salud (base de datos)
repository/repository.py Postgres (pgvector y tsvector), embeddings, búsqueda híbrida (RRF), reranker
db/schema.sql            tablas documents y chunks, índices HNSW y GIN
docker-compose.yml       Postgres 17 con pgvector (puerto 5433)
exceptions/              excepciones de dominio
evals/                   dataset y scripts de evaluación
tests/                   tests unitarios y de integración (pytest)
scripts/                 utilidades (reindexado)
samples/                 documentos de ejemplo
```

## Limitaciones conocidas

- Corpus chico y de un solo tema; los resultados son orientativos.
- **Una sola clave compartida (`API_KEY`), sin usuarios ni permisos:** quien la tenga puede consultar y también reemplazar cualquier documento subiendo otro con el mismo nombre. El límite de pedidos está en memoria de un proceso, y el tope de subida lo comprueba la app pero el archivo ya llegó al servidor; conviene que el proxy de la plataforma también lo imponga.
- **Credenciales de desarrollo** (`rag` / `rag`) en `docker-compose.yml`: no sirven para un deploy.
- El reranker agrega una llamada a Voyage por consulta; si falla, se degrada a búsqueda híbrida. Su latencia y costo no están medidos.
- El schema se aplica solo al crear la base (`initdb.d`) y no hay migraciones.
- `ts_rank_cd` no es BM25 (sin IDF ni normalización por largo): la rama de palabras rankea peor, y se compensa con el reranker.
- La ingesta es síncrona (sin cola ni worker).
- Sin historial de conversación: cada pregunta es independiente ("¿y el estadio?" no funciona).
- Las citas no se verifican automáticamente (falta evaluar groundedness).
- El chunking corta por tokens y un fragmento puede mezclar varios temas; falta probar el corte por párrafos.

## Datos

Los textos de `samples/` provienen de Wikipedia en español (artículo sobre el Club Atlético Banfield), bajo licencia CC BY-SA 4.0.
