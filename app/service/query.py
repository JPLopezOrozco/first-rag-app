import logging
import re
from app.exceptions.exceptions import RerankError
from app.repository.repository import hybrid_search, rerank_search
from app.schema import QueryResponse, QueryRequest, Source
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()
logger = logging.getLogger(__name__)

ALPHA = 0.75

SYSTEM_PROMPT = (
    "Sos un asistente que responde preguntas usando únicamente el contexto provisto. "
    "Cada fragmento del contexto está numerado como [1], [2], etc. "
    "Citá los fragmentos que uses con ese formato, por ejemplo [1]. "
    "Si la respuesta no está en el contexto, respondé exactamente: "
    "'No encuentro esa información en los documentos.' "
    "No uses conocimiento externo."
)

NOT_FOUND = "No encuentro esa información en los documentos."
ROUTER_PROMPT = (
    "Clasificá el mensaje del usuario. Respondé SOLO una palabra.\n"
    "SMALLTALK: saludos, agradecimientos, despedidas, confirmaciones ('ok', 'dale', 'listo'), "
    "risas, o preguntas sobre vos ('¿quién sos?').\n"
    "SEARCH: cualquier mensaje que pida información, aunque parezca ajena o ambigua, "
    "y cualquier mensaje que mezcle saludo con pregunta.\n"
    "Ejemplos: 'ok' -> SMALLTALK; 'gracias!' -> SMALLTALK; "
    "'hola, ¿quién fundó el club?' -> SEARCH; '¿cómo se hace una pizza?' -> SEARCH."
)
SMALLTALK_REPLY = "Hola, preguntame lo que quieras sobre los documentos."

def need_retrieval(query:str)->bool:
    r = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=10,
        system=ROUTER_PROMPT,
        messages=[{"role": "user", "content": query }]
    )
    
    return not r.content[0].text.strip().upper().startswith("SMALLTALK")

def default_retrieve(query:str, n: int):
    try:
        return rerank_search(query, n)
    except RerankError:
        logger.warning("Falló el reranker, se usa la búsqueda híbrida", exc_info=True)
        return hybrid_search(query, n, ALPHA)
        
        

def answer(query:QueryRequest, retrieve=default_retrieve)->QueryResponse:
    
    if not need_retrieval(query.question):
        return QueryResponse(answer=SMALLTALK_REPLY, sources=[])
        
    retrieval = retrieve(query.question, query.n_results)
    
    hits = list(zip(retrieval["documents"][0], retrieval["metadatas"][0]))
    
    if not hits:
        return QueryResponse(answer=NOT_FOUND, sources=[])
    
    context = "\n\n".join(f"[{i}] {t}" for i, (t, _) in enumerate(hits, 1))
            
    augmented_response = (llm_call(query.question, context)).content[0].text
    
    if augmented_response.strip().startswith(NOT_FOUND):
        return QueryResponse(answer=NOT_FOUND, sources=[])

    cited = {int(n) for n in re.findall(r"\[(\d+)\]", augmented_response)}
    
    if not cited:
        return QueryResponse(answer=NOT_FOUND, sources=[])
    
    source = [
        Source(source=m["source"], chunk=m["chunk"], text=t)
            for i, (t,m)in enumerate(hits, 1) if i in cited
    ]
    
    return QueryResponse(answer=augmented_response, sources=source)

def build_user_message(context: str, question: str) -> str:
    return f"Contexto:\n{context}\n\nPregunta: {question}"
    
def llm_call(question:str, context:list[str]):
     
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(context, question)}],
    )
    
    return response
