import re
from repository.repository import search, hybrid_search
from schema import QueryResponse, QueryRequest, Source
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

ALPHA = 0.75

SYSTEM_PROMPT = (
    "Sos un asistente que responde preguntas usando únicamente el contexto provisto. "
    "Cada fragmento del contexto está numerado como [1], [2], etc. "
    "Citá los fragmentos que uses con ese formato, por ejemplo [1]. "
    "Si la respuesta no está en el contexto, respondé exactamente: "
    "'No encuentro esa información en los documentos.' "
    "No uses conocimiento externo."
)
MAX_DISTANCE = 0.7
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

def answer_question(query:QueryRequest)->QueryResponse:
    if not need_retrieval(query.question):
        return QueryResponse(answer=SMALLTALK_REPLY, sources=[])
        
    retrieval = search(query.question, query.n_results)
    
    hits = [
        (t,m,d)
        for t, m, d in zip(
            retrieval["documents"][0],
            retrieval["metadatas"][0],
            retrieval["distances"][0],
        )
        if d <= MAX_DISTANCE
    ]
    
    if not hits:
        return QueryResponse(answer=NOT_FOUND, sources=[])
    
    context = "\n\n".join(f"[{i}] {t}" for i, (t, _, _) in enumerate(hits, 1))
            
    augmented_response = (llm_call(query.question, context)).content[0].text
    
    source = [
        Source(source=m["source"], chunk=m["chunk"], distance=d, text=t)
            for t, m, d in hits
    ]
    
    return QueryResponse(answer=augmented_response, sources=source)

def answer_hybrid(query:QueryRequest)->QueryResponse:
    
    if not need_retrieval(query.question):
        return QueryResponse(answer=SMALLTALK_REPLY, sources=[])
        
    retrieval = hybrid_search(query.question, query.n_results, ALPHA)
    
    hits = [
        (t,m)
        for t, m in zip(
            retrieval["documents"][0],
            retrieval["metadatas"][0],
        )
    ]
    
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
            for t, m in hits
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