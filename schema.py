from pydantic import BaseModel, Field, ConfigDict

class QueryRequest(BaseModel): 
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=1000)
    n_results: int = Field(default=4, ge=1, le=10)

class Source(BaseModel):
    source: str 
    text: str
    chunk: int
    distance: float | None = None
    
class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]