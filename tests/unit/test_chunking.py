from app.service.ingest import chunking_overlap, _enc
from pytest import raises

def test_overlap_overflow():
    text = ' '.join(str(i) for i in range(3000))
    with raises(ValueError):
         chunking_overlap(text, chunk_size=500, overlap=500)

def test_overlap():
    size, overlap = 500, 100
    text = " ".join(str(i) for i in range(3000))
    tokens = _enc.encode(text)
    zone = _enc.decode(tokens[size - overlap:size])
    
    result = chunking_overlap(text, chunk_size=size, overlap=overlap)
    
    
    assert result[0].endswith(zone)
    assert result[1].startswith(zone)
