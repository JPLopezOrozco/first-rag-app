import pytest

from app.exceptions import exceptions as ex


@pytest.mark.parametrize("cls", [
    ex.UnsupportedFileType,
    ex.EmptyDocumentError,
    ex.InvalidChunkingParameters,
    ex.InconsistentChunkData,
    ex.RerankError,
])
def test_domain_exceptions_share_a_base(cls):
    assert issubclass(cls, ex.RagError)
    assert not issubclass(cls, ValueError)
