class RagError(Exception):
    pass


class UnsupportedFileType(RagError):
    pass


class EmptyDocumentError(RagError):
    pass


class InvalidChunkingParameters(RagError):
    pass


class InconsistentChunkData(RagError):
    pass


class RerankError(RagError):
    pass
