from .buffer_memory import BufferMemory
from .summary_memory import SummaryMemory

def __getattr__(name: str):
    if name == "VectorMemory":
        from .vector_memory import VectorMemory
        return VectorMemory
    raise AttributeError(f"module 'memories' has no attribute {name!r}")

__all__ = ["BufferMemory", "SummaryMemory", "VectorMemory"]
