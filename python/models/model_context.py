
from abc import ABC, abstractmethod

class ModelContext(ABC):
    """Abstract base class for model context, holding session-specific information."""
    def __init__(self):
        pass

    @abstractmethod
    def get_history(self):
        pass

    @abstractmethod
    def add_message(self, role: str, content: str):
        pass 