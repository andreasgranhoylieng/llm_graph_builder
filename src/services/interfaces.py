from abc import ABC, abstractmethod
from typing import List


class IFileRepository(ABC):
    @abstractmethod
    def get_files(self, folder_path: str, extensions: List[str]) -> List[str]:
        pass


class INeo4jRepository(ABC):
    @abstractmethod
    def add_graph_documents(self, graph_documents: List, include_source: bool = True):
        pass

    @abstractmethod
    def query(self, question: str) -> str:
        pass

    @abstractmethod
    def refresh_schema(self):
        pass
