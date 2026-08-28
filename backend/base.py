from abc import ABC, abstractmethod
from pathlib import Path


class Backend(ABC):

    @abstractmethod
    def compile(self, source: Path, output: Path) -> Path:
        raise NotImplementedError

    @abstractmethod
    def run(self, executable: Path) -> str:
        raise NotImplementedError