from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunResult:
    """Outcome of executing a compiled kernel binary.

    Reports what happened; does not judge whether it was acceptable.
    That judgment belongs to the referee.
    """
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class Backend(ABC):

    @abstractmethod
    def compile(self, source: Path, output: Path) -> Path:
        raise NotImplementedError

    @abstractmethod
    def run(self, executable: Path) -> RunResult:
        raise NotImplementedError