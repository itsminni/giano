"""Small adapter contract. Publication and provenance belong to the importer."""

from abc import ABC, abstractmethod
from pathlib import Path


class DatasetAdapter(ABC):
    """Write only inside a fresh staging directory supplied by the caller."""

    @abstractmethod
    def source_paths(self) -> list[Path]:
        """Return every local input used by this adapter."""

    @abstractmethod
    def write(self, destination: Path) -> list[Path]:
        """Normalize inputs and return files under destination/<storage-group>."""
