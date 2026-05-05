from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tracer:
    """
    Defines a galaxy tracer (galaxy type) and its associated parameters.

    Parameters
    ----------
    name : str
        Unique identifier for the tracer (e.g. "LRG", "ELG", "QSO").
    params : dict[str, Any]
        Tracer-specific parameters forwarded to the backend
        (e.g. HOD parameters, magnitude cuts, color selections).
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Provide a string representation of the tracer, including its name and parameters."""
        return f"Tracer(name={self.name!r}, params={self.params})"
