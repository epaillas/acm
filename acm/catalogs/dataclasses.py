from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pandas import DataFrame


@dataclass(eq=True, frozen=True)
class Tracer:
    """
    Defines a galaxy tracer (galaxy type) and its associated parameters.
    
    Frozen to ensure immutability and hashability, allowing it to be used as a key in dictionaries.
    The params field is excluded from comparison and hashing to allow for mutable parameters if needed, but the tracer identity is determined solely by its name.

    Parameters
    ----------
    name : str
        Unique identifier for the tracer (e.g. "LRG", "ELG", "QSO").
    params : dict[str, Any]
        Tracer-specific parameters forwarded to the backend
        (e.g. HOD parameters, magnitude cuts, color selections).
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


@dataclass
class Transform:
    """
    A named transform with its arguments, stored in the pipeline.

    Parameters
    ----------
    name : str
        Unique identifier, used to avoid duplicate transforms (e.g. "rsd", "ap").
    func : Callable[[DataFrame, ...], DataFrame]
        Pure function that takes a DataFrame and returns a transformed DataFrame.
    kwargs : dict
        Arguments forwarded to func at application time.
    tracer: str or None
        If specified, this transform only applies to the given tracer. If None, it applies to all tracers.
    """

    name: str
    func: Callable[..., DataFrame]
    kwargs: dict = field(default_factory=dict)
    tracer: str | None = None  # None means catalog-level

    def apply(self, data: DataFrame) -> DataFrame:
        """Apply the transform function to the data with the stored kwargs."""
        return self.func(data, **self.kwargs)
