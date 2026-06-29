"""Module for typing definitions used in the ACM package."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import lsstypes

type LsstypeObject = "lsstypes.ObservableLeaf | lsstypes.ObservableTree"
