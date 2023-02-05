from typing import Any
from collections.abc import Iterable
import numpy as np

PointArray = np.ndarray[Any,np.dtype[np.int32]]
LoopTree = tuple[tuple[PointArray,'LoopTree'],...]

def normalize_tree(loops: Iterable[PointArray]) -> LoopTree: ...

def normalize_flat(loops: Iterable[PointArray]) -> tuple[PointArray,...]: ...

def winding_dir(loop: PointArray) -> int: ...
