import enum
from typing import Any, Literal
from collections.abc import Iterable
import numpy as np
from numpy.typing import ArrayLike, DTypeLike

PointArray = np.ndarray[Any,np.dtype[np.int32]]
LoopTree = tuple[tuple[PointArray,'LoopTree'],...]
CastingKind = Literal["no","equiv","safe","same_kind","unsafe"]

class BoolOp(enum.Enum):
    union = ...
    intersection = ...
    xor = ...
    difference = ...
    normalize = ...

class BoolSet(enum.Enum):
    subject = ...
    clip = ...

def union_tree(
    loops: Iterable[ArrayLike],
    *,
    casting: CastingKind = ...,
    dtype: DTypeLike = ...) -> LoopTree: ...

def union_flat(
    loops: Iterable[ArrayLike],
    *,
    casting: CastingKind = ...,
    dtype: DTypeLike = ...) -> tuple[PointArray,...]: ...

def normalize_tree(
    loops: Iterable[ArrayLike],
    *,
    casting: CastingKind = ...,
    dtype: DTypeLike = ...) -> LoopTree: ...

def normalize_flat(
    loops: Iterable[ArrayLike],
    *,
    casting: CastingKind = ...,
    dtype: DTypeLike = ...) -> tuple[PointArray,...]: ...

def boolean_op_tree(
    subject: Iterable[ArrayLike],
    clip: Iterable[ArrayLike],
    op: BoolOp,
    *,
    casting: CastingKind = ...,
    dtype: DTypeLike = ...) -> LoopTree: ...

def boolean_op_flat(
    loops: Iterable[ArrayLike],
    clip: Iterable[ArrayLike],
    op: BoolOp,
    *,
    casting: CastingKind = ...,
    dtype: DTypeLike = ...) -> tuple[PointArray,...]: ...

def winding_dir(loop: ArrayLike,*,casting: CastingKind = ...) -> int: ...

class Clipper:
    def add_loop(self,loop: ArrayLike,cat: BoolSet,*,casting: CastingKind = ...) -> None: ...

    def add_loop_subject(self,loop: ArrayLike,*,casting: CastingKind = ...) -> None: ...

    def add_loop_clip(self,loop: ArrayLike,*,casting: CastingKind = ...) -> None: ...

    def add_loops(self,loops: Iterable[ArrayLike],cat: BoolSet,*,casting: CastingKind = ...) -> None: ...

    def add_loops_subject(self,loops: Iterable[ArrayLike],*,casting: CastingKind = ...) -> None: ...

    def add_loops_clip(self,loops: Iterable[ArrayLike],*,casting: CastingKind = ...) -> None: ...

    def execute_tree(self,op: BoolOp,*,dtype: DTypeLike = ...) -> LoopTree: ...

    def execute_flat(self,op: BoolOp,*,dtype: DTypeLike = ...) -> tuple[PointArray,...]: ...

    def reset(self) -> None: ...
