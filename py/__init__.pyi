import enum
from typing import Any, Literal, NamedTuple, overload, SupportsInt
from collections.abc import Iterable
import numpy as np
from numpy.typing import ArrayLike, DTypeLike

PointArrayLike = ArrayLike
PointArray = np.ndarray
IndexArray = np.ndarray[Any,np.dtype[np.intp]]
CastingKind = Literal["no","equiv","safe","same_kind","unsafe"]

class PointMap:
    offsets: IndexArray
    indices: IndexArray

    def __len__(self) -> int: ...
    def __getitem__(self,i: SupportsInt, /) -> IndexArray: ...

    def index_map(self,out: IndexArray|None = ...) -> IndexArray: ...

# the real TrackedLoop is actually created with collections.namedtuple
class TrackedLoop(NamedTuple):
    loop: PointArray
    originals: PointMap

# the real RecursiveLoop is actually created with collections.namedtuple
class RecursiveLoop(NamedTuple):
    loop: PointArray
    children: tuple[RecursiveLoop,...]

# the real TrackedRecursiveLoop is actually created with collections.namedtuple
class TrackedRecursiveLoop(NamedTuple):
    loop: PointArray
    children: tuple[TrackedRecursiveLoop,...]
    originals: PointMap

class BoolOp(enum.IntEnum):
    union = ...
    intersection = ...
    xor = ...
    difference = ...
    normalize = ...

class BoolSet(enum.IntEnum):
    subject = ...
    clip = ...

@overload
def union(
    loops: Iterable[PointArrayLike],
    *,
    casting: CastingKind = ...,
    dtype: DTypeLike = ...,
    tree_out: Literal[False]=False,
    track_points: Literal[False]=False) -> tuple[PointArray,...]: ...

@overload
def union(
    loops: Iterable[PointArrayLike],
    *,
    casting: CastingKind = ...,
    dtype: DTypeLike = ...,
    tree_out: Literal[True]=...,
    track_points: Literal[False]=False) -> tuple[RecursiveLoop,...]: ...

@overload
def union(
    loops: Iterable[PointArrayLike],
    *,
    casting: CastingKind = ...,
    dtype: DTypeLike = ...,
    tree_out: Literal[False]=False,
    track_points: Literal[True]=...) -> tuple[TrackedLoop,...]: ...

@overload
def union(
    loops: Iterable[PointArrayLike],
    *,
    casting: CastingKind = ...,
    dtype: DTypeLike = ...,
    tree_out: Literal[True]=...,
    track_points: Literal[True]=...) -> tuple[TrackedRecursiveLoop,...]: ...

@overload
def normalize(
    loops: Iterable[PointArrayLike],
    *,
    casting: CastingKind = ...,
    dtype: DTypeLike = ...,
    tree_out: Literal[False]=False,
    track_points: Literal[False]=False) -> tuple[PointArray,...]: ...

@overload
def normalize(
    loops: Iterable[PointArrayLike],
    *,
    casting: CastingKind = ...,
    dtype: DTypeLike = ...,
    tree_out: Literal[True]=...,
    track_points: Literal[False]=False) -> tuple[RecursiveLoop,...]: ...

@overload
def normalize(
    loops: Iterable[PointArrayLike],
    *,
    casting: CastingKind = ...,
    dtype: DTypeLike = ...,
    tree_out: Literal[False]=False,
    track_points: Literal[True]=...) -> tuple[TrackedLoop,...]: ...

@overload
def normalize(
    loops: Iterable[PointArrayLike],
    *,
    casting: CastingKind = ...,
    dtype: DTypeLike = ...,
    tree_out: Literal[True]=...,
    track_points: Literal[True]=...) -> tuple[TrackedRecursiveLoop,...]: ...

@overload
def boolean_op(
    subject: Iterable[PointArrayLike],
    clip: Iterable[PointArrayLike],
    op: BoolOp,
    *,
    casting: CastingKind = ...,
    dtype: DTypeLike = ...,
    tree_out: Literal[False]=False,
    track_points: Literal[False]=False) -> tuple[PointArray,...]: ...

@overload
def boolean_op(
    subject: Iterable[PointArrayLike],
    clip: Iterable[PointArrayLike],
    op: BoolOp,
    *,
    casting: CastingKind = ...,
    dtype: DTypeLike = ...,
    tree_out: Literal[False]=False,
    track_points: Literal[True]=...) -> tuple[TrackedLoop,...]: ...

@overload
def boolean_op(
    subject: Iterable[PointArrayLike],
    clip: Iterable[PointArrayLike],
    op: BoolOp,
    *,
    casting: CastingKind = ...,
    dtype: DTypeLike = ...,
    tree_out: Literal[True]=...,
    track_points: Literal[False]=False) -> tuple[RecursiveLoop,...]: ...

@overload
def boolean_op(
    subject: Iterable[PointArrayLike],
    clip: Iterable[PointArrayLike],
    op: BoolOp,
    *,
    casting: CastingKind = ...,
    dtype: DTypeLike = ...,
    tree_out: Literal[True]=...,
    track_points: Literal[True]=...) -> tuple[TrackedRecursiveLoop, ...]: ...

@overload
def offset(
    loops: Iterable[PointArrayLike],
    magnitude: float,
    arc_step_size: int,
    *,
    casting: CastingKind = ...,
    dtype: DTypeLike = ...,
    tree_out: Literal[False]=False,
    track_points: Literal[False]=False) -> tuple[PointArray,...]: ...

@overload
def offset(
    loops: Iterable[PointArrayLike],
    magnitude: float,
    arc_step_size: int,
    *,
    casting: CastingKind = ...,
    dtype: DTypeLike = ...,
    tree_out: Literal[False]=False,
    track_points: Literal[True]=...) -> tuple[TrackedLoop,...]: ...

@overload
def offset(
    loops: Iterable[PointArrayLike],
    magnitude: float,
    arc_step_size: int,
    *,
    casting: CastingKind = ...,
    dtype: DTypeLike = ...,
    tree_out: Literal[True]=...,
    track_points: Literal[False]=False) -> tuple[RecursiveLoop,...]: ...

@overload
def offset(
    loops: Iterable[PointArrayLike],
    magnitude: float,
    arc_step_size: int,
    *,
    casting: CastingKind = ...,
    dtype: DTypeLike = ...,
    tree_out: Literal[True]=...,
    track_points: Literal[True]=...) -> tuple[TrackedRecursiveLoop, ...]: ...

def winding_dir(loop: PointArrayLike,*,casting: CastingKind = ...) -> int: ...

class Clipper:
    def add_loop(self,loop: PointArrayLike,bset: BoolSet,*,casting: CastingKind = ...) -> None: ...

    def add_loop_subject(self,loop: PointArrayLike,*,casting: CastingKind = ...) -> None: ...

    def add_loop_clip(self,loop: PointArrayLike,*,casting: CastingKind = ...) -> None: ...

    def add_loops(self,loops: Iterable[PointArrayLike],bset: BoolSet,*,casting: CastingKind = ...) -> None: ...

    def add_loops_subject(self,loops: Iterable[PointArrayLike],*,casting: CastingKind = ...) -> None: ...

    def add_loops_clip(self,loops: Iterable[PointArrayLike],*,casting: CastingKind = ...) -> None: ...

    def add_loop_offset(self,loop: PointArrayLike,bset: BoolSet,magnitude: float,arc_step_size: int,*,casting: CastingKind = ...) -> None: ...

    def add_loop_offset_subject(self,loop: PointArrayLike,magnitude: float,arc_step_size: int,*,casting: CastingKind = ...) -> None: ...

    def add_loop_offset_clip(self,loop: PointArrayLike,magnitude: float,arc_step_size: int,*,casting: CastingKind = ...) -> None: ...

    def add_loops_offset(self,loops: Iterable[PointArrayLike],bset: BoolSet,magnitude: float,arc_step_size: int,*,casting: CastingKind = ...) -> None: ...

    def add_loops_offset_subject(self,loops: Iterable[PointArrayLike],magnitude: float,arc_step_size: int,*,casting: CastingKind = ...) -> None: ...

    def add_loops_offset_clip(self,loops: Iterable[PointArrayLike],magnitude: float,arc_step_size: int,*,casting: CastingKind = ...) -> None: ...

    @overload
    def execute(
        self,
        op: BoolOp,
        *,
        dtype: DTypeLike = ...,
        tree_out: Literal[False]=False) -> tuple[PointArray,...]: ...
    
    @overload
    def execute(
        self,
        op: BoolOp,
        *,
        dtype: DTypeLike = ...,
        tree_out: Literal[True]=...) -> tuple[RecursiveLoop,...]: ...

    def reset(self) -> None: ...

class TrackedClipper:
    def add_loop(self,loop: PointArrayLike,bset: BoolSet,*,casting: CastingKind = ...) -> None: ...

    def add_loop_subject(self,loop: PointArrayLike,*,casting: CastingKind = ...) -> None: ...

    def add_loop_clip(self,loop: PointArrayLike,*,casting: CastingKind = ...) -> None: ...

    def add_loops(self,loops: Iterable[PointArrayLike],bset: BoolSet,*,casting: CastingKind = ...) -> None: ...

    def add_loops_subject(self,loops: Iterable[PointArrayLike],*,casting: CastingKind = ...) -> None: ...

    def add_loops_clip(self,loops: Iterable[PointArrayLike],*,casting: CastingKind = ...) -> None: ...

    def add_loop_offset(self,loop: PointArrayLike,bset: BoolSet,magnitude: float,arc_step_size: int,*,casting: CastingKind = ...) -> None: ...

    def add_loop_offset_subject(self,loop: PointArrayLike,magnitude: float,arc_step_size: int,*,casting: CastingKind = ...) -> None: ...

    def add_loop_offset_clip(self,loop: PointArrayLike,magnitude: float,arc_step_size: int,*,casting: CastingKind = ...) -> None: ...

    def add_loops_offset(self,loops: Iterable[PointArrayLike],bset: BoolSet,magnitude: float,arc_step_size: int,*,casting: CastingKind = ...) -> None: ...

    def add_loops_offset_subject(self,loops: Iterable[PointArrayLike],magnitude: float,arc_step_size: int,*,casting: CastingKind = ...) -> None: ...

    def add_loops_offset_clip(self,loops: Iterable[PointArrayLike],magnitude: float,arc_step_size: int,*,casting: CastingKind = ...) -> None: ...

    @overload
    def execute(
        self,
        op: BoolOp,
        *,
        dtype: DTypeLike = ...,
        tree_out: Literal[False]=False) -> tuple[TrackedLoop,...]: ...
    
    @overload
    def execute(
        self,
        op: BoolOp,
        *,
        dtype: DTypeLike = ...,
        tree_out: Literal[True]=...) -> tuple[TrackedRecursiveLoop,...]: ...

    def reset(self) -> None: ...
