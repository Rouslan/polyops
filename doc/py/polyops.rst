poly_ops module
==================

.. py:module:: poly_ops


.. py:data:: PointArray
    :value: np.ndarray

    Arrays of this type must have a shape with a length of two, with the second
    dimension being two. In other words: it must be an array of two dimensional
    arrays. This requirement is currently not enforced by the typing system, but
    functions that expect this type will throw an instance of `TypeError` if the
    requirement is not met.

    .. important::

        This type alias only exists in the stub file.


.. py:data:: LoopTree
    :value: tuple[tuple[PointArray,LoopTree],...]

    .. important::

        This type alias only exists in the stub file.


.. py:data:: CastingKind
    :value: Literal["no","equiv","safe","same_kind","unsafe"]

    .. important::

        This type alias only exists in the stub file.


.. py:class:: BoolOp

    Given two sets of polygons, `subject` and `clip`, specifies which operation
    to perform.

    .. py:attribute:: union

        boolean operation `subject` OR `clip`.

        .. image:: /_static/union.svg
            :alt: union operation example

    .. py:attribute:: intersection

        boolean operation `subject` AND `clip`.

        .. image:: /_static/intersection.svg
            :alt: intersection operation example

    .. py:attribute:: xor

        boolean operation `subject` XOR `clip`.

        .. image:: /_static/xor.svg
            :alt: xor operation example

    .. py:attribute:: difference

        boolean operation `subject` AND NOT `clip`.

        .. image:: /_static/difference.svg
            :alt: difference operation example
    
    .. py:attribute:: normalize

        Keep all lines but make it so all outer lines are clockwise polygons,
        all singly nested lines are counter-clockwise polygons, all
        doubly-nested polygons are clockwise polygons, and so forth.


.. py:class:: BoolSet

    Specifies one of two sets.

    .. py:attribute:: subject

    .. py:attribute:: clip


.. py:function:: union_tree(loops: Iterable[PointArray],*,casting: CastingKind = "same_kind",dtype: DTypeLike = None) -> LoopTree

    Generate the union of a set of polygons.


.. py:function:: union_flat(loops: Iterable[PointArray],*,casting: CastingKind = "same_kind",dtype: DTypeLike = None) -> tuple[PointArray,...]

    Generate the union of a set of polygons.


.. py:function:: normalize_tree(loops: Iterable[ArrayLike],*,casting: CastingKind = "same_kind",dtype: DTypeLike = None) -> LoopTree

    Return polygons consisting of the same lines as `loops` except all outer
    lines are clockwise polygons, all singly nested lines are counter-clockwise
    polygons, all doubly-nested polygons are clockwise polygons, and so forth.


.. py:function:: normalize_flat(loops: Iterable[ArrayLike],*,casting: CastingKind = "same_kind",dtype: DTypeLike = None) -> tuple[PointArray,...]

    Return polygons consisting of the same lines as `loops` except all outer
    lines are clockwise polygons, all singly nested lines are counter-clockwise
    polygons, all doubly-nested polygons are clockwise polygons, and so forth.


.. py:function:: boolean_op_tree(subject: Iterable[PointArray],clip: Iterable[PointArray],op: BoolOp,*,casting: CastingKind = "same_kind",dtype: DTypeLike = None) -> LoopTree

    Perform a boolean operation on two sets of polygons.


.. py:function:: boolean_op_flat(loops: Iterable[PointArray],clip: Iterable[PointArray],op: BoolOp,*,casting: CastingKind = "same_kind",dtype: DTypeLike = None) -> tuple[PointArray,...]

    Perform a boolean operation on two sets of polygons.


.. py:function::  offset_tree(loops: Iterable[ArrayLike],magnitude: float,arc_step_size: int,*,casting: CastingKind = "same_kind",dtype: DTypeLike = None) -> LoopTree

    Inflate or shrink the union of `loops`.


.. py:function::  offset_flat(loops: Iterable[ArrayLike],magnitude: float,arc_step_size: int,*,casting: CastingKind = "same_kind",dtype: DTypeLike = None) -> tuple[PointArray,...]

    Inflate or shrink the union of `loops`.


.. py:function:: winding_dir(loop: PointArray,*,casting: CastingKind = "same_kind") -> int

    Return a positive number if `loop` is clockwise, negative if
    counter-clockwise and zero if degenerate or exactly half of the polygon's
    area is inverted.

    This algorithm works on any polygon. For non-overlapping non-inverting
    polygons, more efficient methods exist.


.. py:class:: Clipper

    A class for performing boolean clipping operations.

    An instance of `Clipper` will reuse its allocated memory for subsequent
    operations, making it more efficient than calling :py:func:`boolean_op_flat`
    or :py:func:`boolean_op_tree` for performing multiple operations.

    .. py:method:: add_loop(loop: PointArray,bset: BoolSet,*,casting: CastingKind = "same_kind") -> None

        Add an input polygon.

    .. py:method:: add_loop_subject(loop: PointArray,*,casting: CastingKind = "same_kind") -> None

        Add an input *subject* polygon.

    .. py:method:: add_loop_clip(loop: PointArray,*,casting: CastingKind = "same_kind") -> None

        Add an input *clip* polygon.

    .. py:method:: add_loops(loops: PointArray,bset: BoolSet,*,casting: CastingKind = "same_kind") -> None

        Add input polygons.

    .. py:method:: add_loops_subject(loops: PointArray,*,casting: CastingKind = "same_kind") -> None

        Add input *subject* polygons.

    .. py:method:: add_loops_clip(loops: PointArray,*,casting: CastingKind = "same_kind") -> None

        Add input *clip* polygons.

    .. py:method:: add_loop_offset(self,loop: ArrayLike,bset: BoolSet,magnitude: float,arc_step_size: int,*,casting: CastingKind = "same_kind") -> None

    .. py:method:: add_loop_offset_subject(self,loop: ArrayLike,magnitude: float,arc_step_size: int,*,casting: CastingKind = "same_kind") -> None

    .. py:method:: add_loop_offset_clip(self,loop: ArrayLike,magnitude: float,arc_step_size: int,*,casting: CastingKind = "same_kind") -> None

    .. py:method:: add_loops_offset(self,loops: Iterable[ArrayLike],bset: BoolSet,magnitude: float,arc_step_size: int,*,casting: CastingKind = "same_kind") -> None

    .. py:method:: add_loops_offset_subject(self,loops: Iterable[ArrayLike],magnitude: float,arc_step_size: int,*,casting: CastingKind = "same_kind") -> None

    .. py:method:: add_loops_offset_clip(self,loops: Iterable[ArrayLike],magnitude: float,arc_step_size: int,*,casting: CastingKind = "same_kind") -> None

    .. py:method:: execute_tree(op: BoolOp,*,dtype: DTypeLike = None) -> LoopTree

        Perform a boolean operation and return the result.

        After calling this function, all the input is consumed. To perform
        another operation, polygons must be added again.

    .. py:method:: execute_flat(op: BoolOp,*,dtype: DTypeLike = None) -> tuple[PointArray,...]

        Perform a boolean operation and return the result.

        After calling this function, all the input is consumed. To perform
        another operation, polygons must be added again.

    .. py:method:: reset() -> None

        Discard all polygons added so far.
