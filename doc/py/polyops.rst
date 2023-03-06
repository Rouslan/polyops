poly_ops module
==================

.. py:module:: poly_ops

.. py:data:: PointArray
    :value: np.ndarray[Any,np.dtype[np.int32]]

.. py:data:: LoopTree
    :value: tuple[tuple[PointArray,LoopTree],...]

.. py:class:: BoolOp

    .. py:attribute:: union

    .. py:attribute:: intersection

    .. py:attribute:: xor

    .. py:attribute:: difference


.. py:class:: BoolCat

    .. py:attribute:: subject

    .. py:attribute:: clip


.. py:function:: union_tree(loops: Iterable[PointArray]) -> LoopTree

    Generate the union of a set of polygons.

.. py:function:: union_flat(loops: Iterable[PointArray]) -> tuple[PointArray,...]

    Generate the union of a set of polygons.

.. py:function:: boolean_op_tree(subject: Iterable[PointArray],clip: Iterable[PointArray],op: BoolOp) -> LoopTree

    Perform a boolean operation on two sets of polygons.

.. py:function:: boolean_op_flat(loops: Iterable[PointArray],clip: Iterable[PointArray],op: BoolOp) -> tuple[PointArray,...]

    Perform a boolean operation on two sets of polygons.

.. py:function:: winding_dir(loop: PointArray) -> int

    Return a positive number if clockwise, negative if counter-clockwise and
    zero if degenerate or exactly half of the polygon's area is inverted.

    This algorithm works on any polygon. For non-overlapping non-inverting
    polygons, more efficient methods exist. The magnitude of the return value is
    two times the area of the polygon.

.. py:class:: Clipper

    A class for performing boolean clipping operations.

    An instance of `Clipper` will reuse its allocated memory for subsequent
    operations, making it more efficient than calling :py:func:`boolean_op_flat`
    or :py:func:`boolean_op_tree` for performing multiple operations.

    .. py:method:: add_loop(loop: PointArray,cat: BoolCat) -> None

    .. py:method:: add_loop_subject(loop: PointArray) -> None

    .. py:method:: add_loop_clip(loop: PointArray) -> None

    .. py:method:: add_loops(loops: PointArray,cat: BoolCat) -> None

    .. py:method:: add_loops_subject(loops: PointArray) -> None

    .. py:method:: add_loops_clip(loops: PointArray) -> None

    .. py:method:: execute_tree(op: BoolOp) -> LoopTree

    .. py:method:: execute_flat(op: BoolOp) -> tuple[PointArray,...]

    .. py:method:: reset() -> None

        Discard all loops added so far.
