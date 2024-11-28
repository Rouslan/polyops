polyops module
==================

.. py:module:: polyops


.. py:type:: PointArrayLike
    :canonical: numpy.typing.ArrayLike

    Arrays of this type must have a shape with a length of two, with the second
    dimension being two. In other words: it must be an array of two dimensional
    arrays.

    .. important::

        This type alias only exists in the stub file.


.. py:type:: PointArray
    :canonical: numpy.ndarray

    An array with a shape with a length of two, with the second dimension being
    two. In other words: an array of two dimensional arrays.

    .. important::

        This type alias only exists in the stub file.


.. py:type:: IndexArray
    :canonical: numpy.ndarray[Any,np.dtype[np.intp]]

    A one-dimensional array of integers.

    .. important::

        This type alias only exists in the stub file.


.. py:type:: CastingKind
    :canonical: Literal["no","equiv","safe","same_kind","unsafe"]

    .. important::

        This type alias only exists in the stub file.


.. py:class:: PointMap

    A jagged array of arrays of indices.

    The basic interface is that of a read-only sequence where each element is an
    array of indices corresponding to another array.
    
    For advanced usage, there are the attributes: ``offsets`` and ``indices``.
    All indices are stored contiguously in the one-dimensional array
    ``indices``. ``offsets`` is a one-dimensional array that contains all the
    start indices of the child arrays in ``indices``, followed by the length of
    ``indices``. This means the child array at index ``i`` is
    ``indices[offsets[i]:offsets[i+1]]``.

    .. py:attribute:: offsets
        :type: IndexArray

        This attribute is read-only
    
    .. py:attribute:: indices
        :type: IndexArray

        This attribute is read-only

    .. py:method:: __len__() -> int
    
    .. py:method:: __getitem__(i, /) -> IndexArray
    
    .. py:method:: index_map(out: IndexArray|None = None) -> IndexArray

        Return an array that maps the values in :py:attr:`indices` to their
        position in this jagged array.
        
        The returned value is equivalent to:

        .. code-block:: python

            numpy.concat([[i]*len(a) for i,a in enumerate(self)], dtype=numpy.intp)
        
        Since NumPy doesn't support jagged arrays, this is provided as an
        alternative means to allow pairing each output index with its input
        index while using NumPy's fast operations.

        If ``out`` is not ``None``, the data is written to ``out`` and returned,
        instead of creating a new array. The supplied array must have the same
        length as :py:attr:`indices`, and have a type of ``numpy.intp``.


.. py:class:: TrackedLoop

    .. py:attribute:: loop
        :type: PointArray

        A polygon represented by an array of points.
    
    .. py:attribute:: originals
        :type: PointMap

        A mapping of the indices of `loop` to the indices of the input arrays.


.. py:class:: RecursiveLoop

    .. py:attribute:: loop
        :type: numpy.ndarray

        A polygon represented by an array of points.
    
    .. py:attribute:: children
        :type: tuple[RecursiveLoop,...]

        The polygons inside of `loop`.


.. py:class:: TrackedRecursiveLoop

    .. py:attribute:: loop
        :type: PointArray

        A polygon represented by an array of points.

    .. py:attribute:: children
        :type: tuple[TrackedRecursiveLoop,...]

        The polygons inside of `loop`.

    .. py:attribute:: originals
        :type: PointMap

        A mapping of the indices of `loop` to the indices of the input arrays.


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


.. py:function:: union(loops: Iterable[PointArrayLike],*,casting: CastingKind = "same_kind",dtype: DTypeLike = None,tree_out: bool = False,track_points: bool = False)

    Generate the union of a set of polygons.

    The return type depends on ``tree_out`` and ``track_points``:

    ============ ================ ====================================
    ``tree_out`` ``track_points`` return type
    ============ ================ ====================================
    ``False``    ``False``        ``tuple[PointArray, ...]``
    ``False``    ``True``         ``tuple[TrackedLoop, ...]``
    ``True``     ``False``        ``tuple[RecursiveLoop, ...]``
    ``True``     ``True``         ``tuple[TrackedRecursiveLoop, ...]``
    ============ ================ ====================================


.. py:function:: normalize(loops: Iterable[PointArrayLike],*,casting: CastingKind = "same_kind",dtype: DTypeLike = None,tree_out: bool = False,track_points: bool = False)

    Return polygons consisting of the same lines as ``loops`` except all outer
    lines are clockwise polygons, all singly nested lines are counter-clockwise
    polygons, all doubly-nested polygons are clockwise polygons, and so forth.

    The return type depends on ``tree_out`` and ``track_points``:

    ============ ================ ====================================
    ``tree_out`` ``track_points`` return type
    ============ ================ ====================================
    ``False``    ``False``        ``tuple[PointArray, ...]``
    ``False``    ``True``         ``tuple[TrackedLoop, ...]``
    ``True``     ``False``        ``tuple[RecursiveLoop, ...]``
    ``True``     ``True``         ``tuple[TrackedRecursiveLoop, ...]``
    ============ ================ ====================================


.. py:function:: boolean_op(subject: Iterable[PointArrayLike],clip: Iterable[PointArrayLike],op: BoolOp,*,casting: CastingKind = "same_kind",dtype: DTypeLike = None,tree_out: bool = False,track_points: bool = False)

    Perform a boolean operation on two sets of polygons.

    The return type depends on ``tree_out`` and ``track_points``:

    ============ ================ ====================================
    ``tree_out`` ``track_points`` return type
    ============ ================ ====================================
    ``False``    ``False``        ``tuple[PointArray, ...]``
    ``False``    ``True``         ``tuple[TrackedLoop, ...]``
    ``True``     ``False``        ``tuple[RecursiveLoop, ...]``
    ``True``     ``True``         ``tuple[TrackedRecursiveLoop, ...]``
    ============ ================ ====================================


.. py:function::  offset(loops: Iterable[PointArrayLike],magnitude: float,arc_step_size: int,*,casting: CastingKind = "same_kind",dtype: DTypeLike = None,tree_out: bool = False,track_points: bool = False)

    Inflate or shrink the union of ``loops``.

    The return type depends on ``tree_out`` and ``track_points``:

    ============ ================ ====================================
    ``tree_out`` ``track_points`` return type
    ============ ================ ====================================
    ``False``    ``False``        ``tuple[PointArray, ...]``
    ``False``    ``True``         ``tuple[TrackedLoop, ...]``
    ``True``     ``False``        ``tuple[RecursiveLoop, ...]``
    ``True``     ``True``         ``tuple[TrackedRecursiveLoop, ...]``
    ============ ================ ====================================


.. py:function:: winding_dir(loop: PointArrayLike,*,casting: CastingKind = "same_kind") -> int

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

    .. py:method:: add_loop(loop: PointArrayLike,bset: BoolSet,*,casting: CastingKind = "same_kind") -> None

        Add an input polygon.

    .. py:method:: add_loop_subject(loop: PointArrayLike,*,casting: CastingKind = "same_kind") -> None

        Add an input *subject* polygon.

    .. py:method:: add_loop_clip(loop: PointArrayLike,*,casting: CastingKind = "same_kind") -> None

        Add an input *clip* polygon.

    .. py:method:: add_loops(loops: PointArrayLike,bset: BoolSet,*,casting: CastingKind = "same_kind") -> None

        Add input polygons.

    .. py:method:: add_loops_subject(loops: PointArrayLike,*,casting: CastingKind = "same_kind") -> None

        Add input *subject* polygons.

    .. py:method:: add_loops_clip(loops: PointArrayLike,*,casting: CastingKind = "same_kind") -> None

        Add input *clip* polygons.

    .. py:method:: add_loop_offset(self,loop: PointArrayLike,bset: BoolSet,magnitude: float,arc_step_size: int,*,casting: CastingKind = "same_kind") -> None

    .. py:method:: add_loop_offset_subject(self,loop: PointArrayLike,magnitude: float,arc_step_size: int,*,casting: CastingKind = "same_kind") -> None

    .. py:method:: add_loop_offset_clip(self,loop: PointArrayLike,magnitude: float,arc_step_size: int,*,casting: CastingKind = "same_kind") -> None

    .. py:method:: add_loops_offset(self,loops: Iterable[PointArrayLike],bset: BoolSet,magnitude: float,arc_step_size: int,*,casting: CastingKind = "same_kind") -> None

    .. py:method:: add_loops_offset_subject(self,loops: Iterable[PointArrayLike],magnitude: float,arc_step_size: int,*,casting: CastingKind = "same_kind") -> None

    .. py:method:: add_loops_offset_clip(self,loops: Iterable[PointArrayLike],magnitude: float,arc_step_size: int,*,casting: CastingKind = "same_kind") -> None

    .. py:method:: execute(op: BoolOp,*,dtype: DTypeLike = None,tree_out: bool = False)

        Perform a boolean operation and return the result.

        The return value depends on ``tree_out``:

        ============ =============================
        ``tree_out`` return type
        ============ =============================
        ``False``    ``tuple[PointArray, ...]``
        ``True``     ``tuple[RecursiveLoop, ...]``
        ============ =============================

        After calling this function, all the input is consumed. To perform
        another operation, polygons must be added again.

    .. py:method:: reset() -> None

        Discard all polygons added so far.


.. py:class:: TrackedClipper

    This is identical to :py:class:`Clipper` except the return value of
    :py:meth:`~Clipper.execute` is ``tuple[TrackedLoop, ...]`` if ``tree_out``
    is ``False``, and ``tuple[TrackedRecursiveLoop, ...]`` if ``tree_out`` is
    ``True``.
