poly_ops/clipper.hpp
=====================

.. cpp:namespace:: poly_ops


Types
------------------

.. cpp:class:: template<typename Index> point_tracker

    Point tracking common interface.

    Note that the destructor is not virtual, and is instead protected.

    .. cpp:function:: virtual void new_intersection(Index a,Index b) = 0

        Called when intersecting lines are broken and consequently new points
        are added.

        One of `a` or `b` will refer to a new point but not necessarily both (an
        endpoint touching the middle of another line only requires a new point
        for the other line). The index of a new point will always be one greater
        than the index of the last point added, thus, to know if a point is new,
        keep track of the number of points added.

        :param a: The index of the intersection point of the first line.
        :param b: The index of the intersection point of the second line.

    .. cpp:function:: virtual void point_merge(Index from,Index to) = 0

        Called when points are "merged".

        When parts of the shape are going to be truncated, this is called on the
        first point to be discarded as `from` and the point that follows it as
        `to`. If the next point is also to be discarded, this is called again
        with that point along with the point after that. This is repeated until
        the point reached is not going to be discarded, or the following point
        was already discarded because we've circled around the entire loop.

        @param from The index of the point to be discarded.
        @param to The index of the point that follows `from`.

    .. cpp:function:: virtual void point_added(Index original_i) = 0

        Called for every point initially added.

        Every added point has an implicit index (this index is unrelated to
        `original_i`). This method is first called when point zero is added.
        Every subsequent call corresponds to the index that is one greater than
        the previous call.

        `original_i` is the index of the input point that the added point
        corresponds to. The value is what the array index of the original point
        would be if all the input points were concatenated, in order, into a
        single array. This will not necessarily be called for every point of the
        input; duplicate consecutive points are ignored.

        :param original_i: The index of the input point that this added point
            corresponds to.


.. cpp:enum-class:: bool_op

    Given two sets of polygons, `subject` and `clip`, specifies which operation
    to perform.

    .. cpp::enumerator:: union_

        boolean operation `subject` OR `clip`.

    .. cpp::enumerator:: intersection

        boolean operation `subject` AND `clip`.

    .. cpp::enumerator:: xor_

        boolean operation `subject` XOR `clip`.

    .. cpp::enumerator:: difference

        boolean operation `subject` AND NOT `clip`.
    
    .. cpp:enumerator:: normalize

        Keep all lines but make it so all outer lines are clockwise polygons,
        all singly nested lines are counter-clockwise polygons, all
        doubly-nested polygons are clockwise polygons, and so forth.


.. cpp:enum-class:: bool_set

    Specifies one of two sets.

    .. cpp::enumerator:: subject

    .. cpp::enumerator:: clip


.. cpp:type:: template<typename Coord,typename Index=std::size_t> proto_loop_iterator

    An opaque type that models `std::forward_iterator`. The iterator yields
    instances of `point_t<Coord>`.


.. cpp:class:: template<typename Coord,typename Index=std::size_t> temp_polygon_proxy

    A representation of a polygon with zero or more child polygons.

    This class is not meant to be directly instantiated by users of this
    library. This class models `std::ranges::forward_range` and
    `std::ranges::sized_range` and yields instances of `point<Coord>`.

    .. cpp:function:: proto_loop_iterator<Coord<Index>> begin() const

        Get the iterator to the first element.

    .. cpp:function:: proto_loop_iterator<Coord,Index> end() const

        Get the end iterator

    .. cpp:function:: Index size() const

        Return the number of elements in this range.

    .. cpp:function:: auto inner_loops() const

        Return a range of `temp_polygon_proxy<Coord,Index>` instances
        representing the children of this polygon.


.. cpp:type:: template<typename Coord,typename Index=std::size_t>\
        borrowed_temp_polygon_tree_range

    An opaque type that models `std::ranges::forward_range` and
    `std::ranges::sized_range`.


.. cpp:type:: template<typename Coord,typename Index=std::size_t>\
        borrowed_temp_polygon_range

    An opaque type that models `std::ranges::forward_range` and
    `std::ranges::sized_range`.


.. cpp:type:: template<typename Coord,typename Index=std::size_t>\
        temp_polygon_tree_range

    An opaque type that models `std::ranges::forward_range` and
    `std::ranges::sized_range`.


.. cpp:type:: template<typename Coord,typename Index=std::size_t>\
        temp_polygon_range

    An opaque type that models `std::ranges::forward_range` and
    `std::ranges::sized_range`.


.. cpp:class:: template<coordinate Coord,std::integral Index=std::size_t> clipper

    A class for performing boolean clipping operations.

    An instance of `clipper` will reuse its allocated memory for subsequent
    operations, making it more efficient than calling :cpp:func:`boolean_op` for
    performing multiple operations.

    .. cpp:class:: point_sink

        .. cpp:function:: void operator()(const point_t<Coord> &p,Index orig_i)

        .. cpp:function:: Index last_orig_i() const

        .. cpp:function:: Index &last_orig_i()

    .. cpp:member:: point_tracker<Index> *pt

    .. cpp:function:: explicit clipper(\
            point_tracker<Index> *pt=nullptr,\
            std::pmr::memory_resource *_contig_mem=nullptr)

    .. cpp:function:: template<point_range<Coord> R> void add_loop(R &&loop,bool_set cat)

        Add an input polygon.

        The output returned by :cpp:func:`execute` is invalidated.

    .. cpp:function:: template<point_range<Coord> R> void add_loop_subject(R &&loop)

        Add an input *subject* polygon.

        The output returned by :cpp:func:`execute` is invalidated.

    .. cpp:function:: template<point_range<Coord> R> void add_loop_clip(R &&loop)

        Add an input *clip* polygon.

        The output returned by :cpp:func:`execute` is invalidated.

    .. cpp:function:: template<point_range_range<Coord> R> void add_loops(R &&loops,bool_set cat)

        Add input polygons.

        The output returned by :cpp:func:`execute` is invalidated.

    .. cpp:function:: template<point_range_range<Coord> R> void add_loops_subject(R &&loops)

        Add input *subject* polygons.

        The output returned by :cpp:func:`execute` is invalidated.

    .. cpp:function:: template<point_range_range<Coord> R> void add_loops_clip(R &&loops)

        Add input *clip* polygons.

        The output returned by :cpp:func:`execute` is invalidated.

    .. cpp:function:: point_sink add_loop(bool_set cat)

        Return a "point sink".

        This is an alternative to adding loops with ranges. The return value is
        a functor that allows adding one point at a time. The destructor of the
        return value must be called before any method of this instance of
        `clipper` is called afterwards.

        The output returned by :cpp:func:`execute` is invalidated.

    .. cpp:function:: void reset()

        Discard all polygons added so far.

        The output returned by :cpp:func:`execute` is invalidated.

    .. cpp:function:: template<bool TreeOut>\
        std::conditional_t<TreeOut,\
            borrowed_temp_polygon_tree_range<Coord,Index>,\
            borrowed_temp_polygon_range<Coord,Index>>\
        execute(bool_op op) &

        Perform a boolean operation and return the result.

        After calling this function, all the input is consumed. To perform
        another operation, polygons must be added again.

        .. important::

            The output of this function has references to data in this instance
            of `clipper`. The returned range is invalidated if the instance is
            destroyed or when any of the following are called:
            
                - :cpp:func:`reset`,
                - :cpp:func:`add_loop`
                - :cpp:func:`add_loop_subject`
                - :cpp:func:`add_loop_clip`
                - :cpp:func:`add_loops`
                - :cpp:func:`add_loops_subject`
                - :cpp:func:`add_loops_clip`
                - `execute()` (if called a second time)
            
            This means the return value cannot be fed directly back into the
            same instance of `clipper`. To keep the data, make a copy. The data
            is also not stored sequentially in memory.

    .. cpp:function:: template<bool TreeOut>\
        std::conditional_t<TreeOut,\
            temp_polygon_tree_range<Coord,Index>,\
            temp_polygon_range<Coord,Index>>\
        execute(bool_op op) &&

        Perform a boolean operation and return the result.


Functions
----------------

.. cpp:function:: template<bool TreeOut,coordinate Coord,std::integral Index=std::size_t,point_range_range<Coord> Input>\
    std::conditional_t<TreeOut,\
        temp_polygon_tree_range<Index,Coord>,\
        temp_polygon_range<Index,Coord>>\
    union_op(\
        Input &&input,\
        point_tracker<Index> *pt=nullptr,\
        std::pmr::memory_resource *contig_mem=nullptr)

    Generate the union of a set of polygons.

    This is equivalent to calling :cpp:func:`boolean_op` with an empty range
    passed to `clip` and :cpp:enumerator:`bool_op::union_` passed to `op`.


.. cpp:function:: template<bool TreeOut,coordinate Coord,std::integral Index=std::size_t,point_range_range<Coord> SInput,point_range_range<Coord> CInput>\
    std::conditional_t<TreeOut,\
        temp_polygon_tree_range<Index,Coord>,\
        temp_polygon_range<Index,Coord>>\
    boolean_op(\
        SInput &&subject,\
        CInput &&clip,\
        bool_op op,\
        point_tracker<Index> *pt=nullptr,\
        std::pmr::memory_resource *contig_mem=nullptr)

    Perform a boolean operation on two sets of polygons.
