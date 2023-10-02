C++ API
===========

.. cpp:namespace:: poly_ops

Usage
++++++++++++++++

Basic Usage
----------------------

.. code-block:: cpp

    using namespace poly_ops;

    std::vector<point_t<int>> loop_a{
        {58,42},
        {36,52},
        {20,34},
        {32,13},
        {55,18}};
    std::vector<point_t<int>> loop_b{
        {76,58},
        {43,56},
        {45,23},
        {78,26}};

    auto output = boolean_op<false,int>(loop_a,loop_b,bool_op::difference);
    for(auto loop : output) {
        for(point_t<int> p : loop) {
            std::cout << p[0] << ',' << p[1] << '\n';
        }
        std::cout << '\n';
    }

Or using the :cpp:class:`clipper` class:

.. code-block:: cpp

    clipper<int> clip;
    clip.add_loops_subject(loop_a);
    clip.add_loops_clip(loop_b);

    auto output2 = clip.execute<false>(bool_op::xor_);

Using the :cpp:class:`clipper` class is more efficient than calling
:cpp:func:`boolean_op` multiple times because the class will reuse the memory it
allocated.

In both cases, the output is meant to be a temporary object that is traversed
only once (although there is no harm in traversing it multiple times) and
consumed or stored in a data structure of the user's choice. It references the
internal representation used by the algorithm and is not sequential.

In the case of the :cpp:class:`clipper` class, the output references the data
owned by the class. If the class is destroyed or if any of its methods are
called, the output is invalidated. The exception is, given an instance of
:cpp:class:`clipper` `clip`, calling ``std::move(clip).execute<...>(...)`` will
return an object that takes ownership of the data.


Input Data Format
----------------------------------------

The functions and methods of this library take sequences of 2-dimensional
points, representing closed polygons, where each point is connected to the next
point in the sequence, and the last point is connected to the first.

The inputs can either be ranges (objects that satisfy `std::ranges::range`) of
points (objects that satisfy :cpp:concept:`point`), which represent individual
polygons, or ranges of ranges of points, which represent sets of polygons.

E.g. polygons can be added like this:

.. code-block:: cpp

    std::vector<poly_ops::point_t<int>> loop_a{...};
    std::vector<poly_ops::point_t<int>> loop_b{...};

    poly_ops::clipper<int> clip;
    clip.add_loops_subject(loop_a);
    clip.add_loops_subject(loop_b);

or like this:

.. code-block:: cpp

    std::vector<std::vector<poly_ops::point_t<int>>> loops{...};

    poly_ops::clipper<int> clip;
    clip.add_loops_subject(loops);

In the case of the :cpp:class:`clipper` class, instead of providing a range,
points can be added one at a time using a "point sink":

.. code-block:: cpp

    poly_ops::clipper<int> clip;

    /* the sink object needs to be destroyed before using any other method
    on "clip", hence the curly braces here */
    {
        auto sink = clip.add_loop(poly_ops::bool_set::subject);
        for(int i=0; i<100; ++i) sink(get_next_point(i));

        sink({0,0});
    }

Under the hood, all add_loops- methods and even the free functions use the point
sink to avoid template bloat.

Except for the :cpp:enumerator:`normalize<bool_op::normalize>` operation, the
operations interpret clockwise polygons (assuming positive X is right and
positive Y is down) and clockwise portions of self-intersecting polygons as
solid geometry and counter-clockwise polygons/parts as holes. Input polygons
with fewer than three points are silently discarded.

.. figure:: /_static/union_inverted.svg

    Union operation with polygon that is half clockwise and half
    counter-clockwise.


Custom Point Type in Input
-------------------------------

By default, given a type `T`: `point_t<T>`, `T[2]` and `std::span<T,2>` satisfy
:cpp:concept:`point`. To use your own point type in the input instead of one of
the above, specialize :cpp:class:`point_ops` like so:

.. code-block:: cpp

    struct MyPoint {
        float X;
        float Y;
    };

    template<> struct poly_ops::point_ops<MyPoint> {
        static int get_x(const MyPoint &p) {
            return static_cast<int>(MyPoint.X * 100.0f);
        }
        static int get_y(const MyPoint &p) {
            return static_cast<int>(MyPoint.Y * 100.0f);
        }
    };

    int main() {
        ...

        std::vector<MyPoint> loop_a{
            {5.8f,4.2f},
            {3.6f,5.2f},
            {2.0f,3.4f},
            {3.2f,1.3f},
            {5.5f,1.8f}};
        std::vector<MyPoint> loop_b{
            {7.6f,5.8f},
            {4.3f,5.6f},
            {4.5f,2.3f},
            {7.8f,2.6f}};

        auto output = poly_ops::boolean_op<false,int>(
            loop_a,loop_b,poly_ops::bool_op::difference);

        ...
    }

The functions and methods of poly_ops will still return instances of
:cpp:class:`point_t`.


Getting the Results as a Hierarchy
----------------------------------------

.. code-block:: cpp

    void print_indent(int level) {
        for(int i=0; i<level; ++i) std::cout << "  ";
    }

    void print_polygon(const poly_ops::temp_polygon_proxy<int> &poly,int level=0) {
        for(auto p : poly) {
            print_indent(level);
            std::cout << p[0] << ',' << p[1] << '\n';
        }
        std::cout << '\n';
        auto child_polys = poly.inner_loops();
        if(!child_polys.empty()) {
            print_indent(level);
            std::cout << "nested polygons:\n";
            for(auto child : child_polys) print_polygon(child,level+1);
        }
    }

    int main() {
        ...

        auto output = poly_ops::boolean_op<true,int>(subject_loops,clip_loops,bool_op::normalize);
        for(auto loop : output) print_polygon(loop);

        ...
    }


Coord Template Parameter
-------------------------------

The `Coord` template parameter of the functions and classes of this library is
the type that will represent coordinate values. This can be any built-in signed
integer or any class that satisfies :cpp:concept:`coordinate`. When using your
own class, :cpp:struct:`coord_ops` needs to be specialized to provide equivalent
mathematical functions.


Index Template Parameter
--------------------------------

The `Index` template parameter of the functions and classes of this library is
the type that will represent the numeric indices of the points. By default, this
is `std::size_t`. This library uses this type extensively in it's internal data
structures, thus using a smaller type can save some memory; however, it must be
large enough to represent the total number of points plus two times the number
of intersections, of a single operation. Unlike `Coord`, this must be one of the
built-in integers.


Reference
++++++++++++++

.. toctree::
    base
    clip
    offset
    large_ints
