poly_ops/base.hpp
=================

.. cpp:namespace:: poly_ops


Concepts
-----------

.. cpp:concept:: template<typename T> coordinate

    Defined as:

    .. code-block:: cpp

        detail::arithmetic<T>
        && detail::arithmetic<long_coord_t<T>>
        && detail::arithmetic<real_coord_t<T>>
        && detail::arithmetic_promotes<T,long_coord_t<T>>
        && detail::arithmetic_promotes<T,real_coord_t<T>>
        && std::totally_ordered<T>
        && std::totally_ordered<long_coord_t<T>>
        && std::totally_ordered<real_coord_t<T>>
        && std::convertible_to<T,long_coord_t<T>>
        && std::convertible_to<int,long_coord_t<T>>
        && std::convertible_to<T,real_coord_t<T>>
        && requires(T c,long_coord_t<T> cl,real_coord_t<T> cr) {
            static_cast<long>(c);
            { coord_ops<T>::acos(c) } -> std::same_as<real_coord_t<T>>;
            { coord_ops<T>::acos(cr) } -> std::same_as<real_coord_t<T>>;
            { coord_ops<T>::cos(cr) } -> std::same_as<real_coord_t<T>>;
            { coord_ops<T>::sin(cr) } -> std::same_as<real_coord_t<T>>;
            { coord_ops<T>::sqrt(c) } -> std::same_as<real_coord_t<T>>;
            { coord_ops<T>::sqrt(cr) } -> std::same_as<real_coord_t<T>>;
            { coord_ops<T>::round(cr) } -> std::same_as<T>;
            { coord_ops<T>::ceil(cr) } -> std::same_as<T>;
            { coord_ops<T>::floor(cr) } -> std::same_as<T>;
            { coord_ops<T>::to_coord(cr) } -> std::same_as<T>;
            { coord_ops<T>::pi() } -> std::same_as<real_coord_t<T>>;
            { coord_ops<T>::mul(c,c) } -> std::same_as<long_coord_t<T>>;
            { cr * cr } -> std::same_as<real_coord_t<T>>;
            { cr / cr } -> std::same_as<real_coord_t<T>>;
            { c * cr } -> std::same_as<real_coord_t<T>>;
            { cr * c } -> std::same_as<real_coord_t<T>>;
        }

    where ``detail::arithmetic<typename T>`` is:

    .. code-block:: cpp

        requires(T x) {
            { x + x } -> std::same_as<T>;
            { x - x } -> std::same_as<T>;
            { -x } -> std::same_as<T>;
        }

    and ``detail::arithmetic_promotes<typename Lesser,typename Greater>`` is:

    .. code-block:: cpp

        requires(Lesser a,Greater b) {
            { a + b } -> std::same_as<Greater>;
            { b + a } -> std::same_as<Greater>;
            { a - b } -> std::same_as<Greater>;
            { b - a } -> std::same_as<Greater>;
        }

.. cpp:concept:: template<typename T,typename Coord> point

    Defined as:

    .. code-block:: cpp

        requires(const T &v) {
            { point_ops<T>::get_x(v) } -> std::convertible_to<Coord>;
            { point_ops<T>::get_y(v) } -> std::convertible_to<Coord>;
        }

.. cpp:concept:: template<typename T,typename Coord> point_range

    Defined as:

    .. code-block:: cpp

        std::ranges::range<T> && point<std::ranges::range_value_t<T>,Coord>

.. cpp:concept:: template<typename T,typename Coord> point_range_range

    Defined as:

    .. code-block:: cpp

        std::ranges::range<T> && point_range<std::ranges::range_value_t<T>,Coord>

.. cpp:concept:: template<typename T,typename Coord> point_range_or_range_range

    Defined as:

    .. code-block:: cpp

        std::ranges::range<T> && (point_range<std::ranges::range_value_t<T>,Coord>
            || point<std::ranges::range_value_t<T>,Coord>)


Types
------------------

.. cpp:type:: template<typename Coord> long_coord_t = typename coord_ops<Coord>::long_t;
.. cpp:type:: template<typename Coord> real_coord_t = typename coord_ops<Coord>::real_t;

.. cpp:struct:: template<typename T> point_ops

    A struct containing getters for point-like objects. Specializations exist
    for ``T[2]``, ``std::span<T,2>`` and ``point_t<T>``. This can be specialized
    by the user for other types. Static functions "get_x" and "get_y" should be
    defined to get the X and Y coordinates respectively.

    Example:

    .. code-block:: cpp

        template<> struct poly_ops::point_ops<MyPoint> {
            static int get_x(const MyPoint &p) {
                return static_cast<int>(MyPoint.X * 100.0f);
            }
            static int get_y(const MyPoint &p) {
                return static_cast<int>(MyPoint.Y * 100.0f);
            }
        };

.. cpp:struct:: template<typename Coord> coord_ops

    Mathematical operations on coordinate types. This struct can be specialized
    by users of this library. Arithmetic operators should be defined for every
    permutation of `Coord`, `long_t` and `real_t`. Binary operations with
    `Coord` and `long_t` should return `long_t`, `long_t` and `real_t` should
    return `real_t`, and `Coord` and `real_t` should return `real_t`.

    .. cpp:type:: long_t

        Certain operations require multiplying two coordinate values and thus
        need double the bits of the maximum coordinate value to avoid overflow.

        By default, if the compile target is a 64-bit platform and `Coord` is a
        64 bit type, this is :cpp:class:`basic_int128`. On other platforms, if
        `Coord` is not smaller than `long`, this is `long long`. Otherwise this
        is `long`.

        This can be specialized as a user-defined type.

    .. cpp:type:: real_t = double

        The coordinates of generated points are usually real numbers. By
        default, they are represented by `double` before being rounded back to
        integers. This type can be specialized to use `float` instead, or some
        user-defined type for more precision.
    
    .. cpp:function:: static long_t mul(Coord a,Coord b)

        Multiply `a` and `b` and return the result.

        This should be equivalent to
        :cpp:expr:`static_cast<long_t>(a) * static_cast<long_t>(b)`, except
        `long_t` is not required to support multiplication.

    .. cpp:function:: static real_t acos(Coord x)

        Default implementation:

        .. code-block:: cpp

            return std::acos(static_cast<real_t>(x));

    .. cpp:function:: static real_t acos(real_t x)
        :nocontentsentry:

        Default implementation:

        .. code-block:: cpp

            return std::acos(x);

    .. cpp:function:: static real_t cos(real_t x)

        Default implementation:

        .. code-block:: cpp

            return std::cos(x);

    .. cpp:function:: static real_t sin(real_t x)

        Default implementation:

        .. code-block:: cpp

            return std::sin(x);

    .. cpp:function:: static real_t sqrt(Coord x)

        Default implementation:

        .. code-block:: cpp

            return std::sqrt(static_cast<real_t>(x));

    .. cpp:function:: static real_t sqrt(real_t x)
        :nocontentsentry:

        Default implementation:

        .. code-block:: cpp

            return std::sqrt(x);

    .. cpp:function:: static Coord round(real_t x)

        Default implementation:

        .. code-block:: cpp

            return static_cast<Coord>(std::lround(x));

    .. cpp:function:: static Coord floor(real_t x)

        Default implementation:

        .. code-block:: cpp

            return static_cast<Coord>(std::floor(x));

    .. cpp:function:: static Coord ceil(real_t x)

        Default implementation:

        .. code-block:: cpp

            return static_cast<Coord>(std::ceil(x));

    .. cpp:function:: static Coord to_coord(real_t x)

        After determining how many points to use to approximate an arc using
        real numbers, the value needs to be converted to an integer to use in a
        loop.

        Default implementation:

        .. code-block:: cpp

            return static_cast<Coord>(x);

    .. cpp:function:: static real_t unit(real_t x)

        Return a value with the same sign as `x` but with a magnitude of 1

        Default implementation:

        .. code-block:: cpp

            return std::copysign(1.0,x);

    .. cpp:function:: static real_t pi()

        Return the value of pi.

        Default implementation:

        .. code-block:: cpp

            return std::numbers::pi_v<real_t>;


.. cpp:struct:: template<typename T> point_t

    .. cpp:member:: T _data[2]

    .. cpp:function:: point_t() = default

        The default constructor leaves the values uninitialized

    .. cpp:function:: constexpr point_t(const T &x,const T &y)
        :nocontentsentry:

    .. cpp:function:: template<point<T> U> constexpr point_t(const U &b)
        :nocontentsentry:

        Construct `point_t` from any point-like object

    .. cpp:function:: constexpr T &operator[](std::size_t i) noexcept

    .. cpp:function:: constexpr const T &operator[](std::size_t i) const noexcept
        :nocontentsentry:

    .. cpp:function:: constexpr T &x() noexcept

    .. cpp:function:: constexpr const T &x() const noexcept
        :nocontentsentry:

    .. cpp:function:: constexpr T &y() noexcept

    .. cpp:function:: constexpr const T &y() const noexcept
        :nocontentsentry:

    .. cpp:function:: constexpr T *begin() noexcept

    .. cpp:function:: constexpr const T *begin() const noexcept
        :nocontentsentry:

    .. cpp:function:: constexpr T *end() noexcept

    .. cpp:function:: constexpr const T *end() const noexcept
        :nocontentsentry:

    .. cpp:function:: constexpr std::size_t size() const noexcept

        Always returns 2

    .. cpp:function:: constexpr T *data() noexcept

        Return a pointer to the underlying array

    .. cpp:function:: constexpr const T *data() const noexcept
        :nocontentsentry:

        Return a pointer to the underlying array

    .. cpp:function:: constexpr point_t &operator+=(const point_t &b)

    .. cpp:function:: constexpr point_t &operator-=(const point_t &b)

    .. cpp:function:: constexpr point_t &operator*=(T b)

    .. cpp:function:: constexpr point_t operator-() const

    .. cpp:function:: friend constexpr void swap(point_t &a,point_t &b) noexcept(std::is_nothrow_swappable_v<T>)


Functions
----------------

.. cpp:function:: template<typename T,typename U>\
    constexpr point_t<decltype(std::declval<T>()+std::declval<U>())>\
    operator+(const point_t<T> &a,const point_t<U> &b)

    Element-wise addition.

    Equivalent to :cpp:expr:`point_t{a[0]+b[0],a[1]+b[1]}`

.. cpp:function:: template<typename T,typename U>\
    constexpr point_t<decltype(std::declval<T>()-std::declval<U>())>\
    operator-(const point_t<T> &a,const point_t<U> &b)

    Element-wise subtraction.

    Equivalent to :cpp:expr:`point_t{a[0]-b[0],a[1]-b[1]}`

.. cpp:function:: template<typename T,typename U>\
    constexpr point_t<decltype(std::declval<T>()*std::declval<U>())>\
    operator*(const point_t<T> &a,const point_t<U> &b)

    Element-wise multiplication.

    Equivalent to :cpp:expr:`point_t{a[0]*b[0],a[1]*b[1]}`

.. cpp:function:: template<typename T,typename U>\
    constexpr point_t<decltype(std::declval<T>()*std::declval<U>())>\
    operator*(const point_t<T> &a,U b)
    :nocontentsentry:

    Element-wise multiplication.

    Equivalent to :cpp:expr:`point_t{a[0]*b,a[1]*b}`

.. cpp:function:: template<typename T,typename U>\
    constexpr point_t<decltype(std::declval<T>()*std::declval<U>())>\
    operator*(T a,const point_t<U> &b)
    :nocontentsentry:

    Element-wise multiplication.

    Equivalent to :cpp:expr:`point_t{a*b[0],a*b[1]}`

.. cpp:function:: template<typename T,typename U>\
    constexpr point_t<decltype(std::declval<T>()/std::declval<U>())>\
    operator/(const point_t<T> &a,const point_t<U> &b)

    Element-wise division.

    Equivalent to :cpp:expr:`point_t{a[0]/b[0],a[1]/b[1]}`

.. cpp:function:: template<typename T>\
    constexpr bool operator==(const point_t<T> &a,const point_t<T> &b)

    Equivalent to :cpp:expr:`a[0] == b[0] && a[1] == b[1]`

.. cpp:function:: template<typename T>\
    constexpr bool operator!=(const point_t<T> &a,const point_t<T> &b)

    Equivalent to :cpp:expr:`a[0] != b[0] || a[1] != b[1]`

.. cpp:function:: template<typename T,typename U>\
    constexpr auto vdot(const point_t<T> &a,const point_t<U> &b)

    Return the dot product of `a` and `b`.

.. cpp:function:: template<typename T> constexpr T square(const point_t<T> &a)

    Equivalent to :cpp:expr:`a[0]*a[0] + a[1]*a[1]`

.. cpp:function:: template<typename T,typename U>\
    constexpr point_t<T> vcast(const point_t<U> &x)

.. cpp:function:: template<typename Coord>\
    constexpr point_t<Coord> vround(const point_t<real_coord_t<Coord>> &x)

.. cpp:function:: template<typename Coord,typename T>\
    constexpr real_coord_t<Coord> vmag(const point_t<T> &x)

.. cpp:function:: template<typename Coord,typename T>\
    constexpr real_coord_t<T> vangle(const point_t<T> &a,const point_t<T> &b)

.. cpp:function:: template<typename Coord>\
    constexpr long_coord_t<Coord> triangle_winding(\
        const point_t<Coord> &p1,\
        const point_t<Coord> &p2,\
        const point_t<Coord> &p3)

    Return a positive number if clockwise, negative if counter-clockwise and
    zero if degenerate.

.. cpp:function:: template<coordinate Coord,point_range<Coord> Points>\
    long_coord_t<Coord> winding_dir(Points &&points)

    Return a positive number if clockwise, negative if counter-clockwise and
    zero if degenerate or exactly half of the polygon's area is inverted.

    This algorithm works on any polygon. For non-overlapping non-inverting
    polygons, more efficient methods exist.
