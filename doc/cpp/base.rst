poly_ops/base.hpp
=================

.. cpp:namespace:: poly_ops


Concepts
-----------

.. cpp:concept:: template<typename T> coordinate

    Defined as:

    .. code-block:: cpp

        detail::arithmetic<T>
        && detail::arithmetic<typename coord_ops<T>::long_t>
        && detail::arithmetic<real_coord_t<T>>
        && detail::arithmetic_promotes<T,long_coord_t<T>>
        && detail::arithmetic_promotes<T,real_coord_t<T>>
        && std::totally_ordered<T>
        && std::totally_ordered<long_coord_t<T>>
        && std::totally_ordered<real_coord_t<T>>
        && std::convertible_to<T,long_coord_t<T>>
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
        }

    where ``detail::arithmetic<typename T>`` is:

    .. code-block:: cpp

        requires(T x) {
            { x + x } -> std::same_as<T>;
            { x - x } -> std::same_as<T>;
            { x * x } -> std::same_as<T>;
            { x / x } -> std::same_as<T>;
            { -x } -> std::same_as<T>;
        }

    and ``detail::arithmetic_promotes<typename Lesser,typename Greater>`` is:

    .. code-block:: cpp

        requires(Lesser a,Greater b) {
            { a + b } -> std::same_as<Greater>;
            { b + a } -> std::same_as<Greater>;
            { a - b } -> std::same_as<Greater>;
            { b - a } -> std::same_as<Greater>;
            { a * b } -> std::same_as<Greater>;
            { b * a } -> std::same_as<Greater>;
            { a / b } -> std::same_as<Greater>;
            { b / a } -> std::same_as<Greater>;
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

        std::ranges::sized_range<T> && point_range<std::ranges::range_value_t<T>,Coord>


Classes/Structs
------------------

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

    .. cpp:type:: long long_t

    .. cpp:type:: double real_t

    .. cpp:function:: static real_t acos(Coord x)

        Default implementation:

        .. code-block:: cpp

            return std::acos(static_cast<real_t>(x));

    .. cpp:function:: static real_t acos(real_t x)

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

        Return a value with the same sign as "x" but with a magnitude of 1

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

    .. cpp:function:: template<point<T> U> constexpr point_t(const U &b)

        Construct `point_t` from any point-like object

    .. cpp:function:: constexpr T &operator[](std::size_t i) noexcept

    .. cpp:function:: constexpr const T &operator[](std::size_t i) const noexcept

    .. cpp:function:: constexpr T &x() noexcept

    .. cpp:function:: constexpr const T &x() const noexcept

    .. cpp:function:: constexpr T &y() noexcept

    .. cpp:function:: constexpr const T &y() const noexcept

    .. cpp:function:: constexpr T *begin() noexcept

    .. cpp:function:: constexpr const T *begin() const noexcept

    .. cpp:function:: constexpr T *end() noexcept

    .. cpp:function:: constexpr const T *end() const noexcept

    .. cpp:function:: constexpr std::size_t size() const noexcept

        Always returns 2

    .. cpp:function:: constexpr T *data() noexcept

        Return a pointer to the underlying array

    .. cpp:function:: constexpr const T *data() const noexcept

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

.. cpp:function:: template<typename T,typename U>\
    constexpr point_t<decltype(std::declval<T>()-std::declval<U>())>\
    operator-(const point_t<T> &a,const point_t<U> &b)

.. cpp:function:: template<typename T,typename U>\
    constexpr point_t<decltype(std::declval<T>()*std::declval<U>())>\
    operator*(const point_t<T> &a,const point_t<U> &b)

.. cpp:function:: template<typename T,typename U>\
    constexpr point_t<decltype(std::declval<T>()/std::declval<U>())>\
    operator/(const point_t<T> &a,const point_t<U> &b)

.. cpp:function:: template<typename T,typename U>\
    constexpr point_t<decltype(std::declval<T>()*std::declval<U>())>\
    operator*(const point_t<T> &a,U b)

.. cpp:function:: template<typename T,typename U>\
    constexpr point_t<decltype(std::declval<T>()*std::declval<U>())>\
    operator*(T a,const point_t<U> &b)

.. cpp:function:: template<typename T>\
    constexpr bool operator==(const point_t<T> &a,const point_t<T> &b)

.. cpp:function:: template<typename T>\
    constexpr bool operator!=(const point_t<T> &a,const point_t<T> &b)


.. cpp:function:: template<typename T,typename U>\
    constexpr auto vdot(const point_t<T> &a,const point_t<U> &b)

    Return the dot product of `a` and `b`.

.. cpp:function:: template<typename T> constexpr T square(const point_t<T> &a)

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
    polygons, more efficient methods exist. The magnitude of the return value is
    two times the area of the polygon.
