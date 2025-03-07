#ifndef POLY_OPS_BASE_HPP
#define POLY_OPS_BASE_HPP

#include <cmath>
#include <numbers>
#include <utility>
#include <type_traits>
#include <span>
#include <ranges>

#include "large_ints.hpp"


#ifndef POLY_OPS_ASSERT
#define POLY_OPS_ASSERT assert
#endif

// used for checks that will significantly slow down the algorithm
#ifndef POLY_OPS_ASSERT_SLOW
#define POLY_OPS_ASSERT_SLOW(X) (void)0
#endif


namespace poly_ops {

/**
 * Mathematical operations on coordinate types. This struct can be specialized
 * by users of this library. Arithmetic operators should be defined for every
 * permutation of `Coord`, `long_t` and `real_t`. Binary operations with `Coord`
 * and `long_t` should return `long_t`, `long_t` and `real_t` should return
 * `real_t`, and `Coord` and `real_t` should return `real_t`.
 */
template<typename Coord> struct coord_ops {

    /**
     * Certain operations require multiplying two coordinate values and thus
     * need double the bits of the maximum coordinate value to avoid overflow.
     *
     * By default, if `Coord` is a 64 bit type, this is
     * `large_ints::compound_int`.
     *
     * This can be specialized as a user-defined type.
     */
    using long_t = large_ints::sized_int<sizeof(Coord)*2>;

    /**
     * The coordinates of generated points are usually not integers. By default,
     * they are represented by `double` before being rounded back to integers.
     * This type can be specialized to use some custom type for more precision.
     */
    using real_t = double;

    /**
     * Multiply `a` and `b` and return the result.
     *
     * This should be equivalent to
     * `static_cast<long_t>(a) * static_cast<long_t>(b)`, except
     * `long_t` is not required to support multiplication.
     */
    static long_t mul(Coord a,Coord b) {
        return large_ints::mul(a,b);
    }

    /** 
     * Default implementation:
     *
     *     return std::acos(static_cast<real_t>(x));
     */
    static real_t acos(Coord x) { return std::acos(static_cast<real_t>(x)); }

    /** 
     * Default implementation:
     *
     *     return static_cast<Coord>(std::acos(x));
     */
    static real_t acos(real_t x) { return std::acos(x); }

    /** 
     * Default implementation:
     *
     *     return static_cast<Coord>(std::cos(x));
     */
    static real_t cos(real_t x) { return std::cos(x); }

    /** 
     * Default implementation:
     *
     *     return static_cast<Coord>(std::sin(x));
     */
    static real_t sin(real_t x) { return std::sin(x); }

    /** 
     * Default implementation:
     *
     *     return std::sqrt(static_cast<real_t>(x));
     */
    static real_t sqrt(Coord x) { return std::sqrt(static_cast<real_t>(x)); }

    /** 
     * Default implementation:
     *
     *     return static_cast<Coord>(std::sqrt(x));
     */
    static real_t sqrt(real_t x) { return std::sqrt(x); }

    /** 
     * Default implementation:
     *
     *     return static_cast<Coord>(std::lround(x));
     */
    static Coord round(real_t x) { return static_cast<Coord>(std::lround(x)); }

    /** 
     * Default implementation:
     *
     *     return static_cast<Coord>(std::floor(x));
     */
    static Coord floor(real_t x) { return static_cast<Coord>(std::floor(x)); }

    /**
     * Default implementation:
     *
     *     return static_cast<Coord>(std::ceil(x));
     */
    static Coord ceil(real_t x) { return static_cast<Coord>(std::ceil(x)); }

    /**
     * After determining how many points to use to approximate an arc using
     * real numbers, the value needs to be converted to an integer to use in a
     * loop.
     *
     * Default implementation:
     *
     *     return static_cast<Coord>(x);
     */
    static Coord to_coord(real_t x) { return static_cast<Coord>(x); }

    /**
     * Return the value of pi.
     *
     * Default implementation:
     *
     *     return std::numbers::pi_v<real_t>;
     */
    static real_t pi() { return std::numbers::pi_v<real_t>; }
};

template<typename Coord> using long_coord_t = typename coord_ops<Coord>::long_t;
template<typename Coord> using real_coord_t = typename coord_ops<Coord>::real_t;

/**
 * A struct containing getters for point-like objects. Specializations exist for
 * `T[2]`, `std::span<T,2>` and `point_t<T>`. This can be specialized by the
 * user for other types. Static functions `get_x` and `get_y` should be defined
 * to get the X and Y coordinates respectively.
 *
 * Example:
 *
 *     template<> struct poly_ops::point_ops<MyPoint> {
 *         static int get_x(const MyPoint &p) {
 *             return static_cast<int>(MyPoint.X * 100.0f);
 *         }
 *         static int get_y(const MyPoint &p) {
 *             return static_cast<int>(MyPoint.Y * 100.0f);
 *         }
 *     };
 */
template<typename T> struct point_ops {};

template<typename T> struct point_ops<T[2]> {
    static constexpr const T &get_x(const T (&p)[2]) noexcept { return p[0]; }
    static constexpr const T &get_y(const T (&p)[2]) noexcept { return p[1]; }
};

template<typename T> struct point_ops<std::span<T,2>> {
    static constexpr const T &get_x(const std::span<T,2> &p) noexcept { return p[0]; }
    static constexpr const T &get_y(const std::span<T,2> &p) noexcept { return p[1]; }
};

namespace detail {
template<typename T> concept arithmetic = requires(T x) {
    { x + x } -> std::convertible_to<T>;
    { x - x } -> std::convertible_to<T>;
    { -x } -> std::convertible_to<T>;
};
template<typename Lesser,typename Greater>
concept arithmetic_promotes = requires(Lesser a,Greater b) {
    { a + b } -> std::convertible_to<Greater>;
    { b + a } -> std::convertible_to<Greater>;
    { a - b } -> std::convertible_to<Greater>;
    { b - a } -> std::convertible_to<Greater>;
};
} // namespace detail

/**
 * Defined as:
 *
 *     detail::arithmetic<T>
 *     && detail::arithmetic<long_coord_t<T>>
 *     && detail::arithmetic<real_coord_t<T>>
 *     && detail::arithmetic_promotes<T,long_coord_t<T>>
 *     && detail::arithmetic_promotes<T,real_coord_t<T>>
 *     && std::totally_ordered<T>
 *     && std::totally_ordered<long_coord_t<T>>
 *     && std::totally_ordered<real_coord_t<T>>
 *     && std::convertible_to<T,long_coord_t<T>>
 *     && std::convertible_to<int,long_coord_t<T>>
 *     && std::convertible_to<T,real_coord_t<T>>
 *     && requires(T c,long_coord_t<T> cl,real_coord_t<T> cr) {
 *         static_cast<long>(c);
 *         { coord_ops<T>::acos(c) } -> std::same_as<real_coord_t<T>>;
 *         { coord_ops<T>::acos(cr) } -> std::same_as<real_coord_t<T>>;
 *         { coord_ops<T>::cos(cr) } -> std::same_as<real_coord_t<T>>;
 *         { coord_ops<T>::sin(cr) } -> std::same_as<real_coord_t<T>>;
 *         { coord_ops<T>::sqrt(c) } -> std::same_as<real_coord_t<T>>;
 *         { coord_ops<T>::sqrt(cr) } -> std::same_as<real_coord_t<T>>;
 *         { coord_ops<T>::round(cr) } -> std::same_as<T>;
 *         { coord_ops<T>::ceil(cr) } -> std::same_as<T>;
 *         { coord_ops<T>::floor(cr) } -> std::same_as<T>;
 *         { coord_ops<T>::to_coord(cr) } -> std::same_as<T>;
 *         { coord_ops<T>::pi() } -> std::same_as<real_coord_t<T>>;
 *         { coord_ops<T>::mul(c,c) } -> std::same_as<long_coord_t<T>>;
 *         { cr * cr } -> std::convertible_to<real_coord_t<T>>;
 *         { cr / cr } -> std::convertible_to<real_coord_t<T>>;
 *         { c * cr } -> std::convertible_to<real_coord_t<T>>;
 *         { cr * c } -> std::convertible_to<real_coord_t<T>>;
 *     }
 *
 * where `detail::arithmetic<typename T>` is:
 *
 *     requires(T x) {
 *         { x + x } -> std::convertible_to<T>;
 *         { x - x } -> std::convertible_to<T>;
 *         { -x } -> std::convertible_to<T>;
 *     }
 * 
 * and `detail::arithmetic_promotes<typename Lesser,typename Greater>` is:
 *
 *     requires(Lesser a,Greater b) {
 *         { a + b } -> std::convertible_to<Greater>;
 *         { b + a } -> std::convertible_to<Greater>;
 *         { a - b } -> std::convertible_to<Greater>;
 *         { b - a } -> std::convertible_to<Greater>;
 *     }
 */
template<typename T> concept coordinate =
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
        { cr * cr } -> std::convertible_to<real_coord_t<T>>;
        { cr / cr } -> std::convertible_to<real_coord_t<T>>;
        { c * cr } -> std::convertible_to<real_coord_t<T>>;
        { cr * c } -> std::convertible_to<real_coord_t<T>>;
    };

/**
 * Defined as:
 *
 *     requires(const T &v) {
 *         { point_ops<T>::get_x(v) } -> std::convertible_to<Coord>;
 *         { point_ops<T>::get_y(v) } -> std::convertible_to<Coord>;
 *     }
 */
template<typename T,typename Coord> concept point_like = requires(const T &v) {
    { point_ops<T>::get_x(v) } -> std::convertible_to<Coord>;
    { point_ops<T>::get_y(v) } -> std::convertible_to<Coord>;
};

/**
 * A two-dimensional point or vector
 */
template<typename T> struct point_t {
    T _data[2];

    /** The default constructor leaves the values uninitialized */
    point_t() noexcept = default;

    constexpr point_t(const T &x,const T &y) noexcept(std::is_nothrow_copy_constructible_v<T>)
        : _data{x,y} {}
    constexpr point_t(const point_t &b) = default;

    /** Construct `point_t` from any point-like object */
    template<point_like<T> U> constexpr point_t(const U &b)
        noexcept(noexcept(point_t(point_ops<U>::get_x(b),point_ops<U>::get_y(b))))
        : _data{point_ops<U>::get_x(b),point_ops<U>::get_y(b)} {}
    
    constexpr point_t &operator=(const point_t &b) noexcept(std::is_nothrow_copy_constructible_v<T>) = default;

    constexpr T &operator[](std::size_t i) noexcept { return _data[i]; }
    constexpr const T &operator[](std::size_t i) const noexcept { return _data[i]; }

    constexpr T &x() noexcept { return _data[0]; }
    constexpr const T &x() const noexcept { return _data[0]; }
    constexpr T &y() noexcept { return _data[1]; }
    constexpr const T &y() const noexcept { return _data[1]; }

    constexpr T *begin() noexcept { return _data; }
    constexpr const T *begin() const noexcept { return _data; }
    constexpr T *end() noexcept { return _data+2; }
    constexpr const T *end() const noexcept { return _data+2; }

    /** Always returns 2 */
    constexpr std::size_t size() const noexcept { return 2; }

    /** Return a pointer to the underlying array */
    constexpr T *data() noexcept { return _data; }

    /** Return a const pointer to the underlying array */
    constexpr const T *data() const noexcept { return _data; }

    /** Element-wise addition. */
    constexpr point_t &operator+=(const point_t &b) {
        _data[0] += b[0];
        _data[1] += b[1];
        return *this;
    }

    /** Element-wise subtraction. */
    constexpr point_t &operator-=(const point_t &b) {
        _data[0] -= b[0];
        _data[1] -= b[1];
        return *this;
    }

    constexpr point_t &operator*=(T b) {
        _data[0] *= b;
        _data[1] *= b;
        return *this;
    }

    constexpr point_t operator-() const {
        return {-_data[0],-_data[1]};
    }

    friend constexpr void swap(point_t &a,point_t &b) noexcept(std::is_nothrow_swappable_v<T>) {
        using std::swap;
        swap(a._data[0],b._data[0]);
        swap(a._data[1],b._data[1]);
    }
};

template<typename T> struct point_ops<point_t<T>> {
    static constexpr const T &get_x(const point_t<T> &p) noexcept { return p[0]; }
    static constexpr const T &get_y(const point_t<T> &p) noexcept { return p[1]; }
};

/** Element-wise addition. */
template<typename T,typename U>
constexpr point_t<std::common_type_t<T,U>> operator+(const point_t<T> &a,const point_t<U> &b) {
    using Tr = std::common_type_t<T,U>;
    return {static_cast<Tr>(a[0]+b[0]),static_cast<Tr>(a[1]+b[1])};
}

/** Element-wise subtraction. */
template<typename T,typename U>
constexpr point_t<std::common_type_t<T,U>> operator-(const point_t<T> &a,const point_t<U> &b) {
    using Tr = std::common_type_t<T,U>;
    return {static_cast<Tr>(a[0]-b[0]),static_cast<Tr>(a[1]-b[1])};
}

/** Element-wise multiplication. */
template<typename T,typename U>
constexpr point_t<std::common_type_t<T,U>> operator*(const point_t<T> &a,const point_t<U> &b) {
    using Tr = std::common_type_t<T,U>;
    return {static_cast<Tr>(a[0]*b[0]),static_cast<Tr>(a[1]*b[1])};
}

/** Element-wise division. */
template<typename T,typename U>
constexpr point_t<std::common_type_t<T,U>> operator/(const point_t<T> &a,const point_t<U> &b) {
    using Tr = std::common_type_t<T,U>;
    return {static_cast<Tr>(a[0]/b[0]),static_cast<Tr>(a[1]/b[1])};
}

/** Multiply both elements of `a` by `b` and return the result */
template<typename T,typename U>
constexpr point_t<std::common_type_t<T,U>> operator*(const point_t<T> &a,U b) {
    using Tr = std::common_type_t<T,U>;
    return {static_cast<Tr>(a[0]*b),static_cast<Tr>(a[1]*b)};
}

/** Multiply both elements of `b` by `a` and return the result */
template<typename T,typename U>
constexpr point_t<std::common_type_t<T,U>> operator*(T a,const point_t<U> &b) {
    using Tr = std::common_type_t<T,U>;
    return {static_cast<Tr>(a*b[0]),static_cast<Tr>(a*b[1])};
}

template<typename T>
constexpr bool operator==(const point_t<T> &a,const point_t<T> &b) {
    return a[0] == b[0] && a[1] == b[1];
}
template<typename T>
constexpr bool operator!=(const point_t<T> &a,const point_t<T> &b) {
    return a[0] != b[0] || a[1] != b[1];
}

/**
 * A functor to provide STL containers an arbitrary but consistent order for
 * `point_t`
 */
struct point_less {
    template<typename T>
    constexpr bool operator()(const point_t<T> &a,const point_t<T> &b) const {
        return (a[0] == b[0]) ? (a[1] < b[1]) : (a[0] < b[0]);
    }
};

/** Return the dot product of `a` and `b` */
template<typename T,typename U> constexpr auto vdot(const point_t<T> &a,const point_t<U> &b) {
    return a[0]*b[0] + a[1]*b[1];
}

/** Equivalent to `a[0]*a[0] + a[1]*a[1]` */
template<typename T> constexpr T square(const point_t<T> &a) {
    return vdot(a,a);
}

/** Equivalent to `point_t{static_cast<T>(x[0]),static_cast<T>(x[1])}` */
template<typename T,typename U>
constexpr point_t<T> vcast(const point_t<U> &x) {
    return {static_cast<T>(x[0]),static_cast<T>(x[1])};
}

template<typename Coord>
constexpr point_t<Coord> vround(const point_t<real_coord_t<Coord>> &x) {
    return {coord_ops<Coord>::round(x[0]),coord_ops<Coord>::round(x[1])};
}

template<typename Coord,typename T>
constexpr real_coord_t<Coord> vmag(const point_t<T> &x) {
    return coord_ops<Coord>::sqrt(square(vcast<real_coord_t<Coord>>(x)));
}

template<typename Coord,typename T>
constexpr real_coord_t<T> vangle(const point_t<T> &a,const point_t<T> &b) {
    using real_t = real_coord_t<Coord>;

    auto ra = vcast<real_t>(a);
    auto rb = vcast<real_t>(b);
    real_t cosine = vdot(ra,rb)/(vmag<Coord>(ra)*vmag<Coord>(rb));

    /* this little error occurs frequently when doing open-line offsets */
    if(cosine > real_t(1)) cosine = real_t(1);
    else if(cosine < real_t(-1)) cosine = real_t(-1);

    return coord_ops<Coord>::acos(cosine);
}

/**
 * Returns a positive number if the triangle formed by the given points is
 * clockwise, negative if counter-clockwise and zero if degenerate
 */
template<typename Coord> constexpr long_coord_t<Coord> triangle_winding(
    const point_t<Coord> &p1,
    const point_t<Coord> &p2,
    const point_t<Coord> &p3)
{
    return coord_ops<Coord>::mul(p2[0]-p1[0],p3[1]-p1[1]) -
        coord_ops<Coord>::mul(p3[0]-p1[0],p2[1]-p1[1]);
}

/**
 * Returns a vector that is perpendicular to the vector `p2 - p1` with a
 * magnitude of `magnitude`.
 */
template<typename Coord,typename T>
constexpr point_t<real_coord_t<Coord>> perp_vector(
    const point_t<T> &p1,
    const point_t<T> &p2,
    real_coord_t<Coord> magnitude)
{
    point_t<real_coord_t<Coord>> perp(
        static_cast<real_coord_t<Coord>>(p2[1] - p1[1]),
        static_cast<real_coord_t<Coord>>(p1[0] - p2[0]));
    return perp * (magnitude/vmag<Coord>(perp));
}


/**
 * Defined as:
 *
 *     std::ranges::range<T> && point_like<std::ranges::range_value_t<T>,Coord>
 */
template<typename T,typename Coord> concept point_range
    = std::ranges::range<T> && point_like<std::ranges::range_value_t<T>,Coord>;

/**
 * Defined as:
 *
 *     std::ranges::range<T> && point_range<std::ranges::range_value_t<T>,Coord>
 */
template<typename T,typename Coord> concept point_range_range
    = std::ranges::range<T> && point_range<std::ranges::range_value_t<T>,Coord>;

/**
 * Defined as:
 *
 *     td::ranges::range<T> && (point_range<std::ranges::range_value_t<T>,Coord>
 *         || point_like<std::ranges::range_value_t<T>,Coord>)
 */
template<typename T,typename Coord> concept point_range_or_range_range
    = std::ranges::range<T> && (point_range<std::ranges::range_value_t<T>,Coord>
        || point_like<std::ranges::range_value_t<T>,Coord>);

template<coordinate Coord> class winding_dir_sink {
    point_t<Coord> first;
    point_t<Coord> prev;
    long_coord_t<Coord> r;

public:
    winding_dir_sink(point_t<Coord> first)
        : first{first}, prev{first}, r(0) {}
    
    void operator()(point_t<Coord> p) {
        r += coord_ops<Coord>::mul(prev[0]-p[0],prev[1]+p[1]);
        prev = p;
    }

    long_coord_t<Coord> close() {
        return r + coord_ops<Coord>::mul(prev[0]-first[0],prev[1]+first[1]);
    }
};

/**
 * Returns a positive number if mostly clockwise, negative if mostly
 * counter-clockwise and zero if degenerate or exactly half of the polygon's
 * area is inverted.
 *
 * This algorithm works on any polygon. For non-overlapping non-inverting
 * polygons, more efficient methods exist.
 */
template<coordinate Coord,point_range<Coord> Points>
long_coord_t<Coord> winding_dir(Points &&points) {
    auto itr = std::ranges::begin(points);
    auto end = std::ranges::end(points);
    if(itr == end) return 0;

    winding_dir_sink<Coord> sink{*itr};
    while(++itr != end) sink(*itr);
    return sink.close();
}

namespace detail {

template<typename Index> struct segment_common {
    Index a;
    Index b;

    segment_common() = default;
    segment_common(Index a,Index b) noexcept : a{a}, b{b} {}

    /* returns true if point 'a' comes before point 'b' in the loop */
    template<typename T> bool a_is_main(const T &points) const {
        POLY_OPS_ASSERT(points[a].next == b || points[b].next == a);
        return points[a].next == b;
    }

    friend bool operator==(const segment_common &a,const segment_common &b) {
        return a.a == b.a && a.b == b.b;
    }

protected:
    ~segment_common() = default;
};

template<typename Index> struct segment : segment_common<Index> {
    segment() = default;
    segment(Index a,Index b) noexcept : segment_common<Index>{a,b} {}
    segment(const segment_common<Index> &s) noexcept : segment_common<Index>(s) {}

    auto a_x(const auto &points) const { return points[this->a].data[0]; }
    auto a_y(const auto &points) const { return points[this->a].data[1]; }
    auto b_x(const auto &points) const { return points[this->b].data[0]; }
    auto b_y(const auto &points) const { return points[this->b].data[1]; }
};

template<typename Index,typename Coord> struct cached_segment : segment_common<Index> {
    point_t<Coord> pa;
    point_t<Coord> pb;

    cached_segment() = default;
    cached_segment(Index a,Index b,const point_t<Coord> &pa,const point_t<Coord> &pb)
        noexcept(std::is_nothrow_copy_constructible_v<point_t<Coord>>)
        : segment_common<Index>{a,b}, pa{pa}, pb{pb} {}
    template<typename T> cached_segment(const segment_common<Index> &s,const T &points)
        : segment_common<Index>{s}, pa{points[s.a].data}, pb{points[s.b].data} {}
    template<typename T> cached_segment(Index a,Index b,const T &points)
        : segment_common<Index>{a,b}, pa{points[a].data}, pb{points[b].data} {}
    cached_segment(const cached_segment&) = default;
    cached_segment &operator=(const cached_segment&) = default;
};

} // namespace detail

} // namespace poly_ops

#endif
