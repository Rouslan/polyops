#ifndef POLY_OPS_BASE_HPP
#define POLY_OPS_BASE_HPP

#include <cmath>
#include <numbers>
#include <utility>
#include <type_traits>
#include <span>
#include <ranges>


namespace poly_ops {

/** Mathematical operations on coordinate types. This struct can be specialized
by users of this library. Arithmetic operators should be defined for every
permutation of Coord, long_t and real_t. Binary operations with Coord and long_t
should return long_t, long_t and real_t should return real_t, and Coord and
real_t should return real_t. */
template<typename Coord> struct coord_ops {
    /** Certain operations require double the bits of the maximum coordinate
    value to avoid overflow. This type can be specialized if the default "long"
    type is not wide enough for a given coordinate type. */
    using long_t = long;

    /** The coordinates of generated points are usually real numbers. By
    default, they are represented by double before being rounded back to
    integers. This type can be specialized to use float instead, or some custom
    type for more precision. */
    using real_t = double;

    /** Functions need to be defined that are equivalent to the following
    functions from the "std" namespace: */
    static real_t acos(Coord x) { return std::acos(static_cast<real_t>(x)); }
    static real_t acos(real_t x) { return std::acos(x); }
    static real_t cos(real_t x) { return std::cos(x); }
    static real_t sin(real_t x) { return std::sin(x); }
    static real_t sqrt(Coord x) { return std::sqrt(static_cast<real_t>(x)); }
    static real_t sqrt(real_t x) { return std::sqrt(x); }

    /** These are used to convert real number coordinates to the normal type */
    static Coord round(real_t x) { return static_cast<Coord>(std::lround(x)); }
    static Coord floor(real_t x) { return static_cast<Coord>(std::floor(x)); }
    static Coord ceil(real_t x) { return static_cast<Coord>(std::ceil(x)); }

    /** After dermining how many points to use to approximate an arc using real
    numbers, the value needs to be converted to an integer to use in a loop */
    static Coord to_coord(real_t x) { return static_cast<Coord>(x); }

    /** Return a value with the same sign as "x" but with a magnitude of 1 */
    static real_t unit(real_t x) { return std::copysign(1.0,x); }

    static real_t pi() { return std::numbers::pi_v<real_t>; }
};

template<typename Coord> using long_coord_t = typename coord_ops<Coord>::long_t;
template<typename Coord> using real_coord_t = typename coord_ops<Coord>::real_t;

/* Getters for point-like objects. This can be specialized by the user for other
types. Static functions "get_x" and "get_y" should be defined to get the X and Y
coordinates respectively. */
template<typename T> struct point_ops {};

template<typename T> struct point_ops<T[2]> {
    static constexpr const T &get_x(const T (&p)[2]) { return p[0]; }
    static constexpr const T &get_y(const T (&p)[2]) { return p[1]; }
};

template<typename T> struct point_ops<std::span<T,2>> {
    static constexpr const T &get_x(const std::span<T,2> &p) { return p[0]; }
    static constexpr const T &get_y(const std::span<T,2> &p) { return p[1]; }
};

namespace detail {
template<typename T> concept arithmetic = requires(T x) {
    { x + x } -> std::same_as<T>;
    { x - x } -> std::same_as<T>;
    { x * x } -> std::same_as<T>;
    { x / x } -> std::same_as<T>;
    { -x } -> std::same_as<T>;
};
template<typename Lesser,typename Greater>
concept arithmetic_promotes = requires(Lesser a,Greater b) {
    { a + b } -> std::same_as<Greater>;
    { b + a } -> std::same_as<Greater>;
    { a - b } -> std::same_as<Greater>;
    { b - a } -> std::same_as<Greater>;
    { a * b } -> std::same_as<Greater>;
    { b * a } -> std::same_as<Greater>;
    { a / b } -> std::same_as<Greater>;
    { b / a } -> std::same_as<Greater>;
};
} // namespace detail

template<typename T> concept coordinate =
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
    };


template<typename T,typename Coord> concept point = requires(const T &v) {
    { point_ops<T>::get_x(v) } -> std::convertible_to<Coord>;
    { point_ops<T>::get_y(v) } -> std::convertible_to<Coord>;
};

template<typename T> struct point_t {
    T _data[2];

    point_t() = default;
    constexpr point_t(const T &x,const T &y) {
        _data[0] = x;
        _data[1] = y;
    }
    template<point<T> U> constexpr point_t(const U &b) {
        _data[0] = point_ops<U>::get_x(b);
        _data[1] = point_ops<U>::get_y(b);
    }

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

    constexpr std::size_t size() const noexcept { return 2; }
    constexpr T *data() noexcept { return _data; }
    constexpr const T *data() const noexcept { return _data; }

    constexpr point_t &operator+=(const point_t &b) {
        _data[0] += b[0];
        _data[1] += b[1];
        return *this;
    }

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
    static constexpr const T &get_x(const point_t<T> &p) { return p[0]; }
    static constexpr const T &get_y(const point_t<T> &p) { return p[1]; }
};

template<typename T,typename U>
constexpr point_t<decltype(std::declval<T>()+std::declval<U>())> operator+(const point_t<T> &a,const point_t<U> &b) {
    return {a[0]+b[0],a[1]+b[1]};
}

template<typename T,typename U>
constexpr point_t<decltype(std::declval<T>()-std::declval<U>())> operator-(const point_t<T> &a,const point_t<U> &b) {
    return {a[0]-b[0],a[1]-b[1]};
}

template<typename T,typename U>
constexpr point_t<decltype(std::declval<T>()*std::declval<U>())> operator*(const point_t<T> &a,const point_t<U> &b) {
    return {a[0]*b[0],a[1]*b[1]};
}

template<typename T,typename U>
constexpr point_t<decltype(std::declval<T>()/std::declval<U>())> operator/(const point_t<T> &a,const point_t<U> &b) {
    return {a[0]/b[0],a[1]/b[1]};
}

template<typename T,typename U>
constexpr point_t<decltype(std::declval<T>()*std::declval<U>())> operator*(const point_t<T> &a,U b) {
    return {a[0]*b,a[1]*b};
}
template<typename T,typename U>
constexpr point_t<decltype(std::declval<T>()*std::declval<U>())> operator*(T a,const point_t<U> &b) {
    return {a*b[0],a*b[1]};
}

template<typename T>
constexpr bool operator==(const point_t<T> &a,const point_t<T> &b) {
    return a[0] == b[0] && a[1] == b[1];
}
template<typename T>
constexpr bool operator!=(const point_t<T> &a,const point_t<T> &b) {
    return a[0] != b[0] || a[1] != b[1];
}

/* A functor to provide STL containers an arbitrary but consistent order for
point_t */
struct point_less {
    template<typename T>
    constexpr bool operator()(const point_t<T> &a,const point_t<T> &b) const {
        return (a[0] == b[0]) ? (a[1] < b[1]) : (a[0] < b[0]);
    }
};

template<typename T,typename U> constexpr auto vdot(const point_t<T> &a,const point_t<U> &b) {
    return a[0]*b[0] + a[1]*b[1];
}

template<typename T> constexpr T square(const point_t<T> &a) {
    return vdot(a,a);
}

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
    auto ra = vcast<real_coord_t<Coord>>(a);
    auto rb = vcast<real_coord_t<Coord>>(b);
    return coord_ops<Coord>::acos(vdot(ra,rb)/(vmag<Coord>(ra)*vmag<Coord>(rb)));
}

/* Returns a positive number if clockwise, negative if counter-clockwise and
zero if degenerate */
template<typename Coord> constexpr long_coord_t<Coord> triangle_winding(
    const point_t<Coord> &p1,
    const point_t<Coord> &p2,
    const point_t<Coord> &p3)
{
    return static_cast<long_coord_t<Coord>>(p3[0]-p1[0])*(p2[1]-p1[1]) -
        static_cast<long_coord_t<Coord>>(p2[0]-p1[0])*(p3[1]-p1[1]);
}

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


template<typename T,typename Coord> concept point_range
    = std::ranges::range<T> && point<std::ranges::range_value_t<T>,Coord>;

template<typename T,typename Coord> concept point_range_range
    = std::ranges::sized_range<T> && point_range<std::ranges::range_value_t<T>,Coord>;

/** Returns a positive number if clockwise, negative if counter-clockwise and
zero if degenerate or exactly half of the polygon's area is inverted.

This algorithm works on any polygon. For non-overlapping non-inverting polygons,
more efficient methods exist. The magnitude of the return value is two times the
area of the polygon.*/
template<coordinate Coord,point_range<Coord> Points>
long_coord_t<Coord> winding_dir(Points &&points)
{
    auto itr = std::ranges::begin(points);
    auto end = std::ranges::end(points);
    if(itr == end) return 0;
    point_t first(*itr);
    point_t last=first;
    Coord r = 0;
    while( ++itr != end) {
        point_t p(*itr);
        r += static_cast<long_coord_t<Coord>>(last[0]-p[0]) * (last[1]+p[1]);
        last = p;
    }
    return r + static_cast<long_coord_t<Coord>>(last[0]-first[0]) * (last[1]+first[1]);
}

} // namespace poly_ops

#endif
