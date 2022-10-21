/*
The process to create the inset/outset shapes is the following:

Given a shape consisting of polygons where the points of the outer wall are
oriented clockwise and the holes are oriented counter-clockwise:
 ________________
 \              /
  \            /
   \          /
    \        /
     \      /
      \    /
       \  /
        \/

The lines are moved by the inset/outset magnitude.
  \            /
___\__________/___
    \        /
     \      /
      \    /
       \  /
        \/
        /\
       /  \

The ends of the lines are reconnected. For outer angles in the original polygon,
greater than 180 degrees, the ends are joined with a single line. For outer
angles less than 180 degrees, an arc, whose center is the original point, is
approximated with multiple lines.

Intersecting lines are then broken so that only end-points touch.
 /\            /\
/__\__________/__\
    \        /
     \      /
      \    /
       \  /
        \/
        /\
       /__\

The "line balance" of each line is computed. This number is 0 if the line is
part of normal geometry, a negative number if it's part of inverted geometry,
and a positive number if it's part of nested geometry (neither holes nor lines
inside holes are considered nested).
 -1/\          -1/\
  /__\__________/__\
      \        /
       \      /
       0\    /
         \  /
          \/
          /\
       -1/__\

All lines that have a non-zero "line balance" are removed.
    __________
    \        /
     \      /
      \    /
       \  /
        \/

*/

#ifndef POLY_OPS_HPP
#define POLY_OPS_HPP

#include <stdint.h>
#include <vector>
#include <set>
#include <queue>
#include <map>
#include <tuple>
#include <algorithm>
#include <limits>
#include <cmath>
#include <numbers>
#include <cstddef>
#include <utility>
#include <memory_resource>
#include <iterator>
#include <ranges>
#include <span>

#include "mini_flat_set.hpp"

#ifndef POLY_OPS_GRAPHICAL_DEBUG
#define POLY_OPS_GRAPHICAL_DEBUG 0
#endif

#ifndef POLY_OPS_ASSERT
#include <assert.h>
#define POLY_OPS_ASSERT assert

// used for checks that would drastically slow down the algorithm
#define POLY_OPS_ASSERT_SLOW(X) (void)0
#endif

// the graphical test program defines these (in graphical_test.cpp)
#ifndef DEBUG_STEP_BY_STEP_EVENT_F
#define DEBUG_STEP_BY_STEP_EVENT_F (void)0
#define DEBUG_STEP_BY_STEP_EVENT_B (void)0
#define DEBUG_STEP_BY_STEP_EVENT_CALC_BALANCE (void)0
#define DEBUG_STEP_BY_STEP_LB_CHECK_RET_TYPE void
#define DEBUG_STEP_BY_STEP_LB_CHECK_RETURN(A,B) (void)0
#define DEBUG_STEP_BY_STEP_LB_CHECK_FF
#define DEBUG_STEP_BY_STEP_MISSED_INTR (void)0
#endif

namespace poly_ops {

/* Mathematical operations on coordinate types. This struct can be specialized
for user-defined types. Arithmetic operators should be defined for every
permutation of Coord, long_t and real_t. Binary operations with Coord and long_t
should return long_t, long_t and real_t should return real_t, and Coord and
real_t should return real_t. */
template<typename Coord> struct coord_ops {
    /* Certain operations require double the bits of the maximum coordinate
    value to avoid overflow. This type can be specialized if the default "long"
    type is not wide enough for a given coordinate type. */
    using long_t = long;

    /* The coordinates of generated points are usually real numbers. By default,
    they are represented by double before being rounded back to integers. This
    type can be specialized to use float instead, or some custom type for more
    precision. */
    using real_t = double;

    /* For user-defined real number types, functions need to be defined that are
    equivalent to the following functions from the "std" namespace: */
    template<typename T> static real_t acos(T x) { return std::acos(static_cast<real_t>(x)); }
    static real_t cos(real_t x) { return std::cos(x); }
    static real_t sin(real_t x) { return std::sin(x); }
    template<typename T> static real_t sqrt(T x) { return std::sqrt(static_cast<real_t>(x)); }

    /* This is used to convert real number coordinates to the normal type */
    static Coord round(real_t x) { return static_cast<Coord>(std::lround(x)); }

    /* After dermining how many points to use to approximate an arc, using real
    numbers, the value needs to be converted to an integer to use in a loop */
    static Coord floor(real_t x) { return static_cast<Coord>(x); }

    static real_t pi() { return std::numbers::pi_v<real_t>; }
};

template<typename Coord> using long_coord_t = coord_ops<Coord>::long_t;
template<typename Coord> using real_coord_t = coord_ops<Coord>::real_t;

/* Getters for point-like objects. This can be specialized for any user-defined
types. Static functions "get_x" and "get_y" should be defined to get the X and Y
coordinates respectively. */
template<typename T> struct point_ops {};

template<typename T> struct point_ops<std::span<T,2>> {
    static constexpr const T &get_x(const std::span<T,2> &p) { return p[0]; }
    static constexpr const T &get_y(const std::span<T,2> &p) { return p[1]; }
};

template<typename T> struct point_ops<T[2]> {
    static constexpr const T &get_x(const T (&p)[2]) { return p[0]; }
    static constexpr const T &get_y(const T (&p)[2]) { return p[1]; }
};

template<typename T> struct point_ops<std::tuple<T,T>> {
    static constexpr const T &get_x(const std::tuple<T,T> &p) { return std::get<0>(p); }
    static constexpr const T &get_y(const std::tuple<T,T> &p) { return std::get<1>(p); }
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

template<typename T> concept coordinate =
    arithmetic<T>
    && arithmetic<typename coord_ops<T>::long_t>
    && arithmetic<real_coord_t<T>>
    && arithmetic_promotes<T,long_coord_t<T>>
    && arithmetic_promotes<T,real_coord_t<T>>
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
        { coord_ops<T>::floor(cr) } -> std::same_as<T>;
        { coord_ops<T>::pi() } -> std::same_as<real_coord_t<T>>;
    };
} // namespace detail

template<typename T,typename Coord> concept point = requires(const T &v) {
    { point_ops<T>::get_x(v) } -> std::convertible_to<Coord>;
    { point_ops<T>::get_y(v) } -> std::convertible_to<Coord>;
};

template<typename T,typename Coord> concept point_range
    = std::ranges::range<T> && point<std::ranges::range_value_t<T>,Coord>;

template<typename T,typename Coord> concept point_range_range
    = std::ranges::range<T> && point_range<std::ranges::range_value_t<T>,Coord>;

template<typename T,typename Coord> concept polygon = requires(const T &v) {
    { v.outer_loop() } -> point_range<Coord>;
    { v.inner_loops() } -> point_range_range<Coord>;
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

    constexpr T &operator[](size_t i) noexcept { return _data[i]; }
    constexpr const T &operator[](size_t i) const noexcept { return _data[i]; }

    constexpr T &x() noexcept { return _data[0]; }
    constexpr const T &x() const noexcept { return _data[0]; }
    constexpr T &y() noexcept { return _data[1]; }
    constexpr const T &y() const noexcept { return _data[1]; }

    constexpr T *begin() noexcept { return _data; }
    constexpr const T *begin() const noexcept { return _data; }
    constexpr T *end() noexcept { return _data+2; }
    constexpr const T *end() const noexcept { return _data+2; }

    constexpr size_t size() const noexcept { return 2; }
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

template<typename T> constexpr T vdot(const point_t<T> &a,const point_t<T> &b) {
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
    using long_t = long_coord_t<Coord>;
    return static_cast<long_t>(p2[0]-p1[0]) * (p2[1]+p1[1]) +
        static_cast<long_t>(p3[0]-p2[0]) * (p3[1]+p2[1]) +
        static_cast<long_t>(p1[0]-p3[0]) * (p1[1]+p3[1]);
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

namespace detail {
template<typename Index> using original_sets_t = mini_set_proxy_vector<Index,std::pmr::polymorphic_allocator<Index>>;

template<typename Index,typename Coord> struct loop_point {
    point_t<Coord> data;
    Index original_set;
    Index next;

    static const int UNDEF_LINE_BAL = std::numeric_limits<int>::lowest();

    /* Represents a point that needs to be removed.

    This is not a unique value; any value besides zero will be removed. */
    static const int REMOVE_LINE_BAL = 1;

    int line_bal;

    loop_point() = default;
    loop_point(point_t<Coord> data,Index original_set,Index next,int line_bal=UNDEF_LINE_BAL) :
        data{data}, original_set{original_set}, next{next}, line_bal{line_bal} {}

    friend void swap(loop_point &a,loop_point &b) {
        using std::swap;

        swap(a.data,b.data);
        swap(a.original_set,b.original_set);
        swap(a.next,b.next);
        swap(a.line_bal,b.line_bal);
    }
};
template<typename Index,typename Coord> const int loop_point<Index,Coord>::UNDEF_LINE_BAL;
template<typename Index,typename Coord> const int loop_point<Index,Coord>::REMOVE_LINE_BAL;

template<typename Index> struct segment {
    Index a;
    Index b;

    template<typename T> auto a_x(const T &points) const { return points[a].data[0]; }
    template<typename T> auto a_y(const T &points) const { return points[a].data[1]; }
    template<typename T> auto b_x(const T &points) const { return points[b].data[0]; }
    template<typename T> auto b_y(const T &points) const { return points[b].data[1]; }

    /* returns true if point 'a' comes before point 'b' in the loop */
    template<typename T> bool a_is_main(const T &points) const {
        POLY_OPS_ASSERT(points[a].next == b || points[b].next == a);
        return points[a].next == b;
    }

    friend bool operator==(segment a,segment b) {
        return a.a == b.a && a.b == b.b;
    }
};

/* The order of these determines their priority from greatest to least.

"vforward" and "vbackward" have the same meaning as "forward" and "backward" but
have different priorities. They are used for vertical lines, which, unlike
non-vertical lines, can intersect with lines that start and end at the same X
coordinate as the start/end of the vertical line.

The comparison functor "sweep_cmp" requires that "backward" come before
"forward". */
enum class event_type_t {vforward,backward,forward,vbackward,calc_balance};

template<typename Index> struct event {
    segment<Index> ab;
    event_type_t type;
    segment<Index> line_ba() const { return {ab.b,ab.a}; }
};

// This refers to a 1-dimensional edge, i.e. the start or end of a line segment
enum at_edge_t {
    EDGE_START = -1,
    EDGE_NO = 0,
    EDGE_END = 1};

template<typename Coord> at_edge_t is_edge(Coord val,Coord end_val) {
    if(val == 0) return EDGE_START;
    if(val == end_val) return EDGE_END;
    return EDGE_NO;
}

/* Check if two line segments intersect.

Returns "true" if the lines intersect at a single point, and that point is not
an endpoint for at least one of the lines. */
template<typename Index,typename Coord>
bool intersects(segment<Index> s1,segment<Index> s2,const loop_point<Index,Coord> *points,point_t<Coord> &p,std::span<at_edge_t,2> at_edge) {
    using long_t = long_coord_t<Coord>;

    Coord x1 = s1.a_x(points);
    Coord y1 = s1.a_y(points);
    Coord x2 = s1.b_x(points);
    Coord y2 = s1.b_y(points);
    Coord x3 = s2.a_x(points);
    Coord y3 = s2.a_y(points);
    Coord x4 = s2.b_x(points);
    Coord y4 = s2.b_y(points);

    long_t d = static_cast<long_t>(x1-x2)*(y3-y4) - static_cast<long_t>(y1-y2)*(x3-x4);
    if(d == 0) return false;

    // connected lines do not count as intersecting
    if(s1.a == s2.a || s1.a == s2.b || s1.b == s2.a || s1.b == s2.b) return false;

    long_t t_i = static_cast<long_t>(x1-x3)*(y3-y4) - static_cast<long_t>(y1-y3)*(x3-x4);
    long_t u_i = static_cast<long_t>(x1-x3)*(y1-y2) - static_cast<long_t>(y1-y3)*(x1-x2);

    if(d > 0) {
        if(t_i < 0 || t_i > d || u_i < 0 || u_i > d) return false;
    } else if(t_i > 0 || t_i < d || u_i > 0 || u_i < d) return false;

    if(t_i == 0) {
        at_edge[0] = EDGE_START;
        at_edge[1] = is_edge(u_i,d);
        p[0] = x1;
        p[1] = y1;
    } else if(t_i == d) {
        at_edge[0] = EDGE_END;
        at_edge[1] = is_edge(u_i,d);
        p[0] = x2;
        p[1] = y2;
    } else if(u_i == 0) {
        at_edge[0] = EDGE_NO;
        at_edge[1] = EDGE_START;
        p[0] = x3;
        p[1] = y3;
    } else if(u_i == d) {
        at_edge[0] = EDGE_NO;
        at_edge[1] = EDGE_END;
        p[0] = x4;
        p[1] = y4;
    } else {
        auto t = static_cast<real_coord_t<Coord>>(t_i)/d;

        p[0] = x1 + coord_ops<Coord>::round(t * (x2-x1));
        p[1] = y1 + coord_ops<Coord>::round(t * (y2-y1));

        at_edge[0] = EDGE_NO;
        at_edge[1] = EDGE_NO;

        /* the point might still fall on one of the ends after rounding back to
        integer coordinates */
        if(p[0] == x1 && p[1] == y1) at_edge[0] = EDGE_START;
        else if(p[0] == x2 && p[1] == y2) at_edge[0] = EDGE_END;

        if(p[0] == x3 && p[1] == y3) at_edge[1] = EDGE_START;
        else if(p[0] == x4 && p[1] == y4) at_edge[1] = EDGE_END;
    }

    return at_edge[0] == EDGE_NO || at_edge[1] == EDGE_NO;
}

template<typename Coord>
bool lower_angle(point_t<Coord> ds,point_t<Coord> dp,bool tie_breaker) {
    auto v = static_cast<long_coord_t<Coord>>(ds.y())*dp.x();
    auto w = static_cast<long_coord_t<Coord>>(dp.y())*ds.x();
    return v < w || (v == w && tie_breaker);
}

template<typename Coord>
bool vert_overlap(point_t<Coord> sa,point_t<Coord> sb,point_t<Coord> p,point_t<Coord> dp,bool tie_breaker) {
    Coord ya = sa.y();
    Coord yb = sb.y();
    if(ya > yb) std::swap(ya,yb);
    return
        tie_breaker &&
        dp.x() == 0 &&
        (p.y() == (dp.y() > 0 ? ya : yb));
}

/* Determine if a ray at point 'p' intersects with line segment 'sa-sb'. The ray
is vertical and extends toward negative infinity.

'sa-sb' must not be a vertical line.

If 'p' has the same X coordinate as one of the end-points of 'sa-sb', it does
not intersect if the X coordinate of the other end-point of 'sa-sb' is smaller
than the X coordinate of 'p' and 'hsign' is positive, or if the X coordinate is
greater and 'hsign' is negative, or if the X coordinates are the same. A
vertical line will never intersect.

If 'p' has the same coordinates as one of the end-points of 'sa-sb', it does not
intersect if the Y coordinate of the other end-point of 'sa-sb' is greater than
or equal to the Y coordinate of 'p'. */
template<typename Coord> bool line_segment_up_ray_intersection(
    point_t<Coord> sa,
    point_t<Coord> sb,
    point_t<Coord> p,
    Coord hsign,
    point_t<Coord> dp,
    bool tie_breaker)
{
    using long_t = long_coord_t<Coord>;

    POLY_OPS_ASSERT(sa.x() <= sb.x());

    point_t<Coord> ds = sb - sa;
    if(ds.x() == 0) return vert_overlap(sa,sb,p,dp,tie_breaker);

    if(p.x() < sa.x() || (p.x() == sa.x() && hsign < 0)) return false;
    if(p.x() > sb.x() || (p.x() == sb.x() && hsign > 0)) return false;

    long_t t = static_cast<long_t>(p.x()-sa.x())*ds.y();
    long_t u = static_cast<long_t>(p.y()-sa.y())*ds.x();
    return t < u || (t == u && lower_angle(ds,dp,tie_breaker));
}

/* Determine if a ray at point 'p' intersects with line segment 'sa-sb'. The ray
is vertical and extends toward negative infinity.

This actually performs tests on two line segments. One is p to (p + dp1) and
the other is p to (p + dp2). Otherwise this function is the same as
line_segment_up_ray_intersection. */
template<typename Coord> std::tuple<bool,bool> dual_line_segment_up_ray_intersection(
    point_t<Coord> sa,
    point_t<Coord> sb,
    point_t<Coord> p,
    Coord hsign1,
    point_t<Coord> dp1,
    bool tie_breaker1,
    Coord hsign2,
    point_t<Coord> dp2,
    bool tie_breaker2)
{
    using long_t = long_coord_t<Coord>;

    POLY_OPS_ASSERT(sa.x() <= sb.x());

    point_t<Coord> ds = sb - sa;
    if(ds.x() == 0) return {
        vert_overlap(sa,sb,p,dp1,tie_breaker1),
        vert_overlap(sa,sb,p,dp2,tie_breaker2)};

    if(p.x() < sa.x() || p.x() > sb.x()) return {false,false};

    std::tuple r(true,true);
    if(p.x() == sa.x()) r = {hsign1 > 0,hsign2 > 0};
    else if(p.x() == sb.x()) r = {hsign1 < 0,hsign2 < 0};

    long_t t = static_cast<long_t>(p.x()-sa.x())*ds.y();
    long_t u = static_cast<long_t>(p.y()-sa.y())*ds.x();
    if(t > u) return {false,false};
    if(t == u) {
        std::get<0>(r) = std::get<0>(r) && lower_angle(ds,dp1,tie_breaker1);
        std::get<1>(r) = std::get<1>(r) && lower_angle(ds,dp2,tie_breaker2);
    }

    return r;
}

/* Calculate the "line balance".

The line balance is based on the idea of a winding number, except it
applies the lines instead of points inside. A positive number means nested
geometry, negative means inverted geometry, and zero means normal geometry
(clockwise oriented polygons and counter-clockwise holes). */
template<typename Index,typename Coord> class line_balance {
    /* Polygons are oriented clockwise, and holes counter-clockwise, thus for
    polygons, for every point on every line going right (towards positive
    infinity), there exists a point on the same polygon, with the same x value,
    on the top (towards negative infinity) side, which is part of a line going
    up. For holes, the point is on the bottom side instead, but proper holes
    should also have a corresponding point on the top that is part of a
    polygon. To compute the balance, we start with  a value of -1 for lines
    going left and 0 for lines going right. We pick a point on our line and then
    trace a ray going up. For every line (other than our starting line) that
    crosses this ray, if the line goes right, we add 1 to our value, and if the
    line goes left, we subtract 1.

    When the ray that we trace, intersects a point with lines extending in both
    left and right, we must not count this as two intersections, thus the
    'line_segment_up_ray_intersection' function takes a 'hsign' paramter. When
    a given line A̅B̅ intersects with the ray at point A, it only counts if
    'Bx - Ax' has the same sign as 'hsign'.

    Choosing 'hsign' is also important. It should be negative if our line points
    left (meaning the immediate next point has a lower x coordinate value) and
    positive if our line points right. This way, if the line immediately before
    (if our line is B̅C̅, the line before would be A̅B̅), is above and points in the
    opposite direction, it intersects with the ray. This will give us the same
    result as if the ray extended from the middle of our line, instead of the
    start.

    If our line is vertical, 'hsign' should be negative if our line points up,
    and positive if it points down (lines of zero length should have been
    eliminated in a proir step). */

    const loop_point<Index,Coord> *points;
    Index p1, p2;
    point_t<Coord> dp1, dp2;
    Coord hsign1, hsign2;
    int wn1, wn2;

public:
    line_balance(const loop_point<Index,Coord> *points,Index p1,Index p2) :
        points(points), p1(p1), p2(p2),
        dp1(points[points[p1].next].data - points[p1].data),
        dp2(points[points[p2].next].data - points[p2].data),
        hsign1(dp1.x() ? dp1.x() : dp1.y()), hsign2(dp2.x() ? dp2.x() : dp2.y()),
        wn1(dp1.x() < 0 ? -1 : 0), wn2(dp2.x() < 0 ? -1 : 0)
    {
        POLY_OPS_ASSERT(hsign1 && hsign2 && p1 != p2 && points[p1].data == points[p2].data);
    }

    DEBUG_STEP_BY_STEP_LB_CHECK_RET_TYPE check(segment<Index> s) {
        POLY_OPS_ASSERT(s.a_x(points) <= s.b_x(points));
        //if(s.a_x(points) == s.b_x(points)) return DEBUG_STEP_BY_STEP_LB_CHECK_FF;

        bool a_is_main = s.a_is_main(points);

        // is s the line at p1?
        if((a_is_main ? s.a : s.b) == p1) {
            if(line_segment_up_ray_intersection(
                points[s.a].data,
                points[s.b].data,
                points[p1].data,
                hsign2,
                dp2,
                (a_is_main ? s.a : s.b) > p2))
            {
                wn2 += a_is_main ? 1 : -1;
                DEBUG_STEP_BY_STEP_LB_CHECK_RETURN(false,true);
            }
            return DEBUG_STEP_BY_STEP_LB_CHECK_FF;
        }

        // is s the line at p2?
        if((a_is_main ? s.a : s.b) == p2) {
            if(line_segment_up_ray_intersection(
                points[s.a].data,
                points[s.b].data,
                points[p1].data,
                hsign1,
                dp1,
                (a_is_main ? s.a : s.b) > p1))
            {
                wn1 += a_is_main ? 1 : -1;
                DEBUG_STEP_BY_STEP_LB_CHECK_RETURN(true,false);
            }
            return DEBUG_STEP_BY_STEP_LB_CHECK_FF;
        }

        auto [intr1,intr2] = dual_line_segment_up_ray_intersection(
            points[s.a].data,
            points[s.b].data,
            points[p1].data,
            hsign1,
            dp1,
            (a_is_main ? s.a : s.b) > p1,
            hsign2,
            dp2,
            (a_is_main ? s.a : s.b) > p2);
        if(intr1) wn1 += a_is_main ? 1 : -1;
        if(intr2) wn2 += a_is_main ? 1 : -1;

        POLY_OPS_ASSERT_SLOW(
            intr1 == line_segment_up_ray_intersection(
                points[s.a].data,points[s.b].data,points[p1].data,hsign1,dp1,(a_is_main ? s.a : s.b) > p1));
        POLY_OPS_ASSERT_SLOW(
            intr2 == line_segment_up_ray_intersection(
                points[s.a].data,points[s.b].data,points[p1].data,hsign2,dp2,(a_is_main ? s.a : s.b) > p2));
        DEBUG_STEP_BY_STEP_LB_CHECK_RETURN(intr1,intr2);
    }

    std::tuple<int,int> result() const { return {wn1,wn2}; }
};

/* order events by the X coordinates increasing and then by event type */
template<typename Index,typename Coord> struct event_cmp {
    std::pmr::vector<loop_point<Index,Coord>> &lpoints;

    bool operator()(const event<Index> &l1,const event<Index> &l2) const {
        auto r = l1.ab.a_x(lpoints) - l2.ab.a_x(lpoints);
        if(r) return r > 0;

        return l1.type > l2.type;
    }
};

template<typename Index,typename Coord>
using events_t = std::priority_queue<event<Index>,std::pmr::vector<event<Index>>,event_cmp<Index,Coord>>;

template<typename Index,typename Coord>
void add_event(events_t<Index,Coord> &events,Index sa,Index sb,event_type_t t) {
    events.emplace(segment<Index>{sa,sb},t);
}
template<typename Index>
void add_event(std::pmr::vector<event<Index>> &events,Index sa,Index sb,event_type_t t) {
    events.emplace_back(segment<Index>{sa,sb},t);
}

template<typename Index,typename Coord,typename Events>
void add_fb_events(std::pmr::vector<loop_point<Index,Coord>> &points,Events &events,Index sa,Index sb) {
    auto f = event_type_t::forward;
    auto b = event_type_t::backward;
    if(points[sa].data[0] == points[sb].data[0]) {
        f = event_type_t::vforward;
        b = event_type_t::vbackward;
    }
    add_event(events,sa,sb,f);
    add_event(events,sb,sa,b);
}

/* Line segments are ordered by whether they are "above" other segments or not.
Given two line segments, we take the segment whose start X coordinate is the
least, and check if the start of the other line is above or below. This is done
by seeing if the triangle formed by the start and end of the first line
and the first point of the other line, is clockwise or counter-clockwise. If
both lines have the same start point, the third point of the triangle is the
second point of the second line instead.

This only produces a consistent order if all the lines start before any
intersection point among those lines.
*/
template<typename Index,typename Coord> struct sweep_cmp {
    std::pmr::vector<loop_point<Index,Coord>> &lpoints;

    auto winding(Index p1,Index p2,Index p3) const {
        return triangle_winding(lpoints[p1].data,lpoints[p2].data,lpoints[p3].data);
    }

    bool operator()(segment<Index> s1,segment<Index> s2) const {
        if(s1.a_x(lpoints) == s2.a_x(lpoints)) {
            Coord r = s1.a_y(lpoints) - s2.a_y(lpoints);
            if(r) return r > 0;
            long_coord_t<Coord> r2 = winding(s1.a,s1.b,s2.b);
            if(r2) return r2 > 0;
        } else if(s1.a_x(lpoints) < s2.a_x(lpoints)) {
            if(s1.a_x(lpoints) != s1.b_x(lpoints)) {
                long_coord_t<Coord> r = winding(s1.a,s1.b,s2.a);
                if(r) return r > 0;
                r = winding(s1.a,s1.b,s2.b);
                if(r) return r > 0;
            }
        } else {
            if(s2.a_x(lpoints) != s2.b_x(lpoints)) {
                long_coord_t<Coord> r = winding(s2.b,s2.a,s1.a);
                if(r) return r > 0;
                r = winding(s2.b,s2.a,s1.b);
                if(r) return r > 0;
            }
        }

        return s1.b == s2.b ? (s1.a > s2.a) : (s1.b > s2.b);
    }
};

template<typename Index,typename Coord>
using sweep_t = std::pmr::set<segment<Index>,sweep_cmp<Index,Coord>>;

#ifndef NDEBUG
template<typename Index,typename Coord>
bool check_integrity(
    const std::pmr::vector<loop_point<Index,Coord>> &points,
    const sweep_t<Index,Coord> &sweep)
{
    for(auto &s : sweep) {
        if(!(points[s.a].next == s.b || points[s.b].next == s.a)) return false;
    }
    return true;
}
#endif

template<typename Index,typename Coord>
Index split_segment(
    events_t<Index,Coord> &events,
    sweep_t<Index,Coord> &sweep,
    std::pmr::vector<loop_point<Index,Coord>> &points,
    segment<Index> s,
    const point_t<Coord> &c,
    Index new_set,
    at_edge_t at_edge)
{
    Index sa = s.a;
    Index sb = s.b;

    /* The lines are replaced with split versions. The "BACKWARD" events
    will still exist but will be ignored because the lines are not found in
    "sweep" anymore.
    Even if the intersection is at an edge and the line doesn't need to be
    split, it still needs to be removed from sweep and readded as an event. The
    intersection point is rounded to integer coordinates. If the distance
    between the real intersection point and the edge of the line is small
    enough, but still greater than zero, that tiny nub can still affect the
    sweep order. */
    sweep.erase(s);

    if(at_edge != EDGE_NO) {
        add_event(
            events,
            sa,
            sb,
            points[sa].data[0] == points[sb].data[0] ? event_type_t::forward : event_type_t::vforward);
        return at_edge == EDGE_START ? sa : sb;
    }

    if(!s.a_is_main(points)) std::swap(sa,sb);

    points[sa].next = points.size();
    points.emplace_back(c,new_set,sb,loop_point<Index,Coord>::UNDEF_LINE_BAL);
    POLY_OPS_ASSERT_SLOW(check_integrity(points,sweep));

    /* even if the original line was not vertical, one of the new lines might
    still be vertical after rounding, so the comparisons done by "add_fb_events"
    cannot be consolidated */
    if(points[sa].data[0] <= points[sb].data[0]) {
        add_fb_events(points,events,sa,points[sa].next);
        add_fb_events(points,events,points[sa].next,sb);
    } else {
        add_fb_events(points,events,sb,points[sa].next);
        add_fb_events(points,events,points[sa].next,sa);
    }

    return points[sa].next;
}

template<typename Index,typename Coord>
bool check_intersection(
    events_t<Index,Coord> &events,
    std::pmr::vector<Index> &intrs,
    original_sets_t<Index> &o_sets,
    sweep_t<Index,Coord> &sweep,
    std::pmr::vector<loop_point<Index,Coord>> &points,
    segment<Index> s1,
    segment<Index> s2)
{
    point_t<Coord> intr;
    at_edge_t at_edge[2];
    if(intersects(s1,s2,points.data(),intr,at_edge)) {
        POLY_OPS_ASSERT(!at_edge[0] || !at_edge[1]);

        /* both instances of loop_point are treated like a single point and will
           refer to the same set */
        auto new_set = static_cast<Index>(o_sets.size());
        o_sets.emplace_back();

        Index intr1 = split_segment(events,sweep,points,s1,intr,new_set,at_edge[0]);
        Index intr2 = split_segment(events,sweep,points,s2,intr,new_set,at_edge[1]);

        intrs.push_back(intr1);
        intrs.push_back(intr2);

        add_event(events,intr1,intr2,event_type_t::calc_balance);

        return true;
    }

    return false;
}

template<typename T> concept sized_or_forward_range
    = std::ranges::sized_range<T> || std::ranges::forward_range<T>;

template<typename T,typename Coord> concept sized_or_forward_point_range
    = sized_or_forward_range<T> && point<std::ranges::range_value_t<T>,Coord>;

template<typename T,typename Coord> concept sized_or_forward_point_range_range
    /* It's not enought to know the size of inner_loops(). We have to iterate
    over the loops to get their sizes. */
    = std::ranges::forward_range<T> && sized_or_forward_point_range<std::ranges::range_value_t<T>,Coord>;

template<typename T,typename Coord> concept sized_or_forward_polygon = requires(const T &v) {
    { v.outer_loop() } -> sized_or_forward_point_range<Coord>;
    { v.inner_loops() } -> sized_or_forward_point_range_range<Coord>;
};

template<typename,typename T> struct total_point_count {
    static inline size_t doit(const T&) { return 100; }
};

template<typename Coord,sized_or_forward_polygon<Coord> T> struct total_point_count<Coord,T> {
    static size_t doit(const T &x) {
        size_t s = std::ranges::distance(x.outer_loop());
        for(auto &&loop : x.inner_loops()) s += std::ranges::distance(loop);
        return s;
    }
};

template<typename Index,typename Coord>
struct offset_polygon_point {
    std::pmr::vector<loop_point<Index,Coord>> lpoints;
    original_sets_t<Index> original_sets;
    real_coord_t<Coord> magnitude;
    Coord arc_step_size;
    Index original_index;
    Index lfirst;

    offset_polygon_point(
        real_coord_t<Coord> magnitude,
        Coord arc_step_size,
        Index reserve_size,
        std::pmr::memory_resource *contig_mem)
        : lpoints(contig_mem), original_sets(contig_mem), magnitude(magnitude),
          arc_step_size(arc_step_size), original_index(0)
    {
        lpoints.reserve(reserve_size*2 + reserve_size/3);
        original_sets.reserve(reserve_size);
    }

    void operator()(
        const point_t<Coord> &p1,
        const point_t<Coord> &p2,
        const point_t<Coord> &p3,
        bool first,
        bool last)
    {
        if(first) lfirst = static_cast<Index>(lpoints.size());

        point_t<real_coord_t<Coord>> offset = perp_vector<Coord>(p1,p2,magnitude);
        point_t<Coord> offset_point = p1 + vround<Coord>(offset);

        /* Unless this point is the first point of the loop, the last
        point was added to bridge the newly formed gap after displacing
        the line segments. If this point is the same as the last point
        then the two line segments connected to this point are colinear
        (or very close) and a gap was not created, thus this point
        should overwrite the previous, but all the fields are the same
        so nothing needs to be done. */
        if(first || offset_point != lpoints.back().data) {
            lpoints.emplace_back(offset_point,original_sets.size(),lpoints.size()+1);
        }
        original_sets.emplace_back();
        original_sets.back().insert(original_index);

        Index next_os = last ? lpoints[lfirst].original_set : original_sets.size();

        /* add a point for the new end of this line segment */
        lpoints.emplace_back(
            p2 + vround<Coord>(offset),
            next_os,
            lpoints.size()+1);

        if(triangle_winding(p1,p2,p3) * std::copysign(1.0,magnitude) < 0) {
            // it's concave so we need to approximate an arc

            real_coord_t<Coord> angle = coord_ops<Coord>::pi() - vangle<Coord>(p1-p2,p3-p2);
            Coord steps = coord_ops<Coord>::floor(magnitude * angle / arc_step_size);
            long lsteps = std::abs(static_cast<long>(steps));
            if(lsteps > 1) {
                real_coord_t<Coord> s = coord_ops<Coord>::sin(angle / steps);
                real_coord_t<Coord> c = coord_ops<Coord>::cos(angle / steps);

                for(long i=1; i<lsteps; ++i) {
                    offset = {c*offset[0] - s*offset[1],s*offset[0] + c*offset[1]};
                    offset_point = p2 + vround<Coord>(offset);

                    if(offset_point != lpoints.back().data) {
                        lpoints.emplace_back(offset_point,next_os,lpoints.size()+1);
                    }
                }
            }
        }

        if(last) {
            if(lpoints.back().data == lpoints[lfirst].data) {
                lpoints.pop_back();
            }
            lpoints.back().next = lfirst;
        }

        ++original_index;
    }

    void single_point_loop(const point_t<Coord> &) {
        /* A size of one can be handled as a special case. The result is a
        circle around the single point. For now, we just ignore such "loops". */
        ++original_index;
    }
};

template<typename Index,typename Coord,std::ranges::input_range R>
auto offset_polygon_loop(offset_polygon_point<Index,Coord> &opp,R &&loop) {
    point_t<Coord> p1,p2,pn,pn_1,pn_2;
    auto itr = loop.begin();
    auto end = loop.end();
    if(itr == end) return;
    p1 = *itr;
    if(++itr == end) {
        opp.single_point_loop(p1);
    }
    pn_2 = p1;
    pn_1 = p2 = *itr;
    bool first = true;
    while(++itr != end) {
        pn = *itr;
        opp(pn_2,pn_1,pn,first,false);
        pn_2 = std::exchange(pn_1,pn);
        first = false;
    }
    opp(pn_2,pn_1,p1,false,false);
    opp(pn_1,p1,p2,false,true);
}

template<typename Index,typename Coord,std::ranges::random_access_range R>
void offset_polygon_loop(offset_polygon_point<Index,Coord> &opp,const R &loop) {
    Index size = std::ranges::distance(loop);
    if(size <= 1) {
        if(size) opp.single_point_loop(*loop.begin());
        return;
    }
    for(Index i=0; i<size; ++i) opp(
        loop.begin()[i],
        loop.begin()[(i + 1) % size],
        loop.begin()[(i + 2) % size],
        i == 0,
        i == size-1);
}

template<typename Index,typename Coord,typename Input>
std::tuple<std::pmr::vector<loop_point<Index,Coord>>,original_sets_t<Index>> offset_polygon(
    const Input &input,
    real_coord_t<Coord> magnitude,
    Coord arc_step_size,
    std::pmr::memory_resource *contig_mem)
{
    offset_polygon_point<Index,Coord> opp(
        magnitude,
        arc_step_size,
        total_point_count<Coord,Input>::doit(input),
        contig_mem);

    offset_polygon_loop(opp,input.outer_loop());
    for(auto &&loop : input.inner_loops()) offset_polygon_loop(opp,loop);

    return {std::move(opp.lpoints),std::move(opp.original_sets)};
}

#ifndef NDEBUG
template<typename Index,typename Coord>
bool intersects_any(segment<Index> s1,const sweep_t<Index,Coord> &sweep,const loop_point<Index,Coord> *points) {
    for(auto s2 : sweep) {
        point_t<Coord> intr;
        at_edge_t at_edge[2];
        if(intersects(s1,s2,points,intr,at_edge)) {
            DEBUG_STEP_BY_STEP_MISSED_INTR;
            return true;
        }
    }
    return false;
}
#endif


/* This is a modified version of the Bentley–Ottmann algorithm. Lines are broken
   at intersections. Two end-points touching does not count as an intersection.
   */
template<typename Index,typename Coord>
std::pmr::vector<Index> self_intersection(
    std::pmr::vector<loop_point<Index,Coord>> &lpoints,
    original_sets_t<Index> &original_sets,
    std::pmr::memory_resource *contig_mem,
    std::pmr::memory_resource *discrete_mem)
{
    std::pmr::vector<event<Index>> lines(contig_mem);
    std::pmr::vector<Index> intrs(contig_mem);

    lines.reserve(lpoints.size()*2 + 10);

    for(Index i=0; i<lpoints.size(); ++i) {
        Index j1 = i;
        Index j2 = lpoints[i].next;

        if(lpoints[j1].data[0] > lpoints[j2].data[0]) std::swap(j1,j2);
        add_fb_events(lpoints,lines,j1,j2);
    }

    events_t<Index,Coord> events(event_cmp<Index,Coord>{lpoints},std::move(lines));
    sweep_t<Index,Coord> sweep(sweep_cmp<Index,Coord>{lpoints},discrete_mem);

    /* The sweep sorting functor requires that removing line segments has a
    higher priority than adding them, however, when encountering an intersection
    to test the line-balance on, when need to consider lines on both sides of
    the intersection point, thus removed lines are added to "sweep_removed" and
    kept until we advance to a point with a greater X coordinate.

    An alternative to consider would be to split calc_balance into two events
    with different priorities. It would simplify the code but would require
    testing line intersections twice with lines that don't start or stop at the
    intersection's X coordinate. */
    Coord last_x = std::numeric_limits<Coord>::lowest();
    std::pmr::vector<segment<Index>> sweep_removed(contig_mem);

    /* A sweep is done over the points. We travel from point to point from left
    to right, of each line segment. If the point is on the left side of the
    segment, the segment is added to "sweep". If the point is on the right, the
    segment is removed from "sweep". In the case of vertical lines, left and
    right are chosen arbitrarily. Adding to "sweep" always takes priority for
    vertical lines, otherwise removing takes priority.

    In the Bentley–Ottmann algorithm, we also have to swap the order of lines in
    "sweep" as we pass intersection points, but here we split lines at
    intersection points and create new "forward" and "backward" events instead.
    */
    while(!events.empty()) {
        event<Index> e = events.top();
        events.pop();

        if(e.ab.a_x(lpoints) > last_x) {
            last_x = e.ab.a_x(lpoints);
            sweep_removed.clear();
        }

        /*POLY_OPS_ASSERT_SLOW(std::ranges::all_of(sweep,[=,&lpoints](segment<Index> s) {
            return last_x >= s.a_x(lpoints) && last_x <= s.b_x(lpoints);
        }));*/

        switch(e.type) {
        case event_type_t::forward:
        case event_type_t::vforward:
            {
                auto itr = std::get<0>(sweep.insert(e.ab));

                DEBUG_STEP_BY_STEP_EVENT_F;

                if(itr != sweep.begin() && check_intersection(events,intrs,original_sets,sweep,lpoints,e.ab,*std::prev(itr))) continue;
                ++itr;
                if(itr != sweep.end()) check_intersection(events,intrs,original_sets,sweep,lpoints,e.ab,*itr);
            }
            break;
        case event_type_t::backward:
        case event_type_t::vbackward:
            {
                auto itr = sweep.find(e.line_ba());

                // if it's not in here, the line was split and no longer exists
                if(itr != sweep.end()) {
                    sweep_removed.push_back(*itr);
                    itr = sweep.erase(itr);

                    DEBUG_STEP_BY_STEP_EVENT_B;

                    if(itr != sweep.end() && itr != sweep.begin()) {
                        check_intersection(events,intrs,original_sets,sweep,lpoints,*std::prev(itr),*itr);
                    }

                    POLY_OPS_ASSERT_SLOW(!intersects_any(e.line_ba(),sweep,lpoints.data()));
                }
            }
            break;
        case event_type_t::calc_balance:
            {
                /* This event has nothing to with the Bentley–Ottmann algorithm.
                We need to know the "line balance" number of each line. Since
                the line number only changes between intersections and all the
                lines form loops, we only need to compute the number for lines
                whose first point is an intersection. */

                DEBUG_STEP_BY_STEP_EVENT_CALC_BALANCE;

                line_balance<Index,Coord> lb{lpoints.data(),e.ab.a,e.ab.b};
                for(const segment<Index> &s : sweep) lb.check(s);
                for(const segment<Index> &s : sweep_removed) lb.check(s);
                std::tie(
                    lpoints[e.ab.a].line_bal,
                    lpoints[e.ab.b].line_bal) = lb.result();
            }
            break;
        }
    }

    std::ranges::sort(intrs);
    intrs.resize(std::unique(intrs.begin(),intrs.end()) - intrs.begin());
    return intrs;
}

template<typename Index,typename Coord>
void merge_original_sets(
    loop_point<Index,Coord> *points,
    original_sets_t<Index> &original_sets,
    Index from,
    Index to)
{
    auto &&source = original_sets[points[from].original_set];
    original_sets[points[to].original_set].insert(source.begin(),source.end());
}

template<typename Index,typename Coord>
using broken_starts_t = std::pmr::map<point_t<Coord>,std::vector<Index>,point_less>;

/* Follow the points connected to the point at "intr" until another
intersection is reached or we wrapped around. Along the way, update the
"line_bal" value to the value at "intr", and if "line_bal" at "intr" is not 0,
merge all the "original_set" values into "intr".

If "line_bal" at "intr" is 0 and the "line_bal" of the next intersection is not,
add the point before the intersection to "broken_ends. If vice-versa, add the
intersecting point to "broken_starts". */
template<typename Index,typename Coord>
void follow_balance(
    loop_point<Index,Coord> *points,
    Index intr,
    std::pmr::vector<Index> &broken_ends,
    broken_starts_t<Index,Coord> &broken_starts,
    original_sets_t<Index> &original_sets)
{
    POLY_OPS_ASSERT((points[intr].line_bal != loop_point<Index,Coord>::UNDEF_LINE_BAL));
    Index next = points[intr].next;
    Index prev = intr;
    while(next != intr) {
        if(points[next].line_bal != loop_point<Index,Coord>::UNDEF_LINE_BAL) {
            if(points[next].line_bal == 0) {
                if(points[intr].line_bal != 0) {
                    broken_starts[points[next].data].push_back(next);
                    merge_original_sets(points,original_sets,next,intr);
                    points[next].original_set = points[intr].original_set;
                }
            } else if(points[intr].line_bal == 0) {
                broken_ends.push_back(prev);
            }
            break;
        }

        points[next].line_bal = points[intr].line_bal;
        if(points[intr].line_bal != 0) {
            merge_original_sets(points,original_sets,next,intr);
        }

        prev = next;
        next = points[next].next;
    }
}

template<typename Index> struct loop_location {
    Index start;
    Index size;
};

} // namespace detail

template<typename Index,typename Coord> class proto_loop_iterator {
    const detail::loop_point<Index,Coord> *lpoints;
    Index i;

public:
    bool operator==(const proto_loop_iterator &b) const {
        POLY_OPS_ASSERT(lpoints == b.lpoints);
        return i == b.it;
    }

    bool operator!=(const proto_loop_iterator &b) const {
        POLY_OPS_ASSERT(lpoints == b.lpoints);
        return i != b.it;
    }

    proto_loop_iterator &operator++() {
        i == lpoints[i].next;
        return *this;
    }

    proto_loop_iterator operator++(int) {
        return {lpoints,std::exchange(i,lpoints[i].next)};
    }

    const point_t<Coord> &operator*() const { return lpoints[i]; }

    const point_t<Coord> *operator->() const { return &lpoints[i]; }
};

template<typename Index,typename Coord> class tracked_proto_loop_iterator {
    const detail::loop_point<Index,Coord> *lpoints;
    const detail::mini_flat_set<Index,std::pmr::polymorphic_allocator<Index>> *original_sets;
    Index i;

public:
    bool operator==(const tracked_proto_loop_iterator &b) const {
        POLY_OPS_ASSERT(lpoints == b.lpoints && original_sets == b.original_sets);
        return i == b.it;
    }

    bool operator!=(const tracked_proto_loop_iterator &b) const {
        POLY_OPS_ASSERT(lpoints == b.lpoints && original_sets == b.original_sets);
        return i != b.it;
    }

    tracked_proto_loop_iterator &operator++() {
        i == lpoints[i].next;
        return *this;
    }

    tracked_proto_loop_iterator operator++(int) {
        return {lpoints,original_sets,std::exchange(i,lpoints[i].next)};
    }

    std::tuple<const point_t<Coord>&,std::span<Index>> operator*() const {
        return {lpoints[i],std::span(original_sets[i])};
    }
};

namespace detail {

template<typename Index,typename Coord>
void normalize_polygons(
    std::pmr::vector<loop_point<Index,Coord>> &lpoints,
    original_sets_t<Index> &original_sets,
    bool has_holes,
    std::pmr::memory_resource *contig_mem,
    std::pmr::memory_resource *discrete_mem)
{
    auto intrs = self_intersection(lpoints,original_sets,contig_mem,discrete_mem);

    broken_starts_t<Index,Coord> broken_starts(discrete_mem);
    std::pmr::vector<Index> broken_ends(contig_mem);

    for(auto intr : intrs) {
        follow_balance<Index,Coord>(lpoints.data(),intr,broken_ends,broken_starts,original_sets);
    }

#if POLY_OPS_GRAPHICAL_DEBUG
    if(mc__) mc__->console_line_stream() << "broken_starts: " << pp(broken_starts) << "\nbroken_ends: " << pp(broken_ends);
    delegate_drawing_trimmed(lpoints,original_sets);
#endif

    /* match all the points in broken_starts and broken_ends to make new loops
    where all the points have a line balance of 0 */
    for(auto intr : broken_ends) {
        auto &os = broken_starts[lpoints[lpoints[intr].next].data];

        POLY_OPS_ASSERT(os.size());

        merge_original_sets<Index,Coord>(lpoints.data(),original_sets,lpoints[intr].next,os.back());
        lpoints[intr].next = os.back();
        os.pop_back();
    }

    // there shouldn't be any left
    POLY_OPS_ASSERT(std::all_of(
        broken_starts.begin(),
        broken_starts.end(),
        [](const broken_starts_t<Index,Coord>::value_type &v) { return std::get<1>(v).empty(); }));

    {
        // merge the original point sets at common points
        std::pmr::map<point_t<Coord>,Index,point_less> by_coord(discrete_mem);
        for(auto intr : intrs) {
            auto os = by_coord.find(lpoints[intr].data);
            if(os == by_coord.end()) {
                by_coord.emplace(lpoints[intr].data,lpoints[intr].original_set);
            } else if(os->second != lpoints[intr].original_set) {
                original_sets[os->second].merge(original_sets[lpoints[intr].original_set]);
                lpoints[intr].original_set = os->second;
            }
        }
    }

#if POLY_OPS_GRAPHICAL_DEBUG
    delegate_drawing_trimmed(lpoints,original_sets);
#endif

    // find all the new loops
    std::pmr::vector<loop_location<Index>> loops(contig_mem);
    for(size_t i=0; i<lpoints.size(); ++i) {
        if(lpoints[i].line_bal == 0) {
            loops.emplace_back(i,1);

            /* give this a non-zero value to prevent this point from being
            scanned again */
            lpoints[i].line_bal = 1;

            for(size_t j = lpoints[i].next; j != i; j = lpoints[j].next) {
                POLY_OPS_ASSERT(lpoints[j].line_bal == 0);

                ++loops.back().size;
                lpoints[j].line_bal = 1;
            }
        }
    }

    /*std::vector<std::vector<index_t>> orig_to_new;
    orig_to_new.resize(total);

    for(size_t i=0; i<lpoints.size(); ++i) {
        if(lpoints[i].line_bal == 0) {
            auto &orig = original_sets[lpoints[i].original_set];
            std::copy(orig.begin(),orig.end(),orig_to_new[i].begin());
        }
    }*/

    /*for_each_line(points,sizes,loop_count,[&](unsigned int j,unsigned int j2) {
            std::cout << points[j*2] << ',' << points[j*2+1] << " - " << points[j2*2] << ',' << points[j2*2+1] << '\n';
        });*/

    //dump_loop_points(lpoints);
}

} // namespace detail

template<std::integral Index,detail::coordinate Coord,polygon<Coord> Input>
void offset_stroke_triangulate(
    Input &&input,
    real_coord_t<Coord> magnitude,
    Coord arc_step_size,
    std::pmr::memory_resource *contig_mem,
    std::pmr::memory_resource *discrete_mem)
{
    auto [lpoints,original_sets] = detail::offset_polygon<Index,Coord,Input>(input,magnitude,arc_step_size,contig_mem);
    detail::normalize_polygons<Index,Coord>(
        lpoints,
        original_sets,
        std::ranges::empty(input.inner_loops()),
        contig_mem,
        discrete_mem);
}

template<std::integral Index,detail::coordinate Coord,polygon<Coord> Input>
void offset_stroke_triangulate(
    Input &&input,
    real_coord_t<Coord> magnitude,
    Coord arc_step_size,
    std::pmr::memory_resource *mem)
{
    std::pmr::unsynchronized_pool_resource discrete_mem(mem);
    offset_stroke_triangulate<Index,Coord>(std::forward<Input>(input),magnitude,arc_step_size,mem,&discrete_mem);
}

template<std::integral Index,detail::coordinate Coord,polygon<Coord> Input>
void offset_stroke_triangulate(
    Input &&input,
    real_coord_t<Coord> magnitude,
    Coord arc_step_size)
{
    offset_stroke_triangulate<Index,Coord>(std::forward<Input>(input),magnitude,arc_step_size,std::pmr::get_default_resource());
}

/* A basic polygon type that has all the points in a single array. The first
loop is the outer wall and any subsequent loops are holes. */
template<std::integral Index,detail::coordinate Coord,point<Coord> Point>
struct basic_polygon {
    const Point *points;
    const std::span<const Index> ends;

    auto outer_loop() const {
        return std::span(points,points+ends[0]);
    }

    auto inner_loops() const {
        return std::views::iota(Index(0),static_cast<Index>(ends.size()-1))
            | std::views::transform([points=points,ends=ends](Index i){
                return std::span(points + ends[i],points + ends[i+1]); });
    }
};

} // namespace poly_ops

#undef DEBUG_STEP_BY_STEP_EVENT_F
#undef DEBUG_STEP_BY_STEP_EVENT_B
#undef DEBUG_STEP_BY_STEP_EVENT_CALC_BALANCE
#undef DEBUG_STEP_BY_STEP_LB_CHECK_RET_TYPE
#undef DEBUG_STEP_BY_STEP_LB_CHECK_RETURN
#undef DEBUG_STEP_BY_STEP_LB_CHECK_FF

#endif
