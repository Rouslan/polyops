#ifndef POLY_OPS_NORMALIZE_HPP
#define POLY_OPS_NORMALIZE_HPP

#include <vector>
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
#include <type_traits>
#include <random>

/* POLY_OPS_ASSERT can be defined to use an exception, so it can't be used
inside functions marked as "noexcept". In such cases, "assert" is used. */
#include <cassert>

#include "base.hpp"
#include "sweep_set.hpp"


#ifndef POLY_OPS_DEBUG_LOG
#define POLY_OPS_DEBUG_LOG(...) (void)0
#endif

#ifndef POLY_OPS_DEBUG_ITERATION
#define POLY_OPS_DEBUG_ITERATION
#endif

namespace poly_ops {

/**
 * Point tracking common interface.
 *
 * The "clipper" class stores all points in a single array. To associate extra data
 * with each point, a second array may be created with the extra data. This
 * interface is a set of callbacks that allows keeping an external array
 * synchronized with clipper's internal array.
 *
 * Note that the destructor is not virtual, and is instead protected.
 */
template<typename Index=std::size_t> class i_point_tracker {
public:
    /**
     * Called when a line is split, adding a new point.
     *
     * `a` and `b` are the endpoints of the line that was split. Like
     * `point_added`, the index of this new point is one greater than the index
     * of the last point added.
     *
     * \param a The index of the intersection point of the first line.
     * \param b The index of the intersection point of the second line.
     */
    virtual void new_point_between(Index a,Index b) = 0;

    /**
     * Called when points are "merged".
     *
     * When parts of the shape are going to be truncated, this is called on the
     * first point to be discarded as `from` and the point that follows it as
     * `to`. If the next point is also to be discarded, this is called again
     * with that point along with the point after that. This is repeated until
     * the point reached is not going to be discarded, or the following point
     * was already discarded because we've circled around the entire loop.
     *
     * Discarded points are never referenced again.
     *
     * \param from The index of the point to be discarded.
     * \param to The index of the point that follows `from`.
     */
    virtual void point_merge(Index from,Index to) = 0;

    /**
     * Called for every point initially added.
     *
     * Every added point has an implicit index (this index is unrelated to
     * `original_i`). This method is first called when point zero is added.
     * Every subsequent call corresponds to the index that is one greater than
     * the index of the previously added point.
     *
     * `original_i` is the index of the input point that the added point
     * corresponds to. The value is what the array index of the original point
     * would be if all the input points were added, in order, to a single array.
     * This will not necessarily be called for every point of the input;
     * duplicate consecutive points are ignored.
     *
     * \param original_i The index of the input point that this added point
     *     corresponds to.
     */
    virtual void point_added(Index original_i) = 0;

    /**
     * Called if the last "n" points added were removed.
     *
     * This only happens when a polygon is added with fewer than three points.
     * The clipper class doesn't work with such polygons and after counting the
     * number of points added, will remove the points of those polygons.
     *
     * \param n The number of points to remove. This will be equal to 1 or 2.
     */
    virtual void points_removed(Index n) = 0;

protected:
    ~i_point_tracker() = default;
};

template<typename Coord,typename Index=std::size_t> struct null_tracker;

/**
 * `null_tracker` is stateless. The clipper data types use
 * `[[no_unique_address]]` on pointers to trackers, so an empty struct is used
 * instead of a real pointer for `null_tracker`.
 */
template<typename Coord,typename Index> struct null_tracker_ptr {
    null_tracker<Coord,Index> operator*() const noexcept;
};

template<typename Coord,typename Index> struct null_tracker {
    i_point_tracker<Index> *callbacks() { return nullptr; }
    point_t<Coord> get_value(Index,const point_t<Coord> &p) const { return p; }

    null_tracker_ptr<Coord,Index> operator&() const noexcept { return {}; }
};

template<typename Coord,typename Index>
null_tracker<Coord,Index> null_tracker_ptr<Coord,Index>::operator*() const noexcept { return {}; }

template<typename T,typename Coord,typename Index> concept point_tracker =
    requires(T val,const T cval,Index i,point_t<Coord> p) {
        { val.callbacks() } -> std::convertible_to<i_point_tracker<Index>*>;
        cval.get_value(i,p);
    };

/**
 * Given two sets of polygons, `subject` and `clip`, specifies which operation
 * to perform.
 */
enum class bool_op {
    /**
     * Boolean operation `subject` OR `clip`.
     * 
     * \verbatim embed:rst:leading-asterisk
     * .. image:: /_static/union.svg
     *     :alt: union operation example
     * \endverbatim
     */
    union_ = 0,

    /**
     * Boolean operation `subject` AND `clip`.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. image:: /_static/intersection.svg
     *     :alt: intersection operation example
     * \endverbatim
     */
    intersection,

    /**
     * Boolean operation `subject` XOR `clip`
     *
     * \verbatim embed:rst:leading-asterisk
     * .. image:: /_static/xor.svg
     *     :alt: xor operation example
     * \endverbatim
     */
    xor_,

    /**
     * Boolean operation `subject` AND NOT `clip`
     *
     * \verbatim embed:rst:leading-asterisk
     * .. image:: /_static/difference.svg
     *     :alt: difference operation example
     * \endverbatim
     */
    difference,

    /**
     * Keep all lines but make it so all outer lines are clockwise polygons, all
     * singly nested lines are counter-clockwise polygons, all doubly-nested
     * lines are clockwise polygons, and so forth.
     */
    normalize};

/**
 * Specifies one of two sets.
 *
 * The significance of these sets depends on the operation performed.
 */
enum class bool_set {subject=0,clip=1};

namespace detail {

namespace line_state {
enum type : unsigned char {
    undef=0b100,
    check=0b1000,

    /* Points of type "anchor" are temporary points added to keep track of line
    segments. When the line segments are split and rejoined, both the start and
    end points may be reassigned, so a third point may be added between the two,
    to serve as an anchor point inside a loop. */
    anchor=0b10000,

    discard=0b0,
    keep=0b1,
    reverse=0b10,

    /* this state is needed because if a segment has a state of "keep_rev", it
    needs a new first point that must come from another reversed line, even if
    the other line isn't kept */
    discard_rev=discard|reverse,

    keep_rev=keep|reverse,
    anchor_undef=anchor|undef,
    
    /* these names don't appear in the code but they help with debugging */
    anchor_discard=anchor|discard,
    anchor_discard_rev=anchor|discard_rev,
    anchor_keep=anchor|keep,
    anchor_keep_rev=anchor|keep_rev};

inline type operator|(type a,type b) { return type(static_cast<unsigned char>(a) | static_cast<unsigned char>(b)); }
inline type operator&(type a,type b) { return type(static_cast<unsigned char>(a) & static_cast<unsigned char>(b)); }
};

enum {
    L_INDEX_UNSET = -1,
    L_INDEX_DISCARDED = -2,
    L_INDEX_ANCHOR = -3};

struct line_desc {
    bool_set cat;
    line_state::type state;
};

template<typename Index,typename Coord> struct loop_point {
    point_t<Coord> data;
    Index next;

    union line_aux {
        line_desc desc;
        int loop_index;
    } aux;

    loop_point() = default;
    loop_point(const loop_point &b) = default;
    loop_point(point_t<Coord> data,Index next,bool_set cat)
        noexcept(std::is_nothrow_copy_constructible_v<point_t<Coord>>)
        : data{data}, next{next}, aux{line_desc{cat,line_state::undef}} {}
    
    bool has_line_bal() const {
        POLY_OPS_ASSERT(aux.desc.state != line_state::check);
        return (aux.desc.state & ~line_state::anchor) != line_state::undef;
    }

    bool_set bset() const { return aux.desc.cat; }
    bool_set &bset() { return aux.desc.cat; }
    line_state::type state() const { return aux.desc.state; }
    line_state::type &state() { return aux.desc.state; }

    friend void swap(loop_point &a,loop_point &b) noexcept(std::is_nothrow_swappable_v<point_t<Coord>>) {
        using std::swap;

        swap(a.data,b.data);
        swap(a.next,b.next);
        swap(a.aux,b.aux);
    }
};

/* The order of these determines their priority from greatest to least.

"vforward" and "vbackward" have the same meaning as "forward" and "backward" but
have different priorities. They are used for vertical lines, which, unlike
non-vertical lines, can intersect with lines that start and end at the same X
coordinate as the start/end of the vertical line.

The comparison functor "sweep_cmp" requires that "backward" come before
"forward". */
enum class event_type_t {vforward,backward,forward,vbackward};

/* "normal" is the default state.

"deleted" is used instead of removing events from the array, to avoid spending
time on moving elements backward. */
enum class event_status_t {normal,deleted};

template<typename Index> struct event {
    segment<Index> ab;
    Index sweep_node;
    event_type_t type;
    event_status_t status;

    event() = default;
    event(segment<Index> ab,Index sweep_node,event_type_t type) noexcept
        : ab{ab}, sweep_node{sweep_node}, type{type}, status{event_status_t::normal} {}
    event(const event &b) = default;

    event &operator=(const event&) = default;

    segment<Index> line_ba() const { return {ab.b,ab.a}; }

    friend bool operator==(const event &a,const event &b) {
        return a.type == b.type && a.ab == b.ab;
    }
};

// This refers to a 1-dimensional edge, i.e. the start or end of a line segment
enum class at_edge_t {
    start = -1,
    no = 0,
    end = 1,
    both = 2};

template<typename Coord> at_edge_t is_edge(Coord val,Coord end_val) {
    if(val == 0) return at_edge_t::start;
    if(val == end_val) return at_edge_t::end;
    return at_edge_t::no;
}

template<typename T> bool between(T x,T lo,T hi) {
    return x > lo && x < hi;
}

template<typename Coord> long_coord_t<Coord> int_dot(const point_t<Coord> &a,const point_t<Coord> &b) {
    return coord_ops<Coord>::mul(a[0],b[0]) + coord_ops<Coord>::mul(a[1],b[1]);
}

template<typename Index,typename Coord>
bool intersects_parallel(
    const cached_segment<Index,Coord> &s1,
    const cached_segment<Index,Coord> &s2,
    point_t<Coord> &p,
    std::span<at_edge_t,2> at_edge)
{
    if(triangle_winding(s1.pa,s1.pb,s2.pa) != 0) return false;

    /* Reduce each line segment into a 1D value. Each d* value is the scalar
    projection of the corresponding s*.* point times the magnitude of s1. */
    point_t<Coord> v1b = s1.pb - s1.pa;
    auto d1a = static_cast<long_coord_t<Coord>>(0);
    auto d1b = int_dot(v1b,v1b);
    auto d2a = int_dot(s2.pa-s1.pa,v1b);
    auto d2b = int_dot(s2.pb-s1.pa,v1b);

    POLY_OPS_ASSERT(0 != d1b && d2a != d2b);

    auto [lo1,hi1] = std::minmax(d1a,d1b);
    auto [lo2,hi2] = std::minmax(d2a,d2b);

    if(between(d1a,lo2,hi2)) {
        p = s1.pa;
        at_edge[0] = at_edge_t::start;
        at_edge[1] = at_edge_t::no;
        return true;
    }
    if(between(d1b,lo2,hi2)) {
        p = s1.pb;
        at_edge[0] = at_edge_t::end;
        at_edge[1] = at_edge_t::no;
        return true;
    }
    if(between(d2a,lo1,hi1)) {
        p = s2.pa;
        at_edge[0] = at_edge_t::no;
        at_edge[1] = at_edge_t::start;
        return true;
    }
    if(between(d2b,lo1,hi1)) {
        p = s2.pb;
        at_edge[0] = at_edge_t::no;
        at_edge[1] = at_edge_t::end;
        return true;
    }

    /* if both points intersect, p is not set */
    if(lo1 == lo2 && hi1 == hi2) {
        if(d1a == d2a) {
            if(s1.a != s2.a) {
                if(s1.b != s2.b) {
                    at_edge[0] = at_edge_t::both;
                    at_edge[1] = at_edge_t::both;
                    return true;
                }
                p = s1.pa;
                at_edge[0] = at_edge_t::start;
                at_edge[1] = at_edge_t::start;
                return true;
            } else if(s1.b != s2.b) {
                p = s1.pb;
                at_edge[0] = at_edge_t::end;
                at_edge[1] = at_edge_t::end;
                return true;
            }
        } else {
            if(s1.a != s2.b) {
                if(s1.b != s2.a) {
                    at_edge[0] = at_edge_t::both;
                    at_edge[1] = at_edge_t::both;
                    return true;
                }
                p = s1.pa;
                at_edge[0] = at_edge_t::start;
                at_edge[1] = at_edge_t::end;
                return true;
            } else if(s1.b != s2.a) {
                p = s1.pb;
                at_edge[0] = at_edge_t::end;
                at_edge[1] = at_edge_t::start;
                return true;
            }
        }
    }

    return false;
}

/* This pseudo-random number generator is used to decide rounding of line
intersection points. Its output doesn't need to be high quality. */
using rand_generator = std::minstd_rand;

/* std::uniform_int_distribution(0,1) appears to behave differently between
GCC's and MSVC's libraries, so it isn't used */
bool rand_bool(rand_generator &rgen) {
    constexpr auto mid = rand_generator::min() + (rand_generator::max() - rand_generator::min()) / 2;
    return rgen() > mid;
}

/* Check if two line segments intersect.

Returns "true" if the lines intersect. Adjacent lines do not count as
intersecting.

'p' will be set to the point of intersection, or in the case of coincident
lines, 'p' will be one of the end points that is inside the other line (if both
ends are inside the other segment, one end is chosen arbitrarily). */
template<typename Index,typename Coord>
bool intersects(
    const cached_segment<Index,Coord> &s1,
    const cached_segment<Index,Coord> &s2,
    point_t<Coord> &p,
    std::span<at_edge_t,2> at_edge,
    rand_generator &rgen)
{
    const Coord x1 = s1.pa.x();
    const Coord y1 = s1.pa.y();
    const Coord x2 = s1.pb.x();
    const Coord y2 = s1.pb.y();
    const Coord x3 = s2.pa.x();
    const Coord y3 = s2.pa.y();
    const Coord x4 = s2.pb.x();
    const Coord y4 = s2.pb.y();

    long_coord_t<Coord> d = coord_ops<Coord>::mul(x1-x2,y3-y4) - coord_ops<Coord>::mul(y1-y2,x3-x4);
    if(d == 0) return intersects_parallel(s1,s2,p,at_edge);

    /* Connected lines do not count as intersecting. This check must be done
    after checking for parallel lines because coincident lines can have more
    than one point of intersection. */
    if(s1.a == s2.a || s1.a == s2.b || s1.b == s2.a || s1.b == s2.b) return false;

    long_coord_t<Coord> t_i = coord_ops<Coord>::mul(x1-x3,y3-y4) - coord_ops<Coord>::mul(y1-y3,x3-x4);
    long_coord_t<Coord> u_i = coord_ops<Coord>::mul(x1-x3,y1-y2) - coord_ops<Coord>::mul(y1-y3,x1-x2);

    if(d > 0) {
        if(t_i < 0 || t_i > d || u_i < 0 || u_i > d) return false;
    } else if(t_i > 0 || t_i < d || u_i > 0 || u_i < d) return false;

    if(t_i == 0) {
        at_edge[0] = at_edge_t::start;
        at_edge[1] = is_edge(u_i,d);
        p[0] = x1;
        p[1] = y1;
    } else if(t_i == d) {
        at_edge[0] = at_edge_t::end;
        at_edge[1] = is_edge(u_i,d);
        p[0] = x2;
        p[1] = y2;
    } else if(u_i == 0) {
        at_edge[0] = at_edge_t::no;
        at_edge[1] = at_edge_t::start;
        p[0] = x3;
        p[1] = y3;
    } else if(u_i == d) {
        at_edge[0] = at_edge_t::no;
        at_edge[1] = at_edge_t::end;
        p[0] = x4;
        p[1] = y4;
    } else {
        auto t = static_cast<real_coord_t<Coord>>(t_i)/static_cast<real_coord_t<Coord>>(d);
        real_coord_t<Coord> rx = t * static_cast<real_coord_t<Coord>>(x2-x1);
        real_coord_t<Coord> ry = t * static_cast<real_coord_t<Coord>>(y2-y1);

        /* Simple rounding is not sufficient because there is a tiny chance that
        lines are arranged such that rounding one intersection reorders the
        lines and creates a new intersection, and rounding the new intersection
        creates yet another intersection, and so forth, until the lines are
        nowhere near where they started. */

        p[0] = static_cast<Coord>(x1 + (rand_bool(rgen) ? coord_ops<Coord>::floor(rx) : coord_ops<Coord>::ceil(rx)));
        p[1] = static_cast<Coord>(y1 + (rand_bool(rgen) ? coord_ops<Coord>::floor(ry) : coord_ops<Coord>::ceil(ry)));

        at_edge[0] = at_edge_t::no;
        at_edge[1] = at_edge_t::no;

        if(p[0] == x1 && p[1] == y1) at_edge[0] = at_edge_t::start;
        else if(p[0] == x2 && p[1] == y2) at_edge[0] = at_edge_t::end;

        if(p[0] == x3 && p[1] == y3) at_edge[1] = at_edge_t::start;
        else if(p[0] == x4 && p[1] == y4) at_edge[1] = at_edge_t::end;
    }

    return true;
}

template<typename Coord>
bool lower_angle(point_t<Coord> ds,point_t<Coord> dp,bool tie_breaker) {
    long_coord_t<Coord> v = coord_ops<Coord>::mul(ds.y(),dp.x());
    long_coord_t<Coord> w = coord_ops<Coord>::mul(dp.y(),ds.x());
    return v < w || (v == w && tie_breaker);
}

template<typename Coord>
bool vert_overlap(point_t<Coord> sa,point_t<Coord> sb,point_t<Coord> p,point_t<Coord> dp,bool tie_breaker) {
    /* If both line segments are vertical, two assumptions are made: they either
    overlap completely or not at all, and 'sb' comes directly after 'sa' in the
    loop. */
    return
        tie_breaker &&
        dp.x() == 0 &&
        (dp.y() > 0) == (sb.y() > sa.y()) &&
        p.y() == sa.y();
}

/* Determine if a ray at point 'p' intersects with line segment 'sa-sb'. The ray
is vertical and extends toward negative infinity. */
template<typename Coord> bool line_segment_up_ray_intersection(
    point_t<Coord> sa,
    point_t<Coord> sb,
    point_t<Coord> p,
    Coord hsign,
    point_t<Coord> dp,
    bool tie_breaker)
{
    POLY_OPS_ASSERT(sa.x() <= sb.x());

    point_t<Coord> ds = sb - sa;
    if(ds.x() == 0) return vert_overlap(sa,sb,p,dp,tie_breaker);

    if(p.x() < sa.x() || (p.x() == sa.x() && hsign < 0)) return false;
    if(p.x() > sb.x() || (p.x() == sb.x() && hsign > 0)) return false;

    long_coord_t<Coord> t = coord_ops<Coord>::mul(p.x()-sa.x(),ds.y());
    long_coord_t<Coord> u = coord_ops<Coord>::mul(p.y()-sa.y(),ds.x());
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
    POLY_OPS_ASSERT(sa.x() <= sb.x());

    point_t<Coord> ds = sb - sa;
    if(ds.x() == 0) return {
        vert_overlap(sa,sb,p,dp1,tie_breaker1),
        vert_overlap(sa,sb,p,dp2,tie_breaker2)};

    if(p.x() < sa.x() || p.x() > sb.x()) return {false,false};

    std::tuple r(true,true);
    if(p.x() == sa.x()) r = {hsign1 > 0,hsign2 > 0};
    else if(p.x() == sb.x()) r = {hsign1 < 0,hsign2 < 0};

    long_coord_t<Coord> t = coord_ops<Coord>::mul(p.x()-sa.x(),ds.y());
    long_coord_t<Coord> u = coord_ops<Coord>::mul(p.y()-sa.y(),ds.x());
    if(t > u) return {false,false};
    if(t == u) {
        std::get<0>(r) = std::get<0>(r) && lower_angle(ds,dp1,tie_breaker1);
        std::get<1>(r) = std::get<1>(r) && lower_angle(ds,dp2,tie_breaker2);
    }

    return r;
}

template<typename Coord> Coord hsign_of(const point_t<Coord> &delta) {
    auto r = delta.x() ? delta.x() : delta.y();
    POLY_OPS_ASSERT(r != 0);
    return r;
}

template<typename Index,typename Coord> point_t<Coord> line_delta(
    const loop_point<Index,Coord> *points,Index p)
{
    return end_point(points,p) - points[p].data;
}

/* Calculate the "line balance".

The line balance is based on the idea of a winding number, except it applies to
the lines instead of the points inside. A positive number means nested geometry,
negative means inverted geometry, and zero means normal geometry (clockwise
oriented polygons and counter-clockwise holes). */
template<typename Index,typename Coord> class dual_line_balance {
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
    'line_segment_up_ray_intersection' function takes a 'hsign' parameter. When
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
    eliminated in a prior step). */

    const loop_point<Index,Coord> *points;
    Index p1, p2;
    std::span<std::pmr::vector<Index>,2> intrs_tmp;
    point_t<Coord> dp1, dp2;
    Coord hsign1, hsign2;
    int wn1, wn2;

public:
    dual_line_balance(const loop_point<Index,Coord> *points,Index p1,Index p2,std::span<std::pmr::vector<Index>,2> intrs_tmp) :
        points(points), p1(p1), p2(p2), intrs_tmp(intrs_tmp),
        dp1(line_delta(points,p1)), dp2(line_delta(points,p2)),
        hsign1(hsign_of(dp1)), hsign2(hsign_of(dp2)),
        wn1(dp1.x() < 0 ? -1 : 0), wn2(dp2.x() < 0 ? -1 : 0)
    {
        POLY_OPS_ASSERT(hsign1 && hsign2 && p1 != p2 && points[p1].data == points[p2].data);
    }

    void check(segment<Index> s) {
        POLY_OPS_ASSERT(s.a_x(points) <= s.b_x(points));

        bool a_is_main = s.a_is_main(points);
        Index main_i = a_is_main ? s.a : s.b;

        // is s the line at p1?
        if(main_i == p1) {
            if(line_segment_up_ray_intersection(
                points[s.a].data,
                points[s.b].data,
                points[p1].data,
                hsign2,
                dp2,
                main_i > p2))
            {
                wn2 += a_is_main ? 1 : -1;
                intrs_tmp[1].push_back(main_i);
            }
            return;
        }

        // is s the line at p2?
        if(main_i == p2) {
            if(line_segment_up_ray_intersection(
                points[s.a].data,
                points[s.b].data,
                points[p1].data,
                hsign1,
                dp1,
                main_i > p1))
            {
                wn1 += a_is_main ? 1 : -1;
                intrs_tmp[0].push_back(main_i);
            }
            return;
        }

        auto [intr1,intr2] = dual_line_segment_up_ray_intersection(
            points[s.a].data,
            points[s.b].data,
            points[p1].data,
            hsign1,
            dp1,
            main_i > p1,
            hsign2,
            dp2,
            main_i > p2);
        if(intr1) {
            wn1 += a_is_main ? 1 : -1;
            intrs_tmp[0].push_back(main_i);
        }
        if(intr2) {
            wn2 += a_is_main ? 1 : -1;
            intrs_tmp[1].push_back(main_i);
        }

        POLY_OPS_ASSERT_SLOW(
            intr1 == line_segment_up_ray_intersection(
                points[s.a].data,points[s.b].data,points[p1].data,hsign1,dp1,main_i > p1));
        POLY_OPS_ASSERT_SLOW(
            intr2 == line_segment_up_ray_intersection(
                points[s.a].data,points[s.b].data,points[p1].data,hsign2,dp2,main_i > p2));
    }

    std::tuple<int,int> result() const { return {wn1,wn2}; }
};

/* Find intersections with a vertical ray extending to negative infinity.

This is similar to "dual_line_balance" except it only tests a line with one
origin point.
*/
template<typename Index,typename Coord> class line_balance {
    const loop_point<Index,Coord> *points;
    Index p1;
    std::pmr::vector<Index> &intrs_tmp;
    point_t<Coord> dp1;
    Coord hsign;
    int wn[2];

public:
    line_balance(const loop_point<Index,Coord> *points,Index p1,std::pmr::vector<Index> &intrs_tmp) :
        points(points), p1(p1), intrs_tmp(intrs_tmp),
        dp1(line_delta(points,p1)),
        hsign(hsign_of(dp1)), wn{0,0}
    {
        POLY_OPS_ASSERT(hsign);
        if(dp1.x() < 0) wn[static_cast<int>(points[p1].bset())] = -1;
    }

    void check(segment<Index> s) {
        POLY_OPS_ASSERT(s.a_x(points) <= s.b_x(points));

        bool a_is_main = s.a_is_main(points);
        Index main_i = a_is_main ? s.a : s.b;

        // is s the line at p1?
        if(main_i != p1) {
            if(line_segment_up_ray_intersection(
                points[s.a].data,
                points[s.b].data,
                points[p1].data,
                hsign,
                dp1,
                main_i > p1))
            {
                wn[static_cast<int>(points[main_i].bset())] += a_is_main ? 1 : -1;
                intrs_tmp.emplace_back(main_i);
            }
        }
    }

    line_state::type result(bool_op op) const {
        int ws = wn[static_cast<int>(bool_set::subject)];
        int wc = wn[static_cast<int>(bool_set::clip)];
        int w = ws + wc;

        switch(op) {
        case bool_op::union_:
            return w == 0 ? line_state::keep : line_state::discard;
        case bool_op::intersection:
            if(points[p1].bset() == bool_set::clip) std::swap(ws,wc);
            return (ws == 0 && wc > 0) ? line_state::keep : line_state::discard;
        case bool_op::xor_:
            if(points[p1].bset() == bool_set::clip) std::swap(ws,wc);
            if(ws == 0) {
                if(wc > 0) return line_state::keep_rev;
                return line_state::keep;
            }
            return line_state::discard;
        case bool_op::difference:
            if(points[p1].bset() == bool_set::subject) {
                if(ws == 0) {
                    if(wc > 0) return line_state::discard;
                    return line_state::keep;
                }
            } else if(wc == 0) {
                if(ws > 0) return line_state::keep_rev;
                return line_state::discard_rev;
            }
            return line_state::discard;
        case bool_op::normalize:
            return w % 2 == 0 ? line_state::keep : line_state::keep_rev;
        }

        assert(false);
        return line_state::discard;
    }
};

/* order events by the X coordinates increasing and then by event type */
template<typename Index,typename Coord> struct event_cmp {
    const std::pmr::vector<loop_point<Index,Coord>> &lpoints;

    bool operator()(const event<Index> &l1,const event<Index> &l2) const {
        if(l1.ab.a_x(lpoints) != l2.ab.a_x(lpoints))
            return l1.ab.a_x(lpoints) < l2.ab.a_x(lpoints);
        if(l1.type != l2.type) return l1.type < l2.type;
        if(l1.ab.a != l2.ab.a) return l1.ab.a < l2.ab.a;
        return l1.ab.b < l2.ab.b;
    }
};

template<typename Index,typename Coord> class events_t {
    using points_ref = const std::pmr::vector<loop_point<Index,Coord>> &;
    using cmp = event_cmp<Index,Coord>;

    struct to_insert {
        event<Index> e;
        std::size_t i;
    };

    std::pmr::vector<event<Index>> events;
    std::pmr::vector<to_insert> new_events;
    std::ptrdiff_t current_i;
    std::size_t last_size;

    void incorporate_new() {
        std::size_t new_count = events.size() - last_size;
        for(std::size_t i=new_count, j=events.size()-1; i>0; --i) {
            while(j > new_events[i-1].i) {
                events[j] = std::move(events[j-i]);
                --j;
            }
            events[j--] = std::move(new_events[i-1].e);
        }

        new_events.resize(0);
        last_size = events.size();
    }

public:
    events_t(std::pmr::memory_resource *contig_mem)
        : events(contig_mem), new_events(contig_mem), current_i(-1), last_size(0) {}
    
    void clear() {
        events.clear();
        new_events.clear();
        current_i = -1;
        last_size = 0;
    }

    void rewind() {
        POLY_OPS_ASSERT(new_events.empty());
        current_i = -1;
    }

    event<Index> &operator[](std::size_t i) { return events[i]; }
    const event<Index> &operator[](std::size_t i) const { return events[i]; }

    event<Index> &find(points_ref points,const segment_common<Index> &s,event_type_t type,std::size_t upto) {
        auto itr = std::lower_bound(
            events.begin(),
            events.begin()+std::ptrdiff_t(upto),
            event<Index>{s,0,type},
            cmp{points});
        POLY_OPS_ASSERT(itr != (events.begin()+std::ptrdiff_t(upto)) && itr->ab == s && itr->type == type);
        return *itr;
    }

    event<Index> &find(points_ref points,const segment_common<Index> &s,event_type_t type) {
        return find(points,s,type,last_size);
    }

    std::ptrdiff_t event_ssize() const {
        return static_cast<std::ptrdiff_t>(events.size());
    }

    bool more() const {
        return (current_i+1) < event_ssize() || !new_events.empty();
    }

    std::tuple<event<Index>,bool> next(points_ref points) {
        POLY_OPS_ASSERT(current_i < event_ssize());

        if(!new_events.empty()) {
            --current_i;
            // backtrack to before the first insertion
            if(current_i >= static_cast<std::ptrdiff_t>(new_events[0].i)) return {events[std::size_t(current_i)],false};

            incorporate_new();
            POLY_OPS_ASSERT_SLOW(std::ranges::is_sorted(events,cmp{points}));
        } else if(events.size() > last_size) {
            if(last_size == 0) {
                std::ranges::sort(events,cmp{points});
                last_size = events.size();
            } else {
                /* if there are new items at the end, find their sorted position
                and move them to "new_events" */
                std::size_t new_count = events.size() - last_size;
                new_events.reserve(new_count);
                for(std::size_t i=0; i<new_count; ++i) {
                    auto itr = std::lower_bound(
                        events.begin(),
                        events.begin()+std::ptrdiff_t(last_size),
                        events[last_size+i],
                        cmp{points});
                    
                    new_events.emplace_back(std::move(events[last_size+i]),static_cast<std::size_t>(itr - events.begin()));
                }

                if(!new_events.empty()) {
                    std::ranges::sort(
                        new_events,
                        [&](const to_insert &a,const to_insert &b) {
                            if(a.i != b.i) return a.i < b.i;
                            return cmp{points}(a.e,b.e);
                        });
                    for(std::size_t i=0; i<new_events.size(); ++i) new_events[i].i += i;

                    // backtrack to before the first insertion
                    if(static_cast<std::ptrdiff_t>(new_events[0].i) <= current_i) return {current(),false};

                    incorporate_new();
                    POLY_OPS_ASSERT_SLOW(std::ranges::is_sorted(events,cmp{points}));
                }
            }
        }

        return {events[std::size_t(++current_i)],true};
    }

    void add_event(Index sa,Index sb,Index sweep_node,event_type_t t) {
        events.emplace_back(segment<Index>{sa,sb},sweep_node,t);
    }

    void add_fb_events(points_ref points,Index sa,Index sb,Index sweep_node) {
        auto f = event_type_t::forward;
        auto b = event_type_t::backward;
        if(points[sa].data[0] == points[sb].data[0]) [[unlikely]] {
            /* this is an assumption made by vert_overlap() */
            POLY_OPS_ASSERT(points[sa].next == sb);

            f = event_type_t::vforward;
            b = event_type_t::vbackward;
        }
        add_event(sa,sb,sweep_node,f);
        add_event(sb,sa,sweep_node,b);
    }

    void reserve(std::size_t amount) {
        events.reserve(amount);
    }

    auto touching_removed(points_ref points) const {
        Coord x = current().ab.a_x(points);
        return std::ranges::subrange(events.begin(),events.begin()+current_i)
            | std::views::reverse
            | std::views::filter([](const event<Index> &e) {
                return (e.type == event_type_t::backward
                    || e.type == event_type_t::vbackward)
                    && e.status != event_status_t::deleted; })
            | std::views::take_while(
                [x,&points](const event<Index> &e) { return x == e.ab.a_x(points); })
            | std::views::transform([](const event<Index> &e) { return e.line_ba(); });
    }

    auto touching_pending(points_ref points) const {
        Coord x = current().ab.a_x(points);
        return std::ranges::subrange(events.begin()+current_i+1,events.end())
            | std::views::filter([](const event<Index> &e) {
                return (e.type == event_type_t::forward
                    || e.type == event_type_t::vforward)
                    && e.status != event_status_t::deleted; })
            | std::views::take_while(
                [x,&points](const event<Index> &e) { return x == e.ab.a_x(points); })
            | std::views::transform([](const event<Index> &e) { return e.ab; });
    }

    event<Index> &current() { return events[std::size_t(current_i)]; }
    const event<Index> &current() const { return events[std::size_t(current_i)]; }

    /* these are used by graphical_test_common.hpp of the testing code */
    auto begin() const { return events.begin(); }
    auto end() const { return events.end(); }
    std::size_t size() const { return events.size(); }
};

template<typename Index,typename Coord>
using sweep_node = set_node<cached_segment<Index,Coord>,Index>;

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
    bool operator()(const cached_segment<Index,Coord> &s1,const cached_segment<Index,Coord> &s2) const {
        if(s1.pa.x() == s2.pa.x()) {
            Coord r = s1.pa.y() - s2.pa.y();
            if(r) return r > 0;
            long_coord_t<Coord> r2 = triangle_winding(s1.pb,s1.pa,s2.pb);
            if(r2) return r2 > 0;
            /* the lines are coincident and have the same start point but they
            might be pointing away from each other */
            r = s1.pb.y() - s2.pb.y();
            if(r) return r > 0;
        } else if(s1.pa.x() < s2.pa.x()) {
            if(s1.pa.x() != s1.pb.x()) {
                long_coord_t<Coord> r = triangle_winding(s1.pb,s1.pa,s2.pa);
                if(r) return r > 0;
                r = triangle_winding(s1.pb,s1.pa,s2.pb);
                if(r) return r > 0;
            }
        } else {
            if(s2.pa.x() != s2.pb.x()) {
                long_coord_t<Coord> r = triangle_winding(s2.pa,s2.pb,s1.pa);
                if(r) return r > 0;
                r = triangle_winding(s2.pa,s2.pb,s1.pb);
                if(r) return r > 0;
            }
        }

        return s1.b == s2.b ? (s1.a > s2.a) : (s1.b > s2.b);
    }
};

template<typename Index> struct line_bal_sweep_cmp {
    bool operator()(const segment_common<Index> &s1,const segment_common<Index> &s2) const {
        return s1.b == s2.b ? (s1.a > s2.a) : (s1.b > s2.b);
    }
};

template<typename Index,typename Coord>
using sweep_t = sweep_set<cached_segment<Index,Coord>,Index,sweep_cmp<Index,Coord>>;

template<typename Index,typename Coord>
using line_bal_sweep_t = sweep_set<cached_segment<Index,Coord>,Index,line_bal_sweep_cmp<Index>>;

/* This is only used for debugging */
template<typename Index,typename Coord>
bool check_integrity(
    const std::pmr::vector<loop_point<Index,Coord>> &points,
    const sweep_t<Index,Coord> &sweep)
{
    for(auto &s : sweep) {
        if(!(points[s.value.a].next == s.value.b || points[s.value.b].next == s.value.a)) return false;
    }
    return true;
}

template<typename T> concept sized_or_forward_range
    = std::ranges::sized_range<T> || std::ranges::forward_range<T>;

template<typename T,typename Coord> concept sized_or_forward_point_range
    = sized_or_forward_range<T> && point<std::ranges::range_value_t<T>,Coord>;

template<typename T,typename Coord> concept sized_or_forward_point_range_range
    /* We have to iterate over the loops to get their sizes. */
    = std::ranges::forward_range<T> && sized_or_forward_point_range<std::ranges::range_value_t<T>,Coord>;

template<typename,typename T> struct total_point_count {
    static inline size_t doit(const T&) { return 100; }
};

template<typename Coord,sized_or_forward_point_range_range<Coord> T> struct total_point_count<Coord,T> {
    static std::size_t doit(const T &x) {
        std::size_t s = 0;
        for(auto &&loop : x) s += std::ranges::distance(loop);
        return s;
    }
};

/* This is only used for debugging */
template<typename Index,typename Coord>
bool intersects_any(const sweep_node<Index,Coord> &s1,const sweep_t<Index,Coord> &sweep,std::pmr::vector<loop_point<Index,Coord>> &lpoints) {
    auto is_marked = [&](at_edge_t edge,const sweep_node<Index,Coord> &s) {
        if(edge == at_edge_t::both) {
            return lpoints[s.value.a].state() == line_state::check && lpoints[s.value.b].state() == line_state::check;
        }
        return lpoints[edge == at_edge_t::start ? s.value.a : s.value.b].state() == line_state::check;
    };
    rand_generator rgen;
    for(auto s2 : sweep) {
        point_t<Coord> intr;
        at_edge_t at_edge[2];
        if(intersects(s1.value,s2.value,intr,at_edge,rgen) &&
                (at_edge[0] == at_edge_t::no || at_edge[1] == at_edge_t::no ||
                !(is_marked(at_edge[0],s1) && is_marked(at_edge[1],s2)))) {
            POLY_OPS_DEBUG_LOG("Missed intersection between {} and {}!",s1.value,s2.value);
            return true;
        }
    }
    return false;
}

/* This is only used for debugging */
template<typename Index,typename Coord>
bool consistent_order(const sweep_t<Index,Coord> &sweep) {
    auto &cmp = sweep.node_comp();
    for(auto itr_a = sweep.begin(); itr_a != sweep.end(); ++itr_a) {
        auto itr_b = sweep.begin();
        for(; itr_b != itr_a; ++itr_b) {
            if(cmp(*itr_a,*itr_b)) return false;
        }
        if(cmp(*itr_a,*itr_b++)) return false;
        for(; itr_b != sweep.end(); ++itr_b) {
            if(cmp(*itr_b,*itr_a)) return false;
        }
    }
    return true;
}

template<typename Index> struct virt_point {
    Index next;
    bool keep;
    bool next_is_end;
};

/* After determining where lines cross, the segments are broken up, and later
rejoined to other segments.

broken_starts is a mapping of points to arrays of start points.

broken_ends is a list of points before end-points. The end-points will need to
be replaced with a start point that has the same coordinates.

If a segment needs to be reversed, it will have an extra end point (formally the
start) and a missing start point because the former end point is the start of
another segment (or the same segment if the segment is the entire loop) and
cannot be altered yet.
  +------------------+       +------------------+
  | S-->O-->O-->E--> |   =>  | S<->O<--O   E--> |
  +------------------+       +------------------+
Instead of moving any points, the extra end point is added to a list of
"orphans" and the missing start point is recorded in virtual_points. After all
segments are collected, the orphans are used to supply the missing points of
other segments.
 */
template<typename Index,typename Coord> struct breaks_t {
    std::pmr::map<point_t<Coord>,std::pmr::vector<Index>,point_less> broken_starts;
    std::pmr::vector<Index> broken_ends;
    std::pmr::map<point_t<Coord>,std::pmr::vector<virt_point<Index>>,point_less> virtual_points;
    std::pmr::vector<Index> orphans;

    breaks_t(std::pmr::memory_resource *contig_mem,std::pmr::memory_resource *discrete_mem)
        : broken_starts(discrete_mem), broken_ends(contig_mem), virtual_points(discrete_mem), orphans(contig_mem) {}
    
    void clear() {
        broken_starts.clear();
        broken_ends.clear();
        virtual_points.clear();
        orphans.clear();
    }
};

/* Follow the points connected to the point at "intr" until another intersection
is reached or we wrapped around. Along the way, update the "state()" value to
the value at "intr", and if "state()" at "intr" is not "line_state::keep" or
"line_state::keep_rev", merge all the point values into "intr" via the point
tracker.

If "state()" at "intr" is one of the "keep" values and the "state" of the next
intersection is not, add the point before the intersection to "broken_ends. If
vice-versa, add the intersecting point to "broken_starts". */
template<typename Index,typename Coord>
void follow_balance(
    std::pmr::vector<loop_point<Index,Coord>> &points,
    Index intr,
    breaks_t<Index,Coord> &breaks,
    i_point_tracker<Index> *pt)
{
    POLY_OPS_ASSERT(points[intr].has_line_bal());
    POLY_OPS_ASSERT(!(points[intr].state() & line_state::anchor));
    Index next = points[intr].next;
    Index prev = intr;
    while(!points[next].has_line_bal()) {
        Index real_next_next = points[next].next;

        points[next].state() = points[intr].state() | (points[next].state() & line_state::anchor);
        if(!(points[intr].state() & line_state::keep)) {
            if(pt) pt->point_merge(prev,next);
        } else if(points[intr].state() & line_state::reverse) {
            /* the actual points should not be moved, to make point-tracking
            simpler, except for the temporary anchor points, which must have the
            same coordinates as the (new) previous point */
            if(points[next].state() & line_state::anchor) points[next].data = end_point(points,next);
            points[next].next = prev;
        }

        prev = next;
        next = real_next_next;
    }

    bool next_is_end = false;

    switch(points[intr].state()) {
    case line_state::discard:
        if(pt) pt->point_merge(prev,next);
        break;
    case line_state::keep:
        breaks.broken_starts[points[intr].data].push_back(intr);
        breaks.broken_ends.push_back(prev);
        break;
    case line_state::keep_rev:
    case line_state::discard_rev:
        breaks.orphans.push_back(intr);

        if(points[intr].state() == line_state::keep_rev) {
            /* The new second-last point is the point after "intr". If the point
            after "intr" is the old last point, the second-last point is the new
            first point. */
            if(prev == intr) {
                // add to "broken_ends" later, when we have a new first point
                next_is_end = true;
            } else {
                breaks.broken_ends.push_back(points[intr].next);
            }
        }

        breaks.virtual_points[points[next].data].emplace_back(
            prev,
            points[intr].state() == line_state::keep_rev,
            next_is_end);

        break;
    default:
        POLY_OPS_ASSERT(false);
    }
}

template<typename Index> struct loop_location {
    Index start;
    Index size;
};

template<typename Index> struct temp_polygon {
    loop_location<Index> loop_loc;
    std::pmr::vector<temp_polygon*> children;
};

/* this also removes the "anchor" points from the loops (they're not deleted,
just unlinked), but still sets "loop_index" on the anchor points */
template<typename Index,typename Coord>
void find_loops(
    std::pmr::vector<loop_point<Index,Coord>> &lpoints,
    std::pmr::vector<temp_polygon<Index>> &loops,
    std::pmr::memory_resource *contig_mem)
{
    for(Index i=0; i<static_cast<Index>(lpoints.size()); ++i) {
        if(lpoints[i].aux.loop_index == L_INDEX_UNSET) {
            int loop_i = static_cast<int>(loops.size());
            loops.emplace_back(
                loop_location<Index>{i,1},
                std::pmr::vector<temp_polygon<Index>*>(contig_mem));

            /* prevent this point from being scanned again */
            lpoints[i].aux.loop_index = loop_i;

            Index prev = i; // previous non-anchor point
            for(Index j = lpoints[i].next; j != i; j = lpoints[j].next) {
                POLY_OPS_ASSERT(lpoints[j].aux.loop_index == L_INDEX_UNSET || lpoints[j].aux.loop_index == L_INDEX_ANCHOR);

                if(lpoints[j].aux.loop_index == L_INDEX_ANCHOR) {
                    lpoints[prev].next = lpoints[j].next;
                } else {
                    ++loops.back().loop_loc.size;
                    prev = j;
                }
                lpoints[j].aux.loop_index = loop_i;
            }
        }
    }
}

template<typename Index>
struct intr_t {
    Index p;
    std::pmr::vector<Index> hits;
};

template<typename Index>
using intr_array_t = std::pmr::vector<intr_t<Index>>;


/* Take an array of intr_t instances ("samples") where each "p" member refers to
a point, and create a new array with the same data, except the "p" member refers
to a loop. If the item doesn't refer to a loop, it's not included in the new
array. The new array is then sorted by loop index and deduplicated, so that only
the first item for a given loop index is kept.

The "hits" array is moved, not copied to the corresponding intr_t instance in
the new array, thus the items of "samples" cannot be use after this function is
called. */
template<typename Index,typename Coord>
intr_array_t<Index> unique_sorted_loop_points(
    const std::pmr::vector<loop_point<Index,Coord>> &lpoints,
    intr_array_t<Index> &samples,
    std::pmr::memory_resource *contig_mem)
{
    intr_array_t<Index> ordered_loops(contig_mem);
    ordered_loops.reserve(samples.size());
    for(auto &item : samples) {
        if(lpoints[item.p].aux.loop_index >= 0) {
            ordered_loops.emplace_back(item.p,std::move(item.hits));
        }
    }

    auto loop_id = [&](const intr_t<Index> &item){
        return lpoints[item.p].aux.loop_index;
    };
    std::ranges::sort(ordered_loops,{},loop_id);
    ordered_loops.resize(static_cast<size_t>(
        std::ranges::begin(std::ranges::unique(ordered_loops,{},loop_id)) - ordered_loops.begin()));

    return ordered_loops;
}

auto end_point(const auto &points,auto i) noexcept {
    return points[points[i].next].data;
}
auto non_anchor_end_point(const auto &points,auto i) {
    auto *p = &points[points[i].next];
    if(p->state() & line_state::anchor) p = &points[p->next];
    POLY_OPS_ASSERT(!(p->state() & line_state::anchor));
    return p->data;
}

/* replace the line indices in the "hits" member of each ordered_loops instance,
with loop indices and remove the duplicates */
template<typename Index,typename Coord>
void replace_line_indices_with_loop_indices(
    const std::pmr::vector<loop_point<Index,Coord>> &lpoints,
    const std::pmr::vector<temp_polygon<Index>> &loops,
    intr_array_t<Index> &ordered_loops,
    std::pmr::memory_resource *contig_mem)
{
    POLY_OPS_DEBUG_LOG("LOOP MEMBERSHIP:");

    std::pmr::vector<int> inside(loops.size(),0,contig_mem);
    for(auto& item : ordered_loops) {
        int item_loop_i = lpoints[item.p].aux.loop_index;
        /* calculate the winding number of every loop for this point */
        for(Index i : item.hits) {
            // if the index is negative, the line was discarded
            if(lpoints[i].aux.loop_index < 0
                || lpoints[i].aux.loop_index == item_loop_i) continue;

            POLY_OPS_ASSERT(static_cast<std::size_t>(lpoints[i].aux.loop_index) < loops.size());
            point_t<Coord> d = lpoints[i].data - end_point(lpoints,i);
            POLY_OPS_ASSERT(d.x() != Coord(0) || d.y() != Coord(0));
            inside[static_cast<std::size_t>(lpoints[i].aux.loop_index)]
                += ((d.x() ? d.x() : d.y()) > 0) ? 1 : -1;
        }

        POLY_OPS_DEBUG_LOG("{}: {}",
            item.p,
            delimited(std::views::iota(std::size_t(0),inside.size()) | std::views::filter([&](auto i) { return inside[i] != 0; })));

        item.hits.clear();
        for(std::size_t i=0; i < inside.size(); ++i) {
            if(inside[i]) item.hits.emplace_back(static_cast<Index>(i));
        }

        std::ranges::fill(inside,0);

        item.p = static_cast<Index>(item_loop_i);
    }
}

template<typename Index>
std::pmr::vector<temp_polygon<Index>*> arrange_loops(
    const intr_array_t<Index> &ordered_loops,
    std::pmr::vector<temp_polygon<Index>> &loops,
    std::pmr::vector<temp_polygon<Index>*> &top)
{
    POLY_OPS_ASSERT(std::ranges::equal(
        ordered_loops | std::views::transform([](auto &item) { return item.p; }),
        std::ranges::iota_view<Index,Index>(0,loops.size())));

    for(const auto& item : ordered_loops) {
        if(item.hits.empty()) top.push_back(&loops[item.p]);
        else {
            for(Index outer_i : item.hits) {
                if(ordered_loops[outer_i].hits.size() == (item.hits.size() - 1)) {
                    loops[outer_i].children.push_back(&loops[item.p]);
                    goto next;
                }
            }
            POLY_OPS_ASSERT(false);
          next: ;
        }
    }

    return top;
}

/* after calling this function, the items of "samples" are left in an
unspecified state due to the use of std::move */
template<typename Index,typename Coord>
void loop_hierarchy(
    std::pmr::vector<loop_point<Index,Coord>> &lpoints,
    std::pmr::vector<temp_polygon<Index>> &loops,
    std::pmr::vector<temp_polygon<Index>*> &top,
    intr_array_t<Index> &samples,
    std::pmr::memory_resource *contig_mem)
{
    auto ordered_loops = unique_sorted_loop_points<Index,Coord>(lpoints,samples,contig_mem);
    replace_line_indices_with_loop_indices<Index,Coord>(lpoints,loops,ordered_loops,contig_mem);
    arrange_loops<Index>(ordered_loops,loops,top);
}

} // namespace detail

template<typename Coord,typename Index=std::size_t,typename Tracker=null_tracker<Coord,Index>> class temp_polygon_proxy;

namespace detail {

template<typename Tracker,typename LPoints,bool IsPtr> struct tracked_points {
    [[no_unique_address]] Tracker _tracker;
    LPoints lpoints;

    auto tracker() const noexcept {
        if constexpr(IsPtr) { return _tracker; }
        else { return &_tracker; }
    }
};
template<typename Tracker,typename LPoints> auto make_tracked_points(Tracker &&tracker,LPoints &&lpoints) {
    if constexpr(std::is_lvalue_reference_v<Tracker>) {
        return tracked_points<decltype(&tracker),LPoints,true>{&tracker,std::forward<LPoints>(lpoints)};
    } else {
        return tracked_points<Tracker,LPoints,false>{std::move(tracker),std::forward<LPoints>(lpoints)};
    }
}

template<typename I,typename F>
struct iterator_facade {
    using value_type = std::invoke_result_t<F,std::iter_value_t<I>>;

    I base;
    F *fun;

    iterator_facade &operator++() noexcept {
        ++base;
        return *this;
    }
    iterator_facade operator++(int) noexcept {
        return {base++,fun};
    }
    iterator_facade &operator--() noexcept {
        --base;
        return *this;
    }
    iterator_facade operator--(int) noexcept {
        return {base--,fun};
    }

    iterator_facade &operator+=(std::iter_difference_t<I> n) noexcept {
        base += n;
        return *this;
    }
    iterator_facade &operator-=(std::iter_difference_t<I> n) noexcept {
        base -= n;
        return *this;
    }
    iterator_facade operator+(std::iter_difference_t<I> n) const noexcept {
        return {base+n,fun};
    }
    iterator_facade operator-(std::iter_difference_t<I> n) const noexcept {
        return {base-n,fun};
    }
    std::iter_difference_t<I> operator-(const iterator_facade &b) const noexcept {
        return base - b.base;
    }
    value_type operator[](std::iter_difference_t<I> n) const noexcept {
        return (*fun)(base[n]);
    }
    value_type operator*() const noexcept {
        return (*fun)(*base);
    }
    auto operator<=>(const iterator_facade &b) const noexcept {
        return base <=> b.base;
    }
    bool operator==(const iterator_facade &b) const noexcept {
        return base == b.base;
    }
    friend iterator_facade operator+(std::iter_difference_t<I> a,const iterator_facade &b) noexcept {
        return {a + b.base,b.fun};
    }
};

/* this is basically std::ranges::transform_view except it supports r-value
references on GCC 11 */
template<typename R,typename F> struct range_facade {
    static constexpr bool const_executable = std::is_invocable_v<const F,std::ranges::range_value_t<R>>;

    R base;
    F fun;

    template<typename T,typename U> range_facade(T &&base,U &&fun)
        : base{std::forward<T>(base)}, fun{std::forward<U>(fun)} {}
    
    range_facade(range_facade&&) = default;

    std::size_t size() const noexcept {
        return std::ranges::size(base);
    }
    bool empty() const noexcept { return std::ranges::empty(base); }

    auto begin() noexcept { return iterator_facade<std::ranges::iterator_t<R>,F>{std::ranges::begin(base),&fun}; }
    auto end() noexcept { return iterator_facade<std::ranges::iterator_t<R>,F>{std::ranges::end(base),&fun}; }

    auto begin() const noexcept requires const_executable {
        return iterator_facade<std::ranges::iterator_t<const R>,const F>{std::ranges::begin(base),&fun};
    }
    auto end() const noexcept requires const_executable {
        return iterator_facade<std::ranges::iterator_t<const R>,const F>{std::ranges::end(base),&fun};
    }

    auto operator[](auto n) -> decltype(fun(base[n])) {
        return fun(base[n]);
    }
    auto operator[](auto n) const -> decltype(fun(base[n])) {
        return fun(base[n]);
    }
};

template<typename R,typename F> range_facade(R &&base,F &&fun)
    -> range_facade<R,F>;


template<typename Index,typename Coord,typename Tracker>
auto make_temp_polygon_tree_range(
    const loop_point<Index,Coord> *lpoints,
    std::type_identity_t<std::span<temp_polygon<Index>* const>> top,
    Tracker &&tracker)
{
    return range_facade(
        std::move(top),
        [tl=make_tracked_points(std::forward<Tracker>(tracker),std::move(lpoints))]
        (const temp_polygon<Index> *poly) {
            return temp_polygon_proxy<Coord,Index,Tracker>(tl.lpoints,*poly,tl.tracker());
        });
}

} // namespace detail

/**
 * An opaque type that models `std::forward_iterator`. The iterator yields
 * instances of `point_t<Coord>`.
 */
template<typename Coord,typename Index=std::size_t,typename Tracker=null_tracker<Coord,Index>> class proto_loop_iterator {
public:
    using tracker_ptr = decltype(&std::declval<Tracker&>());
    using value_type = decltype(std::declval<Tracker>().get_value(std::declval<Index>(),std::declval<point_t<Coord>>()));

private:
    friend class temp_polygon_proxy<Coord,Index,Tracker>;

    [[no_unique_address]] tracker_ptr tracker;
    const detail::loop_point<Index,Coord> *lpoints;
    Index i;

    /* Loops have the same "i" value for "begin()" and "end()" thus a different
    value is used to distinguish the end from other iterators. */
    Index dist_from_end;

    

    proto_loop_iterator(const detail::loop_point<Index,Coord> *lpoints,Index i,Index dist_from_end,tracker_ptr tracker) noexcept
        : tracker{tracker}, lpoints{lpoints}, i{i}, dist_from_end{dist_from_end} {}

public:
    proto_loop_iterator &operator++() noexcept {
        i = lpoints[i].next;
        --dist_from_end;
        return *this;
    }

    proto_loop_iterator operator++(int) noexcept {
        return {lpoints,std::exchange(i,lpoints[i].next),dist_from_end--};
    }

    value_type operator*() const noexcept(std::is_nothrow_copy_constructible_v<Coord>) {
        return (*tracker).get_value(i,lpoints[i].data);
    }

    friend bool operator==(const proto_loop_iterator &a,const proto_loop_iterator &b) noexcept {
        assert(a.lpoints == b.lpoints);
        return a.dist_from_end == b.dist_from_end;
    }

    friend std::ptrdiff_t operator-(const proto_loop_iterator &a,const proto_loop_iterator &b) noexcept {
        assert(a.lpoints == b.lpoints);
        return static_cast<std::ptrdiff_t>(b.dist_from_end) - static_cast<std::ptrdiff_t>(a.dist_from_end);
    }

    Index index() const noexcept { return i; }

    proto_loop_iterator() noexcept = default;
    proto_loop_iterator(const proto_loop_iterator&) noexcept = default;
};


/**
 * A representation of a polygon with zero or more child polygons.
 *
 * This class is not meant to be directly instantiated by users of this
 * library. This class models `std::ranges::forward_range` and
 * `std::ranges::sized_range` and yields instances of `point_t<Coord>`.
 */
template<typename Coord,typename Index,typename Tracker>
class temp_polygon_proxy : public std::ranges::view_interface<temp_polygon_proxy<Coord,Index,Tracker>> {
public:
    using tracker_ptr = decltype(&std::declval<Tracker&>());

private:
    const detail::loop_point<Index,Coord> *lpoints;
    const detail::temp_polygon<Index> &data;
    [[no_unique_address]] tracker_ptr tracker;
public:
    temp_polygon_proxy(
        const detail::loop_point<Index,Coord> *lpoints,
        const detail::temp_polygon<Index> &data,
        tracker_ptr tracker)
        : lpoints{lpoints}, data{data}, tracker{tracker} {}

    /** Get the iterator to the first element. */
    proto_loop_iterator<Coord,Index,Tracker> begin() const noexcept {
        return {lpoints,data.loop_loc.start,data.loop_loc.size,tracker};
    }

    /** Get the end iterator */
    proto_loop_iterator<Coord,Index,Tracker> end() const noexcept {
        return {lpoints,data.loop_loc.start,0,tracker};
    }

    /** Return the number of elements in this range. */
    Index size() const noexcept { return data.loop_loc.size; }

    /** Return a range representing the children of this polygon. */
    auto inner_loops() const noexcept {
        return detail::make_temp_polygon_tree_range<Index,Coord,Tracker>(lpoints,data.children,*tracker);
    }
};

} // namespace poly_ops

template<typename Coord,typename Index,typename Tracker>
inline constexpr bool std::ranges::enable_borrowed_range<poly_ops::temp_polygon_proxy<Coord,Index,Tracker>> = true;

namespace poly_ops {

namespace detail {

template<typename Index,typename Coord,typename Tracker> auto make_temp_polygon_tree_range(
    std::pmr::vector<loop_point<Index,Coord>> &&lpoints,
    std::pmr::vector<temp_polygon<Index>> &&loops,
    std::pmr::vector<temp_polygon<Index>*> &&top,
    Tracker &&tracker)
{
    /* "loops" isn't used directly but is referenced by "top" */
    return range_facade(
        std::move(top),
        [tl=make_tracked_points(std::forward<Tracker>(tracker),std::move(lpoints)),loops=std::move(loops)]
        (const temp_polygon<Index> *poly) {
            return temp_polygon_proxy<Coord,Index,Tracker>(tl.lpoints.data(),*poly,tl.tracker());
        });
}

template<typename Index,typename Coord,typename Tracker> auto make_temp_polygon_range(
    std::pmr::vector<loop_point<Index,Coord>> &&lpoints,
    std::pmr::vector<temp_polygon<Index>> &&loops,
    Tracker &&tracker)
{
    return range_facade(
        std::move(loops),
        [tl=make_tracked_points(std::forward<Tracker>(tracker),std::move(lpoints))]
        (const temp_polygon<Index> &poly) {
            return temp_polygon_proxy<Coord,Index,Tracker>(tl.lpoints.data(),poly,tl.tracker());
        });
}

template<typename Index,typename Coord,typename Tracker> auto make_temp_polygon_range(
    const std::pmr::vector<loop_point<Index,Coord>> &lpoints,
    const std::pmr::vector<temp_polygon<Index>> &loops,
    Tracker &&tracker)
{
    return range_facade(
        loops,
        [tl=make_tracked_points(std::forward<Tracker>(tracker),lpoints.data())]
        (const temp_polygon<Index> &poly) {
            return temp_polygon_proxy<Coord,Index,Tracker>(tl.lpoints,poly,tl.tracker());
        });
}


template<typename T> concept sized_forward_range =
    std::ranges::forward_range<T> && std::ranges::sized_range<T>;

} // namespace detail


/** An opaque type that models `std::ranges::forward_range` and
 * `std::ranges::sized_range`.
 *
 * Unlike `temp_polygon_tree_range`, an instance of this type does not own its
 * data.
 */
template<typename Coord,typename Index=std::size_t,typename Tracker=null_tracker<Coord,Index>>
using borrowed_temp_polygon_tree_range = decltype(
    detail::make_temp_polygon_tree_range<Index,Coord,Tracker>(
        std::declval<detail::loop_point<Index,Coord>*>(),
        std::declval<std::span<detail::temp_polygon<Index>*>>(),
        std::declval<Tracker>()));

/**
 * An opaque type that models `std::ranges::forward_range` and
 * `std::ranges::sized_range`.
 *
 * Unlike `temp_polygon_range`, an instance of this type does not own its data.
 */
template<typename Coord,typename Index=std::size_t,typename Tracker=null_tracker<Coord,Index>>
using borrowed_temp_polygon_range = decltype(
    detail::make_temp_polygon_range<Index,Coord,Tracker>(
        std::declval<const std::pmr::vector<detail::loop_point<Index,Coord>>&>(),
        std::declval<const std::pmr::vector<detail::temp_polygon<Index>>&>(),
        std::declval<Tracker>()));

/**
 * An opaque type that models `std::ranges::forward_range` and
 * `std::ranges::sized_range`.
 */
template<typename Coord,typename Index=std::size_t,typename Tracker=null_tracker<Coord,Index>>
using temp_polygon_tree_range = decltype(
    detail::make_temp_polygon_tree_range<Index,Coord>({},{},{},std::declval<Tracker>()));

/**
 * An opaque type that models `std::ranges::forward_range` and
 * `std::ranges::sized_range`.
 */
template<typename Coord,typename Index=std::size_t,typename Tracker=null_tracker<Coord,Index>>
using temp_polygon_range = decltype(
    detail::make_temp_polygon_range<Index,Coord>({},{},std::declval<Tracker>()));


/**
 * A class for performing boolean clipping operations.
 *
 * An instance of `clipper` will reuse its allocated memory for subsequent
 * operations, making it more efficient than calling `boolean_op` for performing
 * multiple operations.
 */
template<coordinate Coord,std::integral Index=std::size_t> class clipper {
    static_assert(
        std::ranges::random_access_range<temp_polygon_tree_range<Coord,Index>>
        && std::ranges::random_access_range<temp_polygon_range<Coord,Index>>
        && std::ranges::random_access_range<borrowed_temp_polygon_tree_range<Coord,Index>>
        && std::ranges::random_access_range<borrowed_temp_polygon_range<Coord,Index>>
        && detail::sized_forward_range<temp_polygon_proxy<Coord,Index,null_tracker<Coord,Index>>>);

    std::pmr::memory_resource *contig_mem;
    std::pmr::unsynchronized_pool_resource discrete_mem;

    std::pmr::vector<detail::loop_point<Index,Coord>> lpoints;
    detail::intr_array_t<Index> samples;
    detail::events_t<Index,Coord> events;

    /* The first value in this array is reserved for the red black tree in
    "sweep_set" */
    std::pmr::vector<detail::sweep_node<Index,Coord>> sweep_nodes;

    detail::breaks_t<Index,Coord> breaks;
    detail::rand_generator rgen;
    Index original_i;
    bool ran;

    std::pmr::vector<detail::temp_polygon<Index>> loops_out;
    std::pmr::vector<detail::temp_polygon<Index>*> top;

    void self_intersection(i_point_tracker<Index> *pt);
    void calc_line_bal(bool_op op);
    Index split_segment(
        detail::sweep_t<Index,Coord> &sweep,
        Index s,
        const point_t<Coord> &c,
        detail::at_edge_t at_edge,
        i_point_tracker<Index> *pt);
    bool check_intersection(
        detail::sweep_t<Index,Coord> &sweep,
        Index s1,
        Index s2,
        i_point_tracker<Index> *pt);
    void add_fb_events(Index sa,Index sb);
    Index insert_anchor_point(Index i);
    void do_op(bool_op op,i_point_tracker<Index> *pt,bool TreeOut);

    template<point_range<Coord> R> void _add_loop(R &&loop,bool_set cat,i_point_tracker<Index> *pt) {
        auto sink = add_loop(cat,pt);
        for(point_t<Coord> p : loop) sink(p,original_i + 1);
    }

public:
    class point_sink {
        friend class clipper;

        clipper &n;
        i_point_tracker<Index> *pt;
        point_t<Coord> prev;
        Index first_i;
        bool_set cat;
        bool started;

        point_sink(clipper &n,bool_set cat,i_point_tracker<Index> *pt) : n{n}, pt{pt}, cat{cat}, started{false} {}
        point_sink(const point_sink&) = delete;
        point_sink(point_sink &&b) : n{b.n}, pt{b.pt}, prev{b.prev}, first_i{b.first_i}, cat{b.cat}, started{b.started} {
            b.started = false;
        }

    public:
        void operator()(const point_t<Coord> &p,Index orig_i=0);
        Index last_orig_i() const { return n.original_i; }
        Index &last_orig_i() { return n.original_i; }
        ~point_sink();
    };

    explicit clipper(std::pmr::memory_resource *_contig_mem=nullptr) :
        contig_mem(_contig_mem == nullptr ? std::pmr::get_default_resource() : _contig_mem),
        discrete_mem(contig_mem),
        lpoints(contig_mem),
        samples(contig_mem),
        events(contig_mem),
        breaks(contig_mem,&discrete_mem),
        original_i(static_cast<Index>(-1)),
        ran(false),
        loops_out(contig_mem),
        top(contig_mem)
    {
        sweep_nodes.emplace_back();
    }

    clipper(clipper &&b) = default;

    /**
     * Add input polygons.
     *
     * The output returned by `execute` is invalidated.
     *
     * `R` must satisfy `point_range_or_range_range`.
     */
    template<typename R> void add_loops(R &&loops,bool_set cat,i_point_tracker<Index> *pt=nullptr) {
        static_assert(point_range_or_range_range<R,Coord>);

        if constexpr(point_range_range<R,Coord>) {
            for(auto &&loop : loops) _add_loop(std::forward<decltype(loop)>(loop),cat,pt);
        } else {
            _add_loop(std::forward<R>(loops),cat,pt);
        }
    }

    /**
     * Add input subject polygons.
     *
     * The output returned by `execute` is invalidated.
     *
     * `R` must satisfy `point_range_or_range_range`.
     */
    template<typename R> void add_loops_subject(R &&loops,i_point_tracker<Index> *pt=nullptr) {
        static_assert(point_range_or_range_range<R,Coord>);
        add_loops(std::forward<R>(loops),bool_set::subject,pt);
    }

    /**
     * Add input clip polygons.
     *
     * The output returned by `execute` is invalidated.
     *
     * `R` must satisfy `point_range_or_range_range`.
     */
    template<typename R> void add_loops_clip(R &&loops,i_point_tracker<Index> *pt=nullptr) {
        static_assert(point_range_or_range_range<R,Coord>);
        add_loops(std::forward<R>(loops),bool_set::clip,pt);
    }

    /**
     * Return a "point sink".
     *
     * This is an alternative to adding loops with ranges. The return value is a
     * functor that allows adding one point at a time. The destructor of the
     * return value must be called before any method of this instance of
     * `clipper` is called.
     *
     * The output returned by `execute` is invalidated.
     */
    point_sink add_loop(bool_set cat,i_point_tracker<Index> *pt=nullptr) {
        if(ran) reset();
        return {*this,cat,pt};
    }

    /**
     * Discard all polygons added so far.
     *
     * The output returned by `execute` is invalidated.
     */
    void reset();

    /**
     * Perform a boolean operation and return the result.
     *
     * After calling this function, all the input is consumed. To perform
     * another operation, polygons must be added again.
     *
     * The output of this function has references to data in this instance of
     * `clipper`. The returned range is invalidated if the instance is destroyed
     * or the next time a method of this instance is called. This means the
     * return value cannot be fed directly back into the same instance of
     * `clipper`. To keep the data, make a copy. The data is also not stored
     * sequentially in memory.
     */
    template<bool TreeOut,typename Tracker>
    std::conditional_t<TreeOut,
        borrowed_temp_polygon_tree_range<Coord,Index,Tracker>,
        borrowed_temp_polygon_range<Coord,Index,Tracker>>
    execute(bool_op op,Tracker &&tracker) &;

    /**
     * Perform a boolean operation and return the result.
     *
     * After calling this function, all the input is consumed. To perform
     * another operation, polygons must be added again.
     *
     * The output of this function has references to data in this instance of
     * `clipper`. The returned range is invalidated if the instance is destroyed
     * or the next time a method of this instance is called. This means the
     * return value cannot be fed directly back into the same instance of
     * `clipper`. To keep the data, make a copy. The data is also not stored
     * sequentially in memory.
     */
    template<bool TreeOut> auto execute(bool_op op) & {
        return execute<TreeOut,null_tracker<Coord,Index>>(op,{});
    }

    /**
     * Perform a boolean operation and return the result.
     */
    template<bool TreeOut,typename Tracker>
    std::conditional_t<TreeOut,
        temp_polygon_tree_range<Coord,Index,Tracker>,
        temp_polygon_range<Coord,Index,Tracker>>
    execute(bool_op op,Tracker &&tracker) &&;

    /**
     * Perform a boolean operation and return the result.
     */
    template<bool TreeOut> auto execute(bool_op op) && {
        return execute<TreeOut,null_tracker<Coord,Index>>(op,{});
    }

    Index &last_orig_i() noexcept { return original_i; }
    Index last_orig_i() const noexcept { return original_i; }
};

template<coordinate Coord,std::integral Index>
void clipper<Coord,Index>::point_sink::operator()(const point_t<Coord> &p,Index orig_i) {
    if(started) [[likely]] {
        if(prev == n.lpoints.back().data) goto skip;
    } else {
        prev = p;
        first_i = static_cast<Index>(n.lpoints.size());
        started = true;
        n.original_i = orig_i;
    }

    /* Normally, points aren't added until this is called with the next point or
    the destructor is called, but duplicate points aren't added anyway and
    adding it on the first call means the "prev != n.lpoints.back().data" checks
    above and in the destructor are always safe. */
    n.lpoints.emplace_back(prev,static_cast<Index>(n.lpoints.size()+1),cat);
    if(pt) pt->point_added(n.original_i);

skip:
    prev = p;
    n.original_i = orig_i;
}
template<coordinate Coord,std::integral Index>
clipper<Coord,Index>::point_sink::~point_sink() {
    if(started) [[likely]] {
        if(prev != n.lpoints.back().data && prev != n.lpoints[first_i].data) [[likely]] {
            n.lpoints.emplace_back(prev,Index(0),cat);
            if(pt) pt->point_added(n.original_i);
        }

        /* This code doesn't work with polygons with fewer than three points. A
        polygon with one point has no effect anyway. A polygon with two points
        almost works, except the two lines have the same end-point indices. The
        sweep sets expect line segments with unique indices. */
        std::size_t new_points = n.lpoints.size() - static_cast<std::size_t>(first_i);
        if(new_points < 3) [[unlikely]] {
            while(new_points-- > 0) {
                if(pt) pt->points_removed(static_cast<Index>(new_points));
                n.lpoints.pop_back();
            }
        } else {
            n.lpoints.back().next = first_i;
            n.lpoints.back().state() = detail::line_state::check;
        }
    }
}

template<coordinate Coord,std::integral Index>
void clipper<Coord,Index>::add_fb_events(Index sa,Index sb) {
    events.add_fb_events(lpoints,sa,sb,static_cast<Index>(sweep_nodes.size()));
    sweep_nodes.emplace_back(detail::cached_segment<Index,Coord>(sa,sb,lpoints));
}

template<coordinate Coord,std::integral Index>
Index clipper<Coord,Index>::split_segment(
    detail::sweep_t<Index,Coord> &sweep,
    Index s,
    const point_t<Coord> &c,
    detail::at_edge_t at_edge,
    i_point_tracker<Index> *pt)
{
    using namespace detail;

    POLY_OPS_ASSERT(at_edge != at_edge_t::both);

    Index sa = sweep_nodes[s].value.a;
    Index sb = sweep_nodes[s].value.b;

    if(at_edge != at_edge_t::no) {
        return at_edge == at_edge_t::start ? sa : sb;
    }
    sweep.erase(s);
    sweep.init_node(s);
    events.find(
        lpoints,
        sweep_nodes[s].value,
        lpoints[sa].data[0] == lpoints[sb].data[0] ? event_type_t::vforward : event_type_t::forward).status = event_status_t::deleted;

    POLY_OPS_ASSERT(lpoints[sa].data != c && lpoints[sb].data != c);

    if(!segment<Index>(sa,sb).a_is_main(lpoints)) std::swap(sa,sb);

    Index mid = static_cast<Index>(lpoints.size());
    lpoints[sa].next = mid;
    lpoints.emplace_back(c,sb,lpoints[sa].bset());
    POLY_OPS_ASSERT(lpoints[sa].bset() == lpoints[sb].bset());
    POLY_OPS_ASSERT_SLOW(check_integrity(lpoints,sweep));

    if(pt) pt->new_point_between(sa,sb);

    /* even if the original line was not vertical, one of the new lines might
    be vertical after rounding */
    if(lpoints[sa].data[0] <= lpoints[mid].data[0]) {
        add_fb_events(sa,mid);
    } else {
        add_fb_events(mid,sa);
    }
    if(lpoints[mid].data[0] <= lpoints[sb].data[0]) {
        add_fb_events(mid,sb);
    } else {
        add_fb_events(sb,mid);
    }

    return mid;
}

template<coordinate Coord,std::integral Index>
bool clipper<Coord,Index>::check_intersection(
    detail::sweep_t<Index,Coord> &sweep,
    Index s1,
    Index s2,
    i_point_tracker<Index> *pt)
{
    using namespace detail;

    point_t<Coord> intr;
    at_edge_t at_edge[2];
    if(intersects(sweep_nodes[s1].value,sweep_nodes[s2].value,intr,at_edge,rgen)) {
        POLY_OPS_ASSERT((at_edge[0] == at_edge_t::both) == (at_edge[1] == at_edge_t::both));

        POLY_OPS_DEBUG_LOG("checking intersection of {} and {}: {}",sweep_nodes[s1].value,sweep_nodes[s2].value,intr);

        Index intr1,intr2;

        if(at_edge[0] == at_edge_t::no || at_edge[1] == at_edge_t::no) [[likely]] {
            intr1 = split_segment(sweep,s1,intr,at_edge[0],pt);
            intr2 = split_segment(sweep,s2,intr,at_edge[1],pt);

            lpoints[intr1].state() = line_state::check;
            lpoints[intr2].state() = line_state::check;

            return true;
        } else {
            if(at_edge[0] == at_edge_t::both) [[unlikely]] {
                auto &s1_seg = sweep_nodes[s1].value;
                auto &s2_seg = sweep_nodes[s2].value;
                lpoints[s1_seg.a].state() = line_state::check;
                lpoints[s1_seg.b].state() = line_state::check;
                lpoints[s2_seg.a].state() = line_state::check;
                lpoints[s2_seg.b].state() = line_state::check;
            } else {
                intr1 = at_edge[0] == at_edge_t::start ? sweep_nodes[s1].value.a : sweep_nodes[s1].value.b;
                intr2 = at_edge[1] == at_edge_t::start ? sweep_nodes[s2].value.a : sweep_nodes[s2].value.b;

                lpoints[intr1].state() = line_state::check;
                lpoints[intr2].state() = line_state::check;
            }
        }
    } else {
        POLY_OPS_DEBUG_LOG("checking intersection of {} and {}: none",sweep_nodes[s1].value,sweep_nodes[s2].value);
    }

    return false;
}

/* This is a modified version of the Bentley–Ottmann algorithm. Lines are broken
at intersections. */
template<coordinate Coord,std::integral Index>
void clipper<Coord,Index>::self_intersection(i_point_tracker<Index> *pt) {
    using namespace detail;

    events.reserve(lpoints.size()*2 + 10);

    for(Index i=0; i<static_cast<Index>(lpoints.size()); ++i) {
        Index j1 = i;
        Index j2 = lpoints[i].next;
        
        /* in the case of vertical line segments, the points must retain their
        original order due to an assumption made by vert_overlap() */
        if(lpoints[j1].data[0] > lpoints[j2].data[0]) std::swap(j1,j2);
        add_fb_events(j1,j2);
    }

    sweep_t<Index,Coord> sweep(sweep_nodes);

    /* A sweep is done over the points. We travel from point to point from left
    to right, of each line segment. If the point is on the left side of the
    segment, the segment is added to "sweep". If the point is on the right, the
    segment is removed from "sweep". Adding to "sweep" always takes priority for
    vertical lines, otherwise removing takes priority.

    In the Bentley–Ottmann algorithm, we also have to swap the order of lines in
    "sweep" as we pass intersection points, but here we split lines at
    intersection points and create new "forward" and "backward" events instead.
    */
    while(events.more()) {
        POLY_OPS_DEBUG_ITERATION

        auto [e,forward] = events.next(lpoints);

        if(e.status == event_status_t::deleted) continue;

        POLY_OPS_ASSERT_SLOW(std::ranges::all_of(sweep,[&,e=e](auto s) {
            return e.ab.a_x(lpoints) <= s.value.pa.x() || e.ab.a_x(lpoints) <= s.value.pb.x();
        }));
        POLY_OPS_ASSERT_SLOW(consistent_order(sweep));

        switch(e.type) {
        case event_type_t::forward:
        case event_type_t::vforward:
            if(forward) {
                auto [itr,inserted] = sweep.insert(e.sweep_node);
                POLY_OPS_ASSERT(inserted);

                POLY_OPS_DEBUG_LOG("FORWARD {} at {}",e.ab,lpoints[e.ab.a].data);

                if(itr != sweep.begin() && check_intersection(sweep,e.sweep_node,std::prev(itr).index(),pt)) continue;
                ++itr;
                if(itr != sweep.end()) check_intersection(sweep,e.sweep_node,itr.index(),pt);
            } else {
                sweep.erase(e.sweep_node);
                POLY_OPS_DEBUG_LOG("UNDO FORWARD {}",e.ab);
            }
            break;
        case event_type_t::backward:
        case event_type_t::vbackward:
            if(forward) {
                // if it's not in here, the line was split and no longer exists
                if(!sweep.unique(e.sweep_node)) {
                    auto itr = sweep.erase(e.sweep_node);

                    POLY_OPS_DEBUG_LOG("BACKWARD {} at {}",e.ab,lpoints[e.ab.a].data);

                    if(itr != sweep.end() && itr != sweep.begin()) {
                        check_intersection(sweep,std::prev(itr).index(),itr.index(),pt);
                    }

                    POLY_OPS_ASSERT_SLOW(!intersects_any(sweep_nodes[e.sweep_node],sweep,lpoints));
                } else {
                    events.current().status = event_status_t::deleted;
                }
            } else {
                sweep.insert(e.sweep_node);
                POLY_OPS_DEBUG_LOG("UNDO BACKWARD {}",e.ab);
            }
            break;
        }
    }
}

template<coordinate Coord,std::integral Index>
void clipper<Coord,Index>::calc_line_bal(bool_op op) {
    using namespace detail;

    line_bal_sweep_t<Index,Coord> sweep(sweep_nodes);

    auto scan = [&,this](Index i) {
        if(lpoints[i].state() == line_state::check) {
            samples.emplace_back(i,std::pmr::vector<Index>(contig_mem));
            line_balance<Index,Coord> lb{lpoints.data(),i,samples.back().hits};
            for(const sweep_node<Index,Coord> &s : sweep) lb.check(s.value);
            for(segment<Index> s : events.touching_removed(lpoints)) lb.check(s);
            for(segment<Index> s : events.touching_pending(lpoints)) lb.check(s);
            lpoints[i].state() = lb.result(op);

            POLY_OPS_DEBUG_LOG("LINE TEST at {}: {}",i,delimited(samples.back().hits));
        }
    };

    events.rewind();

    /* Another simpler sweep is done */
    while(events.more()) {

        auto [e,forward] = events.next(lpoints);
        POLY_OPS_ASSERT(forward);

        if(e.status == event_status_t::deleted) continue;

        switch(e.type) {
        case event_type_t::forward:
        case event_type_t::vforward:
            sweep.insert(e.sweep_node);
            scan(e.ab.a);
            break;
        case event_type_t::backward:
        case event_type_t::vbackward:
            scan(e.ab.a);
            sweep.erase(e.sweep_node);
            break;
        }
    }
}

template<coordinate Coord,std::integral Index>
Index clipper<Coord,Index>::insert_anchor_point(Index i) {
    if(!(lpoints[lpoints[i].next].state() & detail::line_state::anchor)) {
        Index new_i = Index(lpoints.size());
        lpoints.push_back(lpoints[i]);
        lpoints.back().state() = detail::line_state::anchor_undef;
        lpoints[i].next = new_i;

        POLY_OPS_DEBUG_LOG("Adding anchor point {} after {}",new_i,i);
    }
    return lpoints[i].next;
}

template<coordinate Coord,std::integral Index>
void clipper<Coord,Index>::do_op(bool_op op,i_point_tracker<Index> *pt,bool tree_out) {
    using namespace detail;

    POLY_OPS_ASSERT(!ran);

    ran = true;

    events.clear();
    breaks.clear();
    loops_out.clear();
    top.clear();

    self_intersection(pt);
    calc_line_bal(op);

    /* replace the points in "hits" of each sample, with anchor points

    TODO: add test to make sure this doesn't mess things up do due the order of
    overlapping line segments changing in the next few steps */
    if(tree_out) {
        for(auto &intr : samples) {
            for(Index &i : intr.hits) i = insert_anchor_point(i);
        }
        for(auto &intr : samples) {
            follow_balance<Index,Coord>(
                lpoints,
                std::exchange(intr.p,insert_anchor_point(intr.p)),
                breaks,pt);
        }
    } else {
        for(auto &intr : samples) {
            follow_balance<Index,Coord>(lpoints,intr.p,breaks,pt);
        }
    }

    /* match all the orphan points to virtual points to make the virtual into
    real points */
    for(Index intr : breaks.orphans) {
        auto virts = breaks.virtual_points.find(lpoints[intr].data);

        POLY_OPS_ASSERT(virts != breaks.virtual_points.end() && !virts->second.empty());

        virt_point<Index> vp = virts->second.back();
        virts->second.pop_back();

        /* by now, it doesn't matter what the value of
        "lpoints[intr].state() & line_state::reverse" is*/
        if(vp.keep) {
            breaks.broken_starts[lpoints[intr].data].push_back(intr);
            lpoints[intr].state() = line_state::keep;
        } else {
            lpoints[intr].state() = line_state::discard;
        }

        if(vp.next_is_end) {
            assert(vp.keep);
            breaks.broken_ends.push_back(intr);
        }
        lpoints[intr].next = vp.next;
    }

    /* match all the points in broken_starts and broken_ends to make new loops
    with the remaining lines */
    for(Index intr : breaks.broken_ends) {
        auto p = end_point(lpoints,intr);
        auto os = breaks.broken_starts.find(p);

        POLY_OPS_ASSERT(os != breaks.broken_starts.end() && !os->second.empty());

        Index b_start = 0;

        /* to minimize the number of holes and prevent overlapping points in
        loops, always connect to the line that results in the greatest
        clock-wise turn */
        if(os->second.size() > 1) {
            bool left = large_ints::negative(triangle_winding(lpoints[intr].data,p,non_anchor_end_point(lpoints,os->second[b_start])));
            for(Index i=1; i<static_cast<Index>(os->second.size()); ++i) {
                auto p3 = non_anchor_end_point(lpoints,os->second[i]);
                bool i_left = large_ints::negative(triangle_winding(lpoints[intr].data,p,p3));
                bool left_of_bs = large_ints::negative(triangle_winding(p,end_point(lpoints,os->second[b_start]),p3));
                if(!(left ? (i_left && left_of_bs) : (i_left || left_of_bs))) {
                    b_start = i;
                    left = i_left;
                }
            }
        }

        lpoints[intr].next = os->second[b_start];
        if(b_start != static_cast<Index>(os->second.size() - 1)) {
            std::swap(os->second[b_start],os->second.back());
        }
        os->second.pop_back();
    }

    // there shouldn't be any left
    POLY_OPS_ASSERT(std::all_of(
        breaks.broken_starts.begin(),
        breaks.broken_starts.end(),
        [](const auto &v) { return v.second.empty(); }));

    for(auto &lp : lpoints) {
        if(lp.state() & line_state::keep) {
            lp.aux.loop_index = (lp.state() & line_state::anchor) ? L_INDEX_ANCHOR : L_INDEX_UNSET;
        } else {
            lp.aux.loop_index = L_INDEX_DISCARDED;
        }
    }

    loops_out.clear();
    find_loops<Index,Coord>(lpoints,loops_out,contig_mem);
}

template<coordinate Coord,std::integral Index>
template<bool TreeOut,typename Tracker>
std::conditional_t<TreeOut,
    borrowed_temp_polygon_tree_range<Coord,Index,Tracker>,
    borrowed_temp_polygon_range<Coord,Index,Tracker>>
clipper<Coord,Index>::execute(bool_op op,Tracker &&tracker) & {
    if(ran) reset();

    do_op(op,tracker.callbacks(),TreeOut);

    if constexpr(TreeOut) {
        loop_hierarchy(lpoints,loops_out,top,samples,contig_mem);
        return make_temp_polygon_tree_range<Index,Coord,Tracker>(lpoints.data(),top,std::forward<Tracker>(tracker));
    } else {
        return make_temp_polygon_range<Index,Coord,Tracker>(lpoints,loops_out,std::forward<Tracker>(tracker));
    }
}

template<coordinate Coord,std::integral Index>
template<bool TreeOut,typename Tracker>
std::conditional_t<TreeOut,
    temp_polygon_tree_range<Coord,Index,Tracker>,
    temp_polygon_range<Coord,Index,Tracker>>
clipper<Coord,Index>::execute(bool_op op,Tracker &&tracker) && {
    if(ran) reset();

    do_op(op,tracker.callbacks(),TreeOut);

    if constexpr(TreeOut) {
        loop_hierarchy(lpoints,loops_out,top,samples,contig_mem);
        return make_temp_polygon_tree_range<Index,Coord>(
            std::move(lpoints),
            std::move(loops_out),
            std::move(top),
            std::forward<Tracker>(tracker));
    } else {
        return make_temp_polygon_range<Index,Coord>(
            std::move(lpoints),
            std::move(loops_out),
            std::forward<Tracker>(tracker));
    }
}

template<coordinate Coord,std::integral Index>
void clipper<Coord,Index>::reset() {
    lpoints.clear();
    samples.clear();
    sweep_nodes.resize(1);
    ran = false;
}


/**
 * A convenience class that combines `clipper` with a specific `point_tracker`
 */
template<coordinate Coord,std::integral Index,point_tracker<Coord,Index> Tracker> class tclipper {
public:
    [[no_unique_address]] Tracker tracker;
    clipper<Coord,Index> base;

    explicit tclipper(std::pmr::memory_resource *_contig_mem=nullptr) requires(std::is_default_constructible_v<Tracker>) :
        base{_contig_mem} {}

    explicit tclipper(Tracker &&tracker,std::pmr::memory_resource *_contig_mem=nullptr) :
        tracker{std::forward<Tracker>(tracker)}, base{_contig_mem} {}

    tclipper(tclipper &&b) = default;

    /**
     * Calls `add_loops(std::forward<R>(loops),cat,tracker.callbacks())`
     */
    template<typename R> void add_loops(R &&loops,bool_set cat) {
        static_assert(point_range_or_range_range<R,Coord>);
        base.add_loops(loops,cat,tracker.callbacks());
    }

    /**
     * Calls `add_loops(std::forward<R>(loops),bool_set::subject,tracker.callbacks())`
     */
    template<typename R> void add_loops_subject(R &&loops) {
        static_assert(point_range_or_range_range<R,Coord>);
        add_loops(std::forward<R>(loops),bool_set::subject,tracker.callbacks());
    }

    /**
     * Calls `add_loops(std::forward<R>(loops),bool_set::clip,tracker.callbacks())`
     */
    template<typename R> void add_loops_clip(R &&loops) {
        static_assert(point_range_or_range_range<R,Coord>);
        add_loops(std::forward<R>(loops),bool_set::clip,tracker.callbacks());
    }

    /**
     * Calls `base.add_loop(cat,tracker.callbacks())`
     */
    clipper<Coord,Index>::point_sink add_loop(bool_set cat) {
        return base.add_loop(cat,tracker.callbacks());
    }

    /**
     * Calls `base.reset()`
     */
    void reset() { base.reset(); }

    /**
     * Calls `base.template execute<TreeOut,Tracker&>(op,tracker)`
     */
    template<bool TreeOut>
    std::conditional_t<TreeOut,
        borrowed_temp_polygon_tree_range<Coord,Index,Tracker&>,
        borrowed_temp_polygon_range<Coord,Index,Tracker&>>
    execute(bool_op op) & { return base.template execute<TreeOut,Tracker&>(op,tracker); }

    /**
     * Calls `std::move(base).template execute<TreeOut,Tracker>(op,std::forward<Tracker>(tracker))`
     */
    template<bool TreeOut>
    std::conditional_t<TreeOut,
        temp_polygon_tree_range<Coord,Index,Tracker>,
        temp_polygon_range<Coord,Index,Tracker>>
    execute(bool_op op) && {
        return std::move(base).template execute<TreeOut,Tracker>(op,std::forward<Tracker>(tracker));
    }
};


/**
 * Generate the union of a set of polygons.
 *
 * This is equivalent to calling `boolean_op` with an empty range passed to
 * `clip` and `bool_op::union_` passed to `op`.
 *
 * `Coord` must satisfy `coordinate`.
 * `Index` must satisfy `std::integral`.
 * `Input` must satisfy `point_range_or_range_range`.
 * `Tracker` must satisfy `point_tracker<Coord,Index>`.
 */
template<bool TreeOut,typename Coord,typename Index=std::size_t,typename Input,typename Tracker>
std::conditional_t<TreeOut,
    temp_polygon_tree_range<Coord,Index,Tracker>,
    temp_polygon_range<Coord,Index,Tracker>>
union_op(
    Input &&input,
    Tracker &&tracker,
    std::pmr::memory_resource *contig_mem=nullptr)
{
    static_assert(point_tracker<Tracker,Coord,Index>);
    static_assert(point_range_or_range_range<Input,Coord>);
    static_assert(coordinate<Coord>);
    static_assert(std::integral<Index>);

    tclipper<Coord,Index,Tracker> n{std::forward<Tracker>(tracker),contig_mem};
    n.add_loops(std::forward<Input>(input),bool_set::subject);
    return std::move(n).template execute<TreeOut>(bool_op::union_);
}

/**
 * Generate the union of a set of polygons.
 *
 * This is equivalent to calling `boolean_op` with an empty range passed to
 * `clip` and `bool_op::union_` passed to `op`.
 *
 * `Coord` must satisfy `coordinate`.
 * `Index` must satisfy `std::integral`.
 * `Input` must satisfy `point_range_or_range_range`.
 */
template<bool TreeOut,typename Coord,typename Index=std::size_t,typename Input>
auto union_op(Input &&input,std::pmr::memory_resource *contig_mem=nullptr) {
    return union_op<TreeOut,Coord,Index,Input,null_tracker<Coord,Index>>(std::forward<Input>(input),{},contig_mem);
}

/**
 * "Normalize" a set of polygons.
 *
 * This is equivalent to calling `boolean_op` with an empty range passed to
 * `clip` and `bool_op::normalize` passed to `op`.
 *
 * `Coord` must satisfy `coordinate`.
 * `Index` must satisfy `std::integral`.
 * `Input` must satisfy `point_range_or_range_range`.
 * `Tracker` must satisfy `point_tracker<Coord,Index>`.
 */
template<bool TreeOut,typename Coord,typename Index=std::size_t,typename Input,typename Tracker>
std::conditional_t<TreeOut,
    temp_polygon_tree_range<Coord,Index,Tracker>,
    temp_polygon_range<Coord,Index,Tracker>>
normalize_op(
    Input &&input,
    Tracker &&tracker,
    std::pmr::memory_resource *contig_mem=nullptr)
{
    static_assert(point_tracker<Tracker,Coord,Index>);
    static_assert(point_range_or_range_range<Input,Coord>);
    static_assert(coordinate<Coord>);
    static_assert(std::integral<Index>);

    tclipper<Coord,Index,Tracker> n{std::forward<Tracker>(tracker),contig_mem};
    n.add_loops(std::forward<Input>(input),bool_set::subject);
    return std::move(n).template execute<TreeOut>(bool_op::normalize);
}

/**
 * "Normalize" a set of polygons.
 *
 * This is equivalent to calling `boolean_op` with an empty range passed to
 * `clip` and `bool_op::normalize` passed to `op`.
 *
 * `Coord` must satisfy `coordinate`.
 * `Index` must satisfy `std::integral`.
 * `Input` must satisfy `point_range_or_range_range`.
 */
template<bool TreeOut,typename Coord,typename Index=std::size_t,typename Input>
auto normalize_op(Input &&input,std::pmr::memory_resource *contig_mem=nullptr) {
    return normalize_op<TreeOut,Coord,Index,Input,null_tracker<Coord,Index>>(std::forward<Input>(input),{},contig_mem);
}

/**
 * Perform a boolean operation on two sets of polygons.
 *
 * This is equivalent to the following:
 *
 *     tclipper<Coord,Index,Tracker> n{std::forward<Tracker>(tracker),contig_mem};
 *     n.add_loops(std::forward<SInput>(subject),bool_set::subject);
 *     n.add_loops(std::forward<CInput>(clip),bool_set::clip);
 *     RETURN_VALUE = std::move(n).execute<TreeOut>(op);
 * 
 * `Coord` must satisfy `coordinate`.
 * `Index` must satisfy `std::integral`.
 * `SInput` must satisfy `point_range_or_range_range`.
 * `CInput` must satisfy `point_range_or_range_range`.
 * `Tracker` must satisfy `point_tracker<Coord,Index>`.
 */
template<bool TreeOut,typename Coord,typename Index=std::size_t,typename SInput,typename CInput,typename Tracker>
std::conditional_t<TreeOut,
    temp_polygon_tree_range<Coord,Index,Tracker>,
    temp_polygon_range<Coord,Index,Tracker>>
boolean_op(
    SInput &&subject,
    CInput &&clip,
    bool_op op,
    Tracker &&tracker,
    std::pmr::memory_resource *contig_mem=nullptr)
{
    static_assert(point_tracker<Tracker,Coord,Index>);
    static_assert(point_range_or_range_range<CInput,Coord>);
    static_assert(point_range_or_range_range<SInput,Coord>);
    static_assert(coordinate<Coord>);
    static_assert(std::integral<Index>);

    tclipper<Coord,Index,Tracker> n{std::forward<Tracker>(tracker),contig_mem};
    n.add_loops(std::forward<SInput>(subject),bool_set::subject);
    n.add_loops(std::forward<CInput>(clip),bool_set::clip);
    return std::move(n).template execute<TreeOut>(op);
}


/**
 * Perform a boolean operation on two sets of polygons.
 *
 * This is equivalent to the following:
 *
 *     clipper<Coord,Index> n{contig_mem};
 *     n.add_loops(std::forward<SInput>(subject),bool_set::subject);
 *     n.add_loops(std::forward<CInput>(clip),bool_set::clip);
 *     RETURN_VALUE = std::move(n).execute<TreeOut>(op);
 * 
 * `Coord` must satisfy `coordinate`.
 * `Index` must satisfy `std::integral`.
 * `SInput` must satisfy `point_range_or_range_range`.
 * `CInput` must satisfy `point_range_or_range_range`.
 */
template<bool TreeOut,typename Coord,typename Index=std::size_t,typename SInput,typename CInput>
auto boolean_op(SInput &&subject,CInput &&clip,bool_op op,std::pmr::memory_resource *contig_mem=nullptr) {
    return boolean_op<TreeOut,Coord,Index,SInput,CInput,null_tracker<Coord,Index>>(
        std::forward<SInput>(subject),
        std::forward<CInput>(clip),
        op,
        {},
        contig_mem);
}

} // namespace poly_ops

#endif
