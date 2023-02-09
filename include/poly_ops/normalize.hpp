#ifndef POLY_OPS_NORMALIZE_HPP
#define POLY_OPS_NORMALIZE_HPP

#include <vector>
#include <set>
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

#include "base.hpp"


#ifndef POLY_OPS_GRAPHICAL_DEBUG
#define POLY_OPS_GRAPHICAL_DEBUG 0
#endif

#ifndef POLY_OPS_ASSERT
#include <cassert>
#define POLY_OPS_ASSERT assert
#endif

// used for checks that will significantly slow down the algorithm
#ifndef POLY_OPS_ASSERT_SLOW
#define POLY_OPS_ASSERT_SLOW(X) (void)0
#endif

// the graphical test program defines these
#ifndef POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_F
#define POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_F
#define POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_FR
#define POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_B
#define POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_BR
#define POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_CALC_BALANCE
#define POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_CALC_SAMPLE
#define POLY_OPS_DEBUG_STEP_BY_STEP_MISSED_INTR (void)0
#define POLY_OPS_DEBUG_ITERATION
#endif

namespace poly_ops {
/** Point tracking common interface.

Note that the destructor is not virtual, and is instead protected.
*/
template<typename Index> class point_tracker {
public:
    /** Called when intersecting lines are broken and consequently new points
    are added.

    One of "a" or "b" will refer to a new point but not necessarily both (an
    endpoint touching the middle of another line only requires a new point for
    the other line). The index of a new point will always be one greater than
    the index of the last point added, thus, to know if a point is new, keep
    track of the number of points added.

    @param a The index of the intersection point of the first line.
    @param b The index of the intersection point of the second line.
    */
    virtual void new_intersection(Index a,Index b) = 0;

    /** Called when points are "merged".

    When parts of the shape are going to be truncated, this is called on the
    first point to be discarded as "from" and the point that follows it as "to".
    If the next point is also to be discarded, this is called again with that
    point along with the point after that. This is repeated until the point
    reached is not going to be discarded, or the following point was already
    discarded because we've circled around the entire loop.

    @param from The index of the point to be discarded.
    @param to The index of the point that follows "from".
    */
    virtual void point_merge(Index from,Index to) = 0;

    /** Called for every point initially added.

    Every added point has an implicit index (this index is unrelated to
    `original_i`). This method is first called when point zero is added. Every
    subsequent call corresponds to the index that is one greater than the
    previous call.

    `original_i` is the index of the input point that the added point
    corresponds to. The value is what the array index of the original point
    would be if all the input points were concatinated, in order, into a single
    array. This will not necessarily be called for every point of the input;
    duplicate consecutive points are ignored.

    @param original_i The index of the input point that this added point
        corresponds to.
    */
    virtual void point_added(Index original_i) = 0;

protected:
    ~point_tracker() = default;
};

namespace detail {
inline constexpr int UNDEF_LINE_BAL = std::numeric_limits<int>::lowest();

template<typename Index,typename Coord> struct loop_point {
    point_t<Coord> data;
    Index next;

    int line_bal;

    loop_point() = default;
    loop_point(point_t<Coord> data,Index next,int line_bal=UNDEF_LINE_BAL) :
        data{data}, next{next}, line_bal{line_bal} {}

    friend void swap(loop_point &a,loop_point &b) {
        using std::swap;

        swap(a.data,b.data);
        swap(a.next,b.next);
        swap(a.line_bal,b.line_bal);
    }
};

template<typename Index> struct segment {
    Index a;
    Index b;

    segment() = default;
    segment(Index a,Index b) : a{a}, b{b} {}
    segment(const segment&) = default;

    auto a_x(const auto &points) const { return points[a].data[0]; }
    auto a_y(const auto &points) const { return points[a].data[1]; }
    auto b_x(const auto &points) const { return points[b].data[0]; }
    auto b_y(const auto &points) const { return points[b].data[1]; }

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
"forward".

Lession learned: don't split calc_balance_intr or calc_intr_sample into positive
and negative versions to reduce how many checks are done. The value of
"hsign_of" can change as points are added. */
enum class event_type_t {vforward,backward,forward,vbackward,calc_balance_intr,calc_intr_sample};

enum class event_status_t {normal,deleted};

template<typename Index> struct event {
    segment<Index> ab;
    event_type_t type;
    event_status_t status;

    event() = default;
    event(segment<Index> ab,event_type_t type)
        : ab{ab}, type{type}, status{event_status_t::normal} {}
    event(const event &b) = default;

    event &operator=(const event&) = default;

    segment<Index> line_ba() const { return {ab.b,ab.a}; }

    Index p() const { return ab.a; }
    Index sample_i() const {
        POLY_OPS_ASSERT(type == event_type_t::calc_intr_sample);
        return ab.b;
    }

    friend bool operator==(const event &a,const event &b) {
        return a.type == b.type && a.ab == b.ab;
    }
};

// This refers to a 1-dimensional edge, i.e. the start or end of a line segment
enum class at_edge_t {
    start = -1,
    no = 0,
    end = 1};

template<typename Coord> at_edge_t is_edge(Coord val,Coord end_val) {
    if(val == 0) return at_edge_t::start;
    if(val == end_val) return at_edge_t::end;
    return at_edge_t::no;
}

template<typename T> bool between(T x,T lo,T hi) {
    return x > lo && x < hi;
}

template<typename Index,typename Coord>
bool intersects_parallel(segment<Index> s1,segment<Index> s2,const loop_point<Index,Coord> *points,point_t<Coord> &p,std::span<at_edge_t,2> at_edge) {
    using long_t = long_coord_t<Coord>;
    using std::swap;

    if(triangle_winding(points[s1.a].data,points[s1.b].data,points[s2.a].data) != 0) return false;

    /* Reduce each line segment into a 1D value. Each d* value is the scalar
    projection of the corresponding s*.* point times the magnitude of s1. */
    point_t<long_t> v1b = vcast<long_t>(points[s1.b].data-points[s1.a].data);
    long_t d1a = static_cast<long_t>(0);
    long_t d1b = square(v1b);
    long_t d2a = vdot(points[s2.a].data-points[s1.a].data,v1b);
    long_t d2b = vdot(points[s2.b].data-points[s1.a].data,v1b);

    POLY_OPS_ASSERT(0 != d1b && d2a != d2b);

    auto [lo1,hi1] = std::minmax(d1a,d1b);
    auto [lo2,hi2] = std::minmax(d2a,d2b);

    if(between(d1a,lo2,hi2)) {
        p = points[s1.a].data;
        at_edge[0] = at_edge_t::start;
        at_edge[1] = at_edge_t::no;
        return true;
    }
    if(between(d1b,lo2,hi2)) {
        p = points[s1.b].data;
        at_edge[0] = at_edge_t::end;
        at_edge[1] = at_edge_t::no;
        return true;
    }
    if(between(d2a,lo1,hi1)) {
        p = points[s2.a].data;
        at_edge[0] = at_edge_t::no;
        at_edge[1] = at_edge_t::start;
        return true;
    }
    if(between(d2b,lo1,hi1)) {
        p = points[s2.b].data;
        at_edge[0] = at_edge_t::no;
        at_edge[1] = at_edge_t::end;
        return true;
    }

    /* if both points intersect, we just return the first one */
    POLY_OPS_ASSERT((s1.a_x(points) < s1.b_x(points) || s1.a_is_main(points)) &&
        (s2.a_x(points) < s2.b_x(points) || s2.a_is_main(points)));
    if(lo1 == lo2 && hi1 == hi2) {
        if(d1a == d2a) {
            if(s1.a != s2.a) {
                p = points[s1.a].data;
                at_edge[0] = at_edge_t::start;
                at_edge[1] = at_edge_t::start;
                return true;
            } else if(s1.b != s2.b) {
                p = points[s1.b].data;
                at_edge[0] = at_edge_t::end;
                at_edge[1] = at_edge_t::end;
                return true;
            }
        } else {
            if(s1.a != s2.b) {
                p = points[s1.a].data;
                at_edge[0] = at_edge_t::start;
                at_edge[1] = at_edge_t::end;
                return true;
            } else if(s1.b != s2.a) {
                p = points[s1.b].data;
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

/* Check if two line segments intersect.

Returns "true" if the lines intersect. Adjecent lines do not count as
intersecting.

'p' will be set to the point of intersection, or in the case of coincident
lines, 'p' will be one of the end points that is inside the other line (if both
ends are inside the other segment, one end is chosen arbitrarily). */
template<typename Index,typename Coord>
bool intersects(
    segment<Index> s1,
    segment<Index> s2,
    const loop_point<Index,Coord> *points,
    point_t<Coord> &p,
    std::span<at_edge_t,2> at_edge,
    rand_generator &rgen)
{
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
    if(d == 0) return intersects_parallel(s1,s2,points,p,at_edge);

    /* Connected lines do not count as intersecting. This check must be done
    after checking for parallel lines because coincident lines can have more
    than one point of intersection. */
    if(s1.a == s2.a || s1.a == s2.b || s1.b == s2.a || s1.b == s2.b) return false;

    long_t t_i = static_cast<long_t>(x1-x3)*(y3-y4) - static_cast<long_t>(y1-y3)*(x3-x4);
    long_t u_i = static_cast<long_t>(x1-x3)*(y1-y2) - static_cast<long_t>(y1-y3)*(x1-x2);

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
        auto t = static_cast<real_coord_t<Coord>>(t_i)/d;
        real_coord_t<Coord> rx = t * (x2-x1);
        real_coord_t<Coord> ry = t * (y2-y1);

        /* Simple rounding is not sufficient because there is a tiny chance that
        lines are arranged such that rounding one intersection reorders the
        lines and creates a new intersection, and rounding the new intersection
        creates yet another intersection, and so forth, until the lines are
        nowhere near where they started. */

        std::uniform_int_distribution<unsigned short> rdist(0,1);

        p[0] = x1 + (rdist(rgen) ? coord_ops<Coord>::floor(rx) : coord_ops<Coord>::ceil(rx));
        p[1] = y1 + (rdist(rgen) ? coord_ops<Coord>::floor(ry) : coord_ops<Coord>::ceil(ry));

        at_edge[0] = at_edge_t::no;
        at_edge[1] = at_edge_t::no;

        if(p[0] == x1 && p[1] == y1) at_edge[0] = at_edge_t::start;
        else if(p[0] == x2 && p[1] == y2) at_edge[0] = at_edge_t::end;

        if(p[0] == x3 && p[1] == y3) at_edge[1] = at_edge_t::start;
        else if(p[0] == x4 && p[1] == y4) at_edge[1] = at_edge_t::end;

        /*Coord xf = x1 + coord_ops<Coord>::floor(rx);
        Coord xc = x1 + coord_ops<Coord>::ceil(rx);
        Coord yf = y1 + coord_ops<Coord>::floor(ry);
        Coord yc = y1 + coord_ops<Coord>::ceil(ry);

        at_edge[0] = at_edge_t::no;
        at_edge[1] = at_edge_t::no;

        if((x1 == xc || x1 == xf) && (y1 == yc || y1 == yf)) {
            p = {x1,y1};
            at_edge[0] = at_edge_t::start;
            if(p[0] == x3 && p[1] == y3) at_edge[1] = at_edge_t::start;
            else if(p[0] == x4 && p[1] == y4) at_edge[1] = at_edge_t::end;
        } else if((x2 == xc || x2 == xf) && (y2 == yc || y2 == yf)) {
            p = {x2,y2};
            at_edge[0] = at_edge_t::end;
            if(p[0] == x3 && p[1] == y3) at_edge[1] = at_edge_t::start;
            else if(p[0] == x4 && p[1] == y4) at_edge[1] = at_edge_t::end;
        } else if((x3 == xc || x3 == xf) && (y3 == yc || y3 == yf)) {
            p = {x3,y3};
            at_edge[1] = at_edge_t::start;
        } else if((x4 == xc || x4 == xf) && (y4 == yc || y4 == yf)) {
            p = {x4,y4};
            at_edge[1] = at_edge_t::end;
        } else {
            p[0] = x1 + coord_ops<Coord>::round(rx);
            p[1] = y1 + coord_ops<Coord>::round(ry);
        }
        

        POLY_OPS_ASSERT((p[0] == x1 && p[1] == y1) == (at_edge[0] == at_edge_t::start));
        POLY_OPS_ASSERT((p[0] == x2 && p[1] == y2) == (at_edge[0] == at_edge_t::end));
        POLY_OPS_ASSERT((p[0] == x3 && p[1] == y3) == (at_edge[1] == at_edge_t::start));
        POLY_OPS_ASSERT((p[0] == x4 && p[1] == y4) == (at_edge[1] == at_edge_t::end));*/
    }

    return true;
}

template<typename Coord>
bool lower_angle(point_t<Coord> ds,point_t<Coord> dp,bool tie_breaker) {
    auto v = static_cast<long_coord_t<Coord>>(ds.y())*dp.x();
    auto w = static_cast<long_coord_t<Coord>>(dp.y())*ds.x();
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

template<typename Coord> Coord hsign_of(const point_t<Coord> &delta) {
    auto r = delta.x() ? delta.x() : delta.y();
    POLY_OPS_ASSERT(r != 0);
    return r;
}

template<typename Index,typename Coord> point_t<Coord> line_delta(
    const loop_point<Index,Coord> *points,Index p)
{
    return points[points[p].next].data - points[p].data;
}

/* Calculate the "line balance".

The line balance is based on the idea of a winding number, except it
applies the lines instead of points inside. A positive number means nested
geometry, negative means inverted geometry, and zero means normal geometry
(clockwise oriented polygons and counter-clockwise holes). */
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
    int wn;

public:
    line_balance(const loop_point<Index,Coord> *points,Index p1,std::pmr::vector<Index> &intrs_tmp) :
        points(points), p1(p1), intrs_tmp(intrs_tmp),
        dp1(line_delta(points,p1)),
        hsign(hsign_of(dp1)), wn(dp1.x() < 0 ? -1 : 0)
    {
        POLY_OPS_ASSERT(hsign);
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
                wn += a_is_main ? 1 : -1;
                intrs_tmp.push_back(main_i);
            }
        }
    }

    int result() const { return wn; }
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

    event<Index> &operator[](std::size_t i) { return events[i]; }
    const event<Index> &operator[](std::size_t i) const { return events[i]; }

    event<Index> &find(points_ref points,const segment<Index> s,event_type_t type,std::size_t upto) {
        auto itr = std::lower_bound(
            events.begin(),
            events.begin()+upto,
            event<Index>{s,type},
            cmp{points});
        POLY_OPS_ASSERT(itr != (events.begin()+upto) && itr->ab == s && itr->type == type);
        return *itr;
    }

    event<Index> &find(points_ref points,const segment<Index> s,event_type_t type) {
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
            if(current_i >= static_cast<std::ptrdiff_t>(new_events[0].i)) return {events[current_i],false};

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
                        events.begin()+last_size,
                        events[last_size+i],
                        cmp{points});
                    
                    /* The event might be a duplicate. This only happens with
                    "calc_balance_intr" events when finding unconnected lines
                    that have touching end-points, but if we have to repeatedly
                    backtrack, the event would be repeatedly added. */
                    if(itr == events.begin()+last_size || *itr != events[last_size+i]) {
                        new_events.emplace_back(std::move(events[last_size+i]),static_cast<std::size_t>(itr - events.begin()));
                    }
                }
                if(new_events.size() != new_count) {
                    events.resize(last_size + new_events.size());
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

        return {events[++current_i],true};
    }

    void add_event(Index sa,Index sb,event_type_t t) {
        events.emplace_back(segment<Index>{sa,sb},t);
    }

    void add_fb_events(points_ref points,Index sa,Index sb) {
        auto f = event_type_t::forward;
        auto b = event_type_t::backward;
        if(points[sa].data[0] == points[sb].data[0]) {
            /* this is an assumption made by vert_overlap() */
            POLY_OPS_ASSERT(points[sa].next == sb);

            f = event_type_t::vforward;
            b = event_type_t::vbackward;
        }
        add_event(sa,sb,f);
        add_event(sb,sa,b);
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

    event<Index> &current_start(points_ref points) {
        POLY_OPS_ASSERT(
            current().type == event_type_t::backward
            || current().type == event_type_t::vbackward);
        return find(
            points,
            current().line_ba(),
            current().type == event_type_t::backward ?
                event_type_t::forward : event_type_t::vforward,
            current_i);
    }

    event<Index> &current() { return events[current_i]; }
    const event<Index> &current() const { return events[current_i]; }

    /* these are used by graphical_test_common.hpp of the testing code */
    auto begin() const { return events.begin(); }
    auto end() const { return events.end(); }
    std::size_t size() const { return events.size(); }
};

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
    const std::pmr::vector<loop_point<Index,Coord>> &lpoints;

    auto winding(Index p1,Index p2,Index p3) const {
        return triangle_winding(lpoints[p1].data,lpoints[p2].data,lpoints[p3].data);
    }

    bool operator()(segment<Index> s1,segment<Index> s2) const {
        if(s1.a_x(lpoints) == s2.a_x(lpoints)) {
            Coord r = s1.a_y(lpoints) - s2.a_y(lpoints);
            if(r) return r > 0;
            long_coord_t<Coord> r2 = winding(s1.a,s1.b,s2.b);
            if(r2) return r2 > 0;
            /* the lines are coincident and have the same start point but they
            might be pointing away from each other */
            r = s1.b_y(lpoints) - s2.b_y(lpoints);
            if(r) return r > 0;
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

template<typename Index,typename Coord>
Index split_segment(
    events_t<Index,Coord> &events,
    sweep_t<Index,Coord> &sweep,
    std::pmr::vector<loop_point<Index,Coord>> &points,
    segment<Index> s,
    const point_t<Coord> &c,
    at_edge_t at_edge)
{
    Index sa = s.a;
    Index sb = s.b;

    if(at_edge != at_edge_t::no) {
        return at_edge == at_edge_t::start ? sa : sb;
    }
    sweep.erase(s);
    events.find(
        points,
        s,
        points[sa].data[0] == points[sb].data[0] ? event_type_t::vforward : event_type_t::forward).status = event_status_t::deleted;

    POLY_OPS_ASSERT(points[sa].data != c && points[sb].data != c);

    if(!s.a_is_main(points)) std::swap(sa,sb);

    Index mid = points.size();
    points[sa].next = mid;
    points.emplace_back(c,sb,UNDEF_LINE_BAL);
    POLY_OPS_ASSERT_SLOW(check_integrity(points,sweep));

    /* even if the original line was not vertical, one of the new lines might
    be vertical after rounding */
    if(points[sa].data[0] <= points[mid].data[0]) {
        events.add_fb_events(points,sa,mid);
    } else {
        events.add_fb_events(points,mid,sa);
    }
    if(points[mid].data[0] <= points[sb].data[0]) {
        events.add_fb_events(points,mid,sb);
    } else {
        events.add_fb_events(points,sb,mid);
    }

    return mid;
}

template<typename Index,typename Coord>
bool check_intersection(
    events_t<Index,Coord> &events,
    sweep_t<Index,Coord> &sweep,
    std::pmr::vector<loop_point<Index,Coord>> &points,
    segment<Index> s1,
    segment<Index> s2,
    rand_generator &rgen,
    point_tracker<Index> *pt)
{
    point_t<Coord> intr;
    at_edge_t at_edge[2];
    if(intersects(s1,s2,points.data(),intr,at_edge,rgen)) {
        if(at_edge[0] == at_edge_t::no || at_edge[1] == at_edge_t::no) [[likely]] {
            Index intr1 = split_segment(events,sweep,points,s1,intr,at_edge[0]);
            Index intr2 = split_segment(events,sweep,points,s2,intr,at_edge[1]);

            if(pt) pt->new_intersection(intr1,intr2);

            events.add_event(intr1,intr2,event_type_t::calc_balance_intr);

            return true;
        } else {
            Index intr1 = at_edge[0] == at_edge_t::start ? s1.a : s1.b;
            Index intr2 = at_edge[1] == at_edge_t::start ? s2.a : s2.b;
            if(points[intr1].line_bal == UNDEF_LINE_BAL || points[intr2].line_bal == UNDEF_LINE_BAL) {
                events.add_event(intr1,intr2,event_type_t::calc_balance_intr);
            }
        }
    }

    return false;
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

template<typename Index>
using intr_map_t = std::pmr::map<Index,std::pmr::vector<Index>>;

template<typename Index>
struct intr_t {
    Index p;
    std::pmr::vector<Index> hits;
};

template<typename Index>
using intr_array_t = std::pmr::vector<intr_t<Index>>;

/* This is only used for debugging */
template<typename Index,typename Coord>
bool intersects_any(segment<Index> s1,const sweep_t<Index,Coord> &sweep,const loop_point<Index,Coord> *points) {
    rand_generator rgen;
    for(auto s2 : sweep) {
        point_t<Coord> intr;
        at_edge_t at_edge[2];
        if(intersects(s1,s2,points,intr,at_edge,rgen) && (at_edge[0] == at_edge_t::no || at_edge[1] == at_edge_t::no)) {
            POLY_OPS_DEBUG_STEP_BY_STEP_MISSED_INTR;
            return true;
        }
    }
    return false;
}

template<typename Index,typename Coord>
bool consistent_order(const sweep_t<Index,Coord> &sweep) {
    auto cmp = sweep.key_comp();
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

template<typename Index,typename Coord>
using broken_starts_t = std::pmr::map<point_t<Coord>,std::pmr::vector<Index>,point_less>;

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
    point_tracker<Index> *pt)
{
    POLY_OPS_ASSERT((points[intr].line_bal != UNDEF_LINE_BAL));
    Index next = points[intr].next;
    Index prev = intr;
    while(next != intr) {
        if(points[next].line_bal != UNDEF_LINE_BAL) {
            if(points[next].line_bal == 0) {
                if(points[intr].line_bal != 0) {
                    broken_starts[points[next].data].push_back(next);
                    if(pt) pt->point_merge(next,intr);
                }
            } else if(points[intr].line_bal == 0) {
                broken_ends.push_back(prev);
            }
            break;
        }

        points[next].line_bal = points[intr].line_bal;
        if(points[intr].line_bal != 0) {
            if(pt) pt->point_merge(next,intr);
        }

        prev = next;
        next = points[next].next;
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

} // namespace detail

template<typename Index,typename Coord> class temp_polygon_proxy;

template<typename Index,typename Coord> class proto_loop_iterator {
    friend temp_polygon_proxy<Index,Coord>;

    const detail::loop_point<Index,Coord> *lpoints;
    Index i;

    /* Loops have the same "i" value for "begin()" and "end()" thus a different
    value is used to distinguish the end from other iterators. */
    Index dist_from_end;

    friend bool operator==(const proto_loop_iterator &a,const proto_loop_iterator &b) {
        POLY_OPS_ASSERT(a.lpoints == b.lpoints);
        return a.dist_from_end == b.dist_from_end;
    }

    friend bool operator==(const proto_loop_iterator &a,const std::default_sentinel_t&) {
        return a.dist_from_end == 0;
    }
    friend bool operator==(const std::default_sentinel_t&,const proto_loop_iterator &b) {
        return b.dist_from_end == 0;
    }

    proto_loop_iterator(const detail::loop_point<Index,Coord> *lpoints,Index i,Index dist_from_end)
        : lpoints(lpoints), i(i), dist_from_end(dist_from_end) {}

public:
    proto_loop_iterator &operator++() {
        i = lpoints[i].next;
        --dist_from_end;
        return *this;
    }

    proto_loop_iterator operator++(int) {
        return {lpoints,std::exchange(i,lpoints[i].next),dist_from_end--};
    }

    const point_t<Coord> &operator*() const {
        return lpoints[i].data;
    }
};

template<typename Index,typename Coord> class temp_polygon_proxy {
    const detail::loop_point<Index,Coord> *lpoints;
    const detail::temp_polygon<Index> &data;
public:
    temp_polygon_proxy(
        const detail::loop_point<Index,Coord> *lpoints,
        const detail::temp_polygon<Index> &data)
        : lpoints(lpoints), data(data) {}

    proto_loop_iterator<Index,Coord> begin() const {
        return {lpoints,data.loop_loc.start,data.loop_loc.size};
    }

    std::default_sentinel_t end() const { return {}; }

    Index size() const { return data.loop_loc.size; }

    auto inner_loops() const {
        return data.children | std::views::transform(
            [this](const detail::temp_polygon<Index> *inner){
                return temp_polygon_proxy{lpoints,*inner};
            });
    }
};
} // namespace poly_ops

template<typename Index,typename Coord>
inline constexpr bool std::ranges::enable_borrowed_range<poly_ops::temp_polygon_proxy<Index,Coord>> = true;

namespace poly_ops {
namespace detail {

template<typename Index,typename Coord>
void find_loops(
    std::pmr::vector<loop_point<Index,Coord>> &lpoints,
    std::pmr::vector<temp_polygon<Index>> &loops,
    std::pmr::memory_resource *contig_mem)
{
    for(Index i=0; i<static_cast<Index>(lpoints.size()); ++i) {
        if(lpoints[i].line_bal == 0) {
            int loop_i = static_cast<int>(loops.size()+1);
            loops.emplace_back(
                loop_location<Index>{i,1},
                std::pmr::vector<temp_polygon<Index>*>(contig_mem));

            /* give this a non-zero value to prevent this point from being
            scanned again */
            lpoints[i].line_bal = loop_i;

            for(Index j = lpoints[i].next; j != i; j = lpoints[j].next) {
                POLY_OPS_ASSERT(lpoints[j].line_bal == 0);

                ++loops.back().loop_loc.size;
                lpoints[j].line_bal = loop_i;
            }
        }
    }
}

template<typename Index,typename Coord>
void add_ordered_loop_item(
    const std::pmr::vector<loop_point<Index,Coord>> &lpoints,
    intr_array_t<Index> &ordered_loops,
    Index p,
    std::pmr::vector<Index> &&hits)
{
    if(lpoints[p].line_bal != UNDEF_LINE_BAL) {
        ordered_loops.emplace_back(p,std::move(hits));
    }
}

template<typename Index,typename Coord>
intr_array_t<Index> unique_sorted_loop_points(
    const std::pmr::vector<loop_point<Index,Coord>> &lpoints,
    intr_map_t<Index> &&intrs,
    intr_array_t<Index> &&samples,
    std::pmr::memory_resource *contig_mem)
{
    intr_array_t<Index> ordered_loops(contig_mem);
    ordered_loops.reserve(intrs.size() + samples.size());
    for(auto &item : intrs) add_ordered_loop_item(lpoints,ordered_loops,std::get<0>(item),std::move(std::get<1>(item)));
    for(auto &item : samples) add_ordered_loop_item(lpoints,ordered_loops,item.p,std::move(item.hits));

    auto loop_id = [&](const intr_t<Index> &item){
        return lpoints[item.p].line_bal;
    };
    std::ranges::sort(ordered_loops,{},loop_id);
    ordered_loops.resize(
        std::ranges::begin(std::ranges::unique(ordered_loops,{},loop_id)) - ordered_loops.begin());

    return ordered_loops;
}

template<typename Index,typename Coord>
void replace_line_indices_with_loop_indices(
    const std::pmr::vector<loop_point<Index,Coord>> &lpoints,
    const std::pmr::vector<temp_polygon<Index>> &loops,
    intr_array_t<Index> &ordered_loops,
    std::pmr::memory_resource *contig_mem)
{
    std::pmr::vector<int> inside(loops.size(),0,contig_mem);
    for(auto& item : ordered_loops) {
        for(Index i : item.hits) {
            point_t<Coord> d = lpoints[i].data - lpoints[lpoints[i].next].data;
            inside[lpoints[i].line_bal-1]
                += ((d.x() ? d.x() : d.y()) > 0) ? 1 : -1;
        }
        std::size_t inside_count = 0;
        for(std::size_t i=0; i<inside.size(); ++i) {
            POLY_OPS_ASSERT(inside_count < item.hits.size());
            if(inside[i]) item.hits[inside_count++] = static_cast<Index>(i);
        }
        item.hits.resize(inside_count);
        std::ranges::fill(inside,0);

        item.p = static_cast<Index>(lpoints[item.p].line_bal-1);
    }
}

template<typename Index>
std::pmr::vector<temp_polygon<Index>*> arrange_loops(
    const intr_array_t<Index> &ordered_loops,
    std::pmr::vector<temp_polygon<Index>> &loops,
    std::pmr::memory_resource *contig_mem)
{
    POLY_OPS_ASSERT(std::ranges::equal(
        ordered_loops | std::views::transform([](auto &item) { return item.p; }),
        std::ranges::iota_view<Index,Index>(0,loops.size())));

    std::pmr::vector<temp_polygon<Index>*> top(contig_mem);

    for(const auto& item : ordered_loops) {
        if(item.hits.empty()) top.push_back(&loops[item.p]);
        else {
            for(auto outer_i : item.hits) {
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

template<typename Index,typename Coord>
std::pmr::vector<temp_polygon<Index>*> loop_hierarchy(
    std::pmr::vector<loop_point<Index,Coord>> &lpoints,
    std::pmr::vector<temp_polygon<Index>> &loops,
    intr_map_t<Index> &intrs,
    intr_array_t<Index> &samples,
    std::pmr::memory_resource *contig_mem)
{
    auto ordered_loops = unique_sorted_loop_points<Index,Coord>(lpoints,intrs,samples,contig_mem);
    replace_line_indices_with_loop_indices<Index,Coord>(lpoints,loops,ordered_loops,contig_mem);
    return arrange_loops<Index>(ordered_loops,loops,contig_mem);
}

template<typename Index,typename Coord> auto make_temp_polygon_tree_range(
    const std::pmr::vector<loop_point<Index,Coord>> &lpoints,
    std::pmr::vector<temp_polygon<Index>> &&loops,
    std::pmr::vector<temp_polygon<Index>*> &&top)
{
    return std::ranges::owning_view(std::move(top))
        | std::views::transform(
            [lpoints=std::move(lpoints),loops=std::move(loops)]
            (const temp_polygon<Index> *poly) {
                return temp_polygon_proxy<Index,Coord>(lpoints.data(),*poly);
            });
}

template<typename Index,typename Coord> auto make_temp_polygon_tree_range(
    const std::pmr::vector<loop_point<Index,Coord>> &lpoints,
    const std::pmr::vector<temp_polygon<Index>> &loops,
    std::pmr::vector<temp_polygon<Index>*> &&top)
{
    return std::ranges::owning_view(std::move(top))
        | std::views::transform(
            [&](const temp_polygon<Index> *poly) {
                return temp_polygon_proxy<Index,Coord>(lpoints.data(),*poly);
            });
}

template<typename Index,typename Coord> auto make_temp_polygon_range(
    std::pmr::vector<loop_point<Index,Coord>> &&lpoints,
    std::pmr::vector<temp_polygon<Index>> &&loops)
{
    return std::ranges::owning_view(std::move(loops))
        | std::views::transform(
            [lpoints=std::move(lpoints)]
            (const temp_polygon<Index> &poly) {
                return temp_polygon_proxy<Index,Coord>(lpoints.data(),poly);
            });
}

template<typename Index,typename Coord> auto make_temp_polygon_range(
    const std::pmr::vector<loop_point<Index,Coord>> &lpoints,
    const std::pmr::vector<temp_polygon<Index>> &loops)
{
    return loops | std::views::transform(
            [&](const temp_polygon<Index> &poly) {
                return temp_polygon_proxy<Index,Coord>(lpoints.data(),poly);
            });
}

} // namespace detail

template<typename Index,typename Coord>
using temp_polygon_tree_range = decltype(
    detail::make_temp_polygon_tree_range<Index,Coord>({},{},{}));

template<typename Index,typename Coord>
using temp_polygon_range = decltype(
    detail::make_temp_polygon_range<Index,Coord>({},{}));

template<std::integral Index,coordinate Coord> class normalizer {
    std::pmr::memory_resource *contig_mem;
    std::pmr::unsynchronized_pool_resource discrete_mem;

    std::pmr::vector<detail::loop_point<Index,Coord>> lpoints;
    detail::intr_array_t<Index> samples;
    detail::intr_map_t<Index> intrs;
    detail::events_t<Index,Coord> events;
    std::pmr::vector<Index> intrs_tmp[2];
    detail::sweep_t<Index,Coord> sweep;
    detail::broken_starts_t<Index,Coord> broken_starts;
    std::pmr::vector<Index> broken_ends;
    detail::rand_generator rgen;
    Index original_i;
    bool ran;

    std::pmr::vector<detail::temp_polygon<Index>> loops_out;

    void self_intersection();
    void normalize();

public:
    point_tracker<Index> *pt;

    class point_sink {
        friend class normalizer;

        normalizer &n;
        point_t<Coord> prev;
        Index first_i;
        bool started;

        point_sink(normalizer &n) : n(n), started(false) {}
        point_sink(const point_sink&) = delete;
        point_sink(point_sink &&b) : n(b.n), prev(b.prev), first_i(b.first_i), started(b.started) {
            b.started = false;
        }

    public:
        void operator()(const point_t<Coord> &p,Index orig_i);
        Index last_orig_i() const { return n.original_i; }
        Index &last_orig_i() { return n.original_i; }
        ~point_sink();
    };

    explicit normalizer(
        point_tracker<Index> *pt=nullptr,
        std::pmr::memory_resource *_contig_mem=nullptr) :
        contig_mem(_contig_mem == nullptr ? std::pmr::get_default_resource() : _contig_mem),
        discrete_mem(contig_mem),
        lpoints(contig_mem),
        samples(contig_mem),
        intrs(&discrete_mem),
        events(contig_mem),
        intrs_tmp{
            std::pmr::vector<Index>(contig_mem),
            std::pmr::vector<Index>(contig_mem)},
        sweep(detail::sweep_cmp<Index,Coord>{lpoints},&discrete_mem),
        broken_starts(&discrete_mem),
        broken_ends(contig_mem),
        original_i(0),
        ran(false),
        loops_out(contig_mem),
        pt(pt) {}
    
    /* Important: you cannot pass the return value of "get_output()" from the
    same instance of "normalizer" to this function. If you want to feed the
    results back into the same instance, make a copy of the data and pass the
    copy here. */
    template<point_range<Coord> R> void add_loop(R &&loop);

    /* Important: you cannot pass the return value of "get_output()" from the
    same instance of "normalizer" to this function. If you want to feed the
    results back into the same instance, make a copy of the data and pass the
    copy here. */
    template<point_range_range<Coord> R> void add_loops(R &&loops) {
        for(auto &&loop : loops) add_loop(std::forward<decltype(loop)>(loop));
    }

    /* Return a "point sink".

    This is an alternative to adding loops with ranges. The return value is a
    functor that allows adding one point at a time. The destructor of the return
    value must be called before any other method of this instance of
    "normalizer" is called.

    Important: you cannot use the return value of "get_output()" from the same
    instance of "normalizer" if this function is called. If you want to feed the
    results back into the same instance, make a copy of the data. */
    point_sink add_loop() {
        if(ran) reset();
        return {*this};
    }

    /* Discard all loops added so far.
    
    The output returned by "get_output()" is invalidated. */
    void reset();

    /* The output of this function has references to data in this instance of
    "normalizer". The returned range is invalidated when "reset()",
    "add_loop()", or "add_loops()" is called or if the instance is destroyed. To
    keep the data, make a copy. The data is also not sequential. "normalize()"
    must be called before calling this function. */
    template<bool TreeOut>
    std::conditional_t<TreeOut,
        temp_polygon_tree_range<Index,Coord>,
        temp_polygon_range<Index,Coord>>
    get_output() &;

    template<bool TreeOut>
    std::conditional_t<TreeOut,
        temp_polygon_tree_range<Index,Coord>,
        temp_polygon_range<Index,Coord>>
    get_output() &&;
};

template<bool TreeOut,std::integral Index,coordinate Coord,point_range_range<Coord> Input>
std::conditional_t<TreeOut,
    temp_polygon_tree_range<Index,Coord>,
    temp_polygon_range<Index,Coord>>
normalize(
    Input &&input,
    point_tracker<Index> *pt=nullptr,
    std::pmr::memory_resource *contig_mem=nullptr)
{
    normalizer<Index,Coord> n{pt,contig_mem};
    n.add_loops(std::forward<Input>(input));
    return std::move(n).template get_output<TreeOut>();
}



template<std::integral Index,coordinate Coord>
template<point_range<Coord> R>
void normalizer<Index,Coord>::add_loop(R &&loop) {
    if(ran) reset();

    auto p_itr = std::ranges::begin(loop);
    auto itr_end = std::ranges::end(loop);
    if(p_itr == itr_end) return; // zero points

    point_t<Coord> prev = *p_itr++;
    Index first_i = static_cast<Index>(lpoints.size());

    samples.emplace_back(first_i,std::pmr::vector<Index>(contig_mem));

    prev = *p_itr++;
    bool first=true;
    for(point_t<Coord> p : std::ranges::subrange(p_itr,itr_end)) {
        ++original_i;
        if(first || prev != lpoints.back().data) {
            lpoints.emplace_back(prev,static_cast<Index>(lpoints.size()+1));
            if(pt) pt->point_added(original_i);
            first = false;
        }
        prev = p;
    }
    ++original_i;
    if(first || (prev != lpoints.back().data && prev != lpoints[first_i].data)) {
        lpoints.emplace_back(prev,0);
        if(pt) pt->point_added(original_i);
    }
    lpoints.back().next = first_i;
}

template<std::integral Index,coordinate Coord>
void normalizer<Index,Coord>::point_sink::operator()(const point_t<Coord> &p,Index orig_i) {
    if(started) {
        if(prev != n.lpoints.back().data) {
            n.lpoints.emplace_back(prev,static_cast<Index>(n.lpoints.size()+1));
            if(n.pt) n.pt->point_added(n.original_i);
        }
        prev = p;
    } else {
        prev = p;
        first_i = static_cast<Index>(n.lpoints.size());
        n.samples.emplace_back(first_i,std::pmr::vector<Index>(n.contig_mem));

        /* Normally points aren't added until another point is added or the
        destructor is called, but duplicate points aren't added anyway and
        adding it now means the "prev != n.lpoints.back().data" checks above and
        in the desctructor are always safe. */
        n.lpoints.emplace_back(prev,first_i+1);
        if(n.pt) n.pt->point_added(n.original_i);

        started = true;
    }
    n.original_i = orig_i;
}
template<std::integral Index,coordinate Coord>
normalizer<Index,Coord>::point_sink::~point_sink() {
    if(started) {
        if(prev != n.lpoints.back().data && prev != n.lpoints[first_i].data) {
            n.lpoints.emplace_back(prev,0);
            if(n.pt) n.pt->point_added(n.original_i);
        }
        n.lpoints.back().next = first_i;
    }
}

/* This is a modified version of the Bentley–Ottmann algorithm. Lines are broken
   at intersections. Two end-points touching does not count as an intersection.
   */
template<std::integral Index,coordinate Coord>
void normalizer<Index,Coord>::self_intersection() {
    using namespace detail;

    intrs_tmp[0].reserve(16);
    intrs_tmp[1].reserve(16);

    events.reserve(lpoints.size()*2 + 10);

    for(Index i=0; i<lpoints.size(); ++i) {
        Index j1 = i;
        Index j2 = lpoints[i].next;

        /* in the case of vertical line segments, the points must retain their
        original order due to an assumption made by vert_overlap() */
        if(lpoints[j1].data[0] > lpoints[j2].data[0]) std::swap(j1,j2);
        events.add_fb_events(lpoints,j1,j2);
    }

    for(Index i=0; i<samples.size(); ++i) {
        Index p = samples[i].p;
        events.add_event(p,i,event_type_t::calc_intr_sample);
    }

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

        POLY_OPS_ASSERT_SLOW(std::ranges::all_of(sweep,[&,e=e](segment<Index> s) {
            return e.ab.a_x(lpoints) <= s.a_x(lpoints) || e.ab.a_x(lpoints) <= s.b_x(lpoints);
        }));
        POLY_OPS_ASSERT_SLOW(consistent_order(sweep));

        switch(e.type) {
        case event_type_t::forward:
        case event_type_t::vforward:
            if(forward) {
                auto itr = std::get<0>(sweep.insert(e.ab));

                POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_F

                if(itr != sweep.begin() && check_intersection(events,sweep,lpoints,e.ab,*std::prev(itr),rgen,pt)) continue;
                ++itr;
                if(itr != sweep.end()) check_intersection(events,sweep,lpoints,e.ab,*itr,rgen,pt);
            } else {
                sweep.erase(e.ab);
                POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_FR
            }
            break;
        case event_type_t::backward:
        case event_type_t::vbackward:
            if(forward) {
                auto itr = sweep.find(e.line_ba());

                // if it's not in here, the line was split and no longer exists
                if(itr != sweep.end()) {
                    itr = sweep.erase(itr);

                    POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_B

                    if(itr != sweep.end() && itr != sweep.begin()) {
                        check_intersection(events,sweep,lpoints,*std::prev(itr),*itr,rgen,pt);
                    }

                    POLY_OPS_ASSERT_SLOW(!intersects_any(e.line_ba(),sweep,lpoints.data()));
                } else {
                    events.current().status = event_status_t::deleted;
                }
            } else {
                sweep.insert(events.current_start(lpoints).ab);
                POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_BR
            }
            break;
        case event_type_t::calc_balance_intr:
            if(forward) {
                /* This event has nothing to with the Bentley–Ottmann algorithm.
                We need to know the "line balance" number of each line. Since
                the line number only changes between intersections and all the
                lines form loops, we only need to compute the number for lines
                whose first point is an intersection. */

                dual_line_balance<Index,Coord> lb{lpoints.data(),e.ab.a,e.ab.b,intrs_tmp};
                for(const segment<Index> &s : sweep) lb.check(s);
                for(segment<Index> s : events.touching_removed(lpoints)) lb.check(s);
                std::tie(
                    lpoints[e.ab.a].line_bal,
                    lpoints[e.ab.b].line_bal) = lb.result();

                POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_CALC_BALANCE

                intrs.emplace(e.ab.a,std::pmr::vector<Index>(intrs_tmp[0],contig_mem));
                intrs.emplace(e.ab.b,std::pmr::vector<Index>(intrs_tmp[1],contig_mem));
                intrs_tmp[0].clear();
                intrs_tmp[1].clear();
            } else {
                intrs.erase(e.ab.a);
                intrs.erase(e.ab.b);
            }
            break;
        case event_type_t::calc_intr_sample:
            if(forward) {
                /* This event has nothing to with the Bentley–Ottmann algorithm.
                Instead of assuming that all holes are completely contained by
                outer polygons, we add one point from every loop to the list of
                points for which we will calculate the enclosure heirarchy (the
                other points are in "intrs"). */

                line_balance<Index,Coord> lb{lpoints.data(),e.p(),samples[e.sample_i()].hits};
                for(const segment<Index> &s : sweep) lb.check(s);
                for(segment<Index> s : events.touching_removed(lpoints)) lb.check(s);
                lpoints[e.p()].line_bal = lb.result();

                POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_CALC_SAMPLE
            } else {
                samples[e.sample_i()].hits.resize(0);
            }
            break;
        }
    }
}

template<std::integral Index,coordinate Coord>
void normalizer<Index,Coord>::normalize() {
    POLY_OPS_ASSERT(!ran);

    ran = true;

    intrs.clear();
    events.clear();
    sweep.clear();
    intrs_tmp[0].clear();
    intrs_tmp[1].clear();
    broken_starts.clear();
    broken_ends.clear();
    loops_out.clear();

    self_intersection();

    for(auto intr : intrs | std::views::keys) {
        detail::follow_balance<Index,Coord>(lpoints.data(),intr,broken_ends,broken_starts,pt);
    }
#if POLY_OPS_GRAPHICAL_DEBUG
    if(mc__) mc__->console_line_stream() << "broken_starts: " << pp(broken_starts,0) << "\nbroken_ends: " << pp(broken_ends,0);
#endif
    for(const auto &intr : samples) {
        // follow_balance must not be called with the same point twice
        if(!intrs.contains(intr.p)) {
            detail::follow_balance<Index,Coord>(lpoints.data(),intr.p,broken_ends,broken_starts,pt);
        }
    }

#if POLY_OPS_GRAPHICAL_DEBUG
    if(mc__) mc__->console_line_stream() << "broken_starts: " << pp(broken_starts,0) << "\nbroken_ends: " << pp(broken_ends,0);
    delegate_drawing_trimmed(lpoints);
#endif

    /* match all the points in broken_starts and broken_ends to make new loops
    where all the points have a line balance of 0 */
    for(auto intr : broken_ends) {
        auto &os = broken_starts[lpoints[lpoints[intr].next].data];

        POLY_OPS_ASSERT(os.size());

        if(pt) pt->point_merge(lpoints[intr].next,os.back());
        lpoints[intr].next = os.back();
        os.pop_back();
    }

    // there shouldn't be any left
    POLY_OPS_ASSERT(std::all_of(
        broken_starts.begin(),
        broken_starts.end(),
        [](const typename detail::broken_starts_t<Index,Coord>::value_type &v) { return std::get<1>(v).empty(); }));

#if POLY_OPS_GRAPHICAL_DEBUG
    delegate_drawing_trimmed(lpoints);
#endif

    for(auto &lp : lpoints) {
        if(lp.line_bal != 0) lp.line_bal = detail::UNDEF_LINE_BAL;
    }

    loops_out.clear();
    find_loops<Index,Coord>(lpoints,loops_out,contig_mem);
}

template<std::integral Index,coordinate Coord>
template<bool TreeOut>
std::conditional_t<TreeOut,
    temp_polygon_tree_range<Index,Coord>,
    temp_polygon_range<Index,Coord>>
normalizer<Index,Coord>::get_output() & {
    if(!ran) normalize();

    if constexpr(TreeOut) {
        auto top = loop_hierarchy(lpoints,loops_out,intrs,samples,contig_mem);
        return make_temp_polygon_tree_range<Index,Coord>(
            lpoints,
            loops_out,
            std::move(top));
    } else {
        return make_temp_polygon_range<Index,Coord>(lpoints,loops_out);
    }
}

template<std::integral Index,coordinate Coord>
template<bool TreeOut>
std::conditional_t<TreeOut,
    temp_polygon_tree_range<Index,Coord>,
    temp_polygon_range<Index,Coord>>
normalizer<Index,Coord>::get_output() && {
    if(!ran) normalize();

    if constexpr(TreeOut) {
        auto top = loop_hierarchy(lpoints,loops_out,intrs,samples,contig_mem);
        return make_temp_polygon_tree_range<Index,Coord>(
            std::move(lpoints),
            std::move(loops_out),
            std::move(top));
    } else {
        return make_temp_polygon_range<Index,Coord>(
            std::move(lpoints),
            std::move(loops_out));
    }
}

template<std::integral Index,coordinate Coord>
void normalizer<Index,Coord>::reset() {
    lpoints.clear();
    samples.clear();
    ran = false;
}

} // namespace poly_ops

#undef POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_F
#undef POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_FR
#undef POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_B
#undef POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_BR
#undef POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_CALC_BALANCE
#undef POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_CALC_SAMPLE
#undef POLY_OPS_DEBUG_STEP_BY_STEP_MISSED_INTR
#undef POLY_OPS_DEBUG_ITERATION

#endif
