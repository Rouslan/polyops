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

#ifndef POLY_OPS_OFFSET_HPP
#define POLY_OPS_OFFSET_HPP

#include <vector>
#include <map>
#include <tuple>
#include <algorithm>
#include <numbers>
#include <cstddef>
#include <utility>
#include <memory_resource>
#include <ranges>

#include "mini_flat_set.hpp"
#include "normalize.hpp"


namespace poly_ops {

/** Point tracking interface for `offset()`.

Note that the destructor is not virtual, and is instead protected.
*/
template<typename Index> class offset_point_tracker : virtual public point_tracker<Index> {
public:
    /** Called with the minimum number of points that will be added.

    The amount of points to be added is not known ahead of time, but will not be
    less than `minimum`. `minimum` will be zero if the loop input ranges given
    to `offset()` do not satisfy the `std::ranges::sized_range` concept.

    This is called once before any other method of this interface (including
    the base class's methods).

    @param minimum The amount of points that will definitely be added.
    */
    virtual void reserve(Index /*minimum*/) {}

    /** Called for every point initially added.

    Every added point has an implicit index (this index is unrelated to
    `original_i`). The first call of this method corresponds to points 0 to
    count-1. Every subsequent call starts with an index one greater than the
    greatest index of the previous call.

    `original_i` is the index of the input point that the added point
    corresponds to. The value is what the array index of the original point
    would be if all the input points were concatinated, in order, into a single
    array. This will be called with every point index from the input except for
    indices of consecutive duplicate points. `offset()` requires extra points
    due to points added to fill in the gaps caused by offsetting the line
    segments, thus `count` is provided, specifying how many points were added
    for a given input point.

    This is called after `reserve()` and before any other method of this
    interface.

    @param original_i The index of the input point that these added points
        correspond to.
    @param count The number of points added.
    */
    virtual void points_added(Index original_i,Index count) = 0;

protected:
    ~offset_point_tracker() = default;
};

namespace detail {
template<typename Index> using original_sets_t = mini_set_proxy_vector<Index,std::pmr::polymorphic_allocator<Index>>;
template<typename Index> using original_set_t = mini_flat_set<Index,std::pmr::polymorphic_allocator<Index>>;

template<typename Index> class origin_point_tracker : virtual public offset_point_tracker<Index> {
    std::pmr::vector<Index> indices;
    original_sets_t<Index> original_sets;

public:
    origin_point_tracker(std::pmr::memory_resource *contig_mem)
        : indices(contig_mem), original_sets(contig_mem) {}

    void points_added(Index original_i,Index count) override {
        indices.insert(indices.end(),count,static_cast<Index>(original_sets.size()));
        original_sets.emplace_back();
        original_sets.back().insert(original_i);
    }

    void new_intersection(Index a,Index b) override {
        for(Index i : {a,b}) {
            if(i >= indices.size()) {
                indices.push_back(static_cast<Index>(original_sets.size()));
                original_sets.emplace_back();
            }
        }
    }

    void point_merge(Index from,Index to) override {
        Index from_os = indices[from];
        Index to_os = indices[to];
        if(from_os != to_os) {
            original_sets[to_os].merge(original_sets[from_os]);
            indices[from] = to_os;
        }
    }
};

template<typename Index,typename Coord>
struct offset_polygon_point {
    std::pmr::vector<loop_point<Index,Coord>> lpoints;
    real_coord_t<Coord> magnitude;
    Coord arc_step_size;
    Index original_index;
    Index lfirst;
    std::size_t last_size;
    offset_point_tracker<Index> *pt;

    offset_polygon_point(
        real_coord_t<Coord> magnitude,
        Coord arc_step_size,
        Index reserve_size,
        std::pmr::memory_resource *contig_mem,
        offset_point_tracker<Index> *pt)
        : lpoints(contig_mem), magnitude(magnitude),
          arc_step_size(arc_step_size), original_index(0), last_size(0), pt(pt)
    {
        lpoints.reserve(reserve_size*2 + reserve_size/3);
        if(pt) pt->reserve(reserve_size);
    }

    void handle_pt() {
        if(!pt) return;

        POLY_OPS_ASSERT(lpoints.size() >= last_size);
        std::size_t d = lpoints.size()-last_size;
        if(!d) return;

        pt->points_added(original_index,static_cast<Index>(d));
        last_size = lpoints.size();
    }

    void operator()(
        const point_t<Coord> &p1,
        const point_t<Coord> &p2,
        const point_t<Coord> &p3,
        bool first,
        bool last)
    {
        if(first) lfirst = static_cast<Index>(last_size);

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
            lpoints.emplace_back(offset_point,lpoints.size()+1);
        }

        handle_pt();

        /* add a point for the new end of this line segment */
        lpoints.emplace_back(p2 + vround<Coord>(offset),lpoints.size()+1);

        if(triangle_winding(p1,p2,p3) * std::copysign(1.0,magnitude) < 0) {
            // it's concave so we need to approximate an arc

            real_coord_t<Coord> angle = coord_ops<Coord>::pi() - vangle<Coord>(p1-p2,p3-p2);
            Coord steps = coord_ops<Coord>::to_coord(magnitude * angle / arc_step_size);
            long lsteps = std::abs(static_cast<long>(steps));
            if(lsteps > 1) {
                real_coord_t<Coord> s = coord_ops<Coord>::sin(angle / steps);
                real_coord_t<Coord> c = coord_ops<Coord>::cos(angle / steps);

                for(long i=1; i<lsteps; ++i) {
                    offset = {c*offset[0] - s*offset[1],s*offset[0] + c*offset[1]};
                    offset_point = p2 + vround<Coord>(offset);

                    if(offset_point != lpoints.back().data) {
                        lpoints.emplace_back(offset_point,lpoints.size()+1);
                    }
                }
            }
        }

        if(last) {
            if(lpoints.back().data == lpoints[lfirst].data) {
                lpoints.pop_back();
            }
            lpoints.back().next = lfirst;

            handle_pt();
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
std::tuple<std::pmr::vector<loop_point<Index,Coord>>,intr_array_t<Index>> offset_polygon(
    const Input &input,
    real_coord_t<Coord> magnitude,
    Coord arc_step_size,
    std::pmr::memory_resource *contig_mem,
    offset_point_tracker<Index> *pt)
{
    offset_polygon_point<Index,Coord> opp(
        magnitude,
        arc_step_size,
        total_point_count<Coord,Input>::doit(input),
        contig_mem,
        pt);

    /* get the index of one point from every loop */
    intr_array_t<Index> to_sample(contig_mem);
    to_sample.reserve(std::ranges::size(input));

    for(auto &&loop : input) {
        to_sample.emplace_back(static_cast<Index>(opp.lpoints.size()),std::pmr::vector<Index>(contig_mem));
        offset_polygon_loop(opp,loop);
    }

    return {std::move(opp.lpoints),std::move(to_sample)};
}

} // namespace detail

template<bool TreeOut,std::integral Index,coordinate Coord,point_range_range<Coord> Input>
std::conditional_t<TreeOut,
    temp_polygon_tree_range<Index,Coord>,
    temp_polygon_range<Index,Coord>>
offset(
    Input &&input,
    real_coord_t<Coord> magnitude,
    Coord arc_step_size,
    offset_point_tracker<Index> *pt,
    std::pmr::memory_resource *contig_mem,
    std::pmr::memory_resource *discrete_mem)
{
    POLY_OPS_ASSERT(contig_mem && discrete_mem);

    auto [lpoints,to_sample] = detail::offset_polygon<Index,Coord,Input>(std::forward<Input>(input),magnitude,arc_step_size,contig_mem,pt);
    return detail::normalize_and_package<TreeOut,Index,Coord>(
        std::move(lpoints),
        std::move(to_sample),
        contig_mem,
        discrete_mem,
        pt);
}

template<bool TreeOut,std::integral Index,coordinate Coord,point_range_range<Coord> Input>
std::conditional_t<TreeOut,
    temp_polygon_tree_range<Index,Coord>,
    temp_polygon_range<Index,Coord>>
offset(
    Input &&input,
    real_coord_t<Coord> magnitude,
    Coord arc_step_size,
    offset_point_tracker<Index> *pt=nullptr,
    std::pmr::memory_resource *contig_mem=nullptr)
{
    if(contig_mem == nullptr) contig_mem = std::pmr::get_default_resource();

    std::pmr::unsynchronized_pool_resource dm(contig_mem);
    return offset<TreeOut,Index,Coord,Input>(std::forward<Input>(input),magnitude,arc_step_size,pt,contig_mem,&dm);
}

} // namespace poly_ops

#endif
