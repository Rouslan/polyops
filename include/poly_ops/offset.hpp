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

namespace detail {
template<typename Index> using original_sets_t = mini_set_proxy_vector<Index,std::pmr::polymorphic_allocator<Index>>;
template<typename Index> using original_set_t = mini_flat_set<Index,std::pmr::polymorphic_allocator<Index>>;

template<typename Index> class origin_point_tracker : virtual public point_tracker<Index> {
    original_sets_t<Index> original_sets;

public:
    origin_point_tracker(std::pmr::memory_resource *contig_mem)
        : original_sets(contig_mem) {}

    void point_added(Index original_i) override {
        original_sets.emplace_back();
        original_sets.back().insert(original_i);
    }

    void new_intersection(Index a,Index b) override {
        for(Index i : {a,b}) {
            if(i >= original_sets.size()) {
                original_sets.emplace_back();
            }
        }
    }

    void point_merge(Index from,Index to) override {
        original_sets[to].merge(original_sets[from]);
    }
};

template<typename Index,typename Coord>
void add_offset_point(
    typename clipper<Index,Coord>::point_sink &sink,
    real_coord_t<Coord> magnitude,
    Coord arc_step_size,
    Index &orig_i,
    point_t<Coord> p1,
    point_t<Coord> p2,
    point_t<Coord> p3)
{
    point_t<real_coord_t<Coord>> offset = perp_vector<Coord>(p1,p2,magnitude);

    sink(p1 + vround<Coord>(offset),++orig_i);

    /* add a point for the new end of this line segment */
    sink(p2 + vround<Coord>(offset),++orig_i);

    if(triangle_winding(p1,p2,p3) * coord_ops<Coord>::unit(magnitude) < 0) {
        // it's concave so we need to approximate an arc

        real_coord_t<Coord> angle = coord_ops<Coord>::pi() - vangle<Coord>(p1-p2,p3-p2);
        Coord steps = coord_ops<Coord>::to_coord(magnitude * angle / arc_step_size);
        long lsteps = std::abs(static_cast<long>(steps));
        if(lsteps > 1) {
            real_coord_t<Coord> s = coord_ops<Coord>::sin(angle / steps);
            real_coord_t<Coord> c = coord_ops<Coord>::cos(angle / steps);

            for(long i=1; i<lsteps; ++i) {
                offset = {c*offset[0] - s*offset[1],s*offset[0] + c*offset[1]};
                sink(p2 + vround<Coord>(offset),++orig_i);
            }
        }
    }
}

} // namespace detail

template<typename Index,typename Coord,point_range<Coord> R>
void add_offset_loop(
    clipper<Index,Coord> &n,
    R &&input,
    bool_cat cat,
    real_coord_t<Coord> magnitude,
    std::type_identity_t<Coord> arc_step_size)
{
    auto sink = n.add_loop(cat);
    Index orig_i = sink.last_orig_i();
    auto itr = std::ranges::begin(input);
    auto end = std::ranges::end(input);
    if(itr == end) return;

    point_t<Coord> prev2(*itr++);
    point_t<Coord> first = prev2;

    if(itr == end) {
        /* A size of one can be handled as a special case. The result is a
        circle around the single point. For now, we just ignore such "loops". */
        sink.last_orig_i() = orig_i + 1;
        return;
    }

    point_t<Coord> prev1(*itr++);

    if(itr == end) {
        detail::add_offset_point(sink,magnitude,arc_step_size,orig_i,prev2,prev1,prev2);
        detail::add_offset_point(sink,magnitude,arc_step_size,orig_i,prev1,prev2,prev1);
        return;
    }

    for(auto &&p : std::ranges::subrange(itr,end)) {
        detail::add_offset_point(sink,magnitude,arc_step_size,orig_i,prev2,prev1,p);
        prev2 = prev1;
        prev1 = p;
    }
    detail::add_offset_point(sink,magnitude,arc_step_size,orig_i,prev2,prev1,first);
}

template<typename Index,typename Coord,point_range<Coord> R>
void add_offset_loop_subject(
    clipper<Index,Coord> &n,
    R &&input,
    real_coord_t<Coord> magnitude,
    std::type_identity_t<Coord> arc_step_size)
{
    add_offset_loop(n,std::forward<R>(input),bool_cat::subject,magnitude,arc_step_size);
}

template<typename Index,typename Coord,point_range<Coord> R>
void add_offset_loop_clip(
    clipper<Index,Coord> &n,
    R &&input,
    real_coord_t<Coord> magnitude,
    std::type_identity_t<Coord> arc_step_size)
{
    add_offset_loop(n,std::forward<R>(input),bool_cat::clip,magnitude,arc_step_size);
}

template<bool TreeOut,std::integral Index,coordinate Coord,point_range_range<Coord> Input>
std::conditional_t<TreeOut,
    temp_polygon_tree_range<Index,Coord>,
    temp_polygon_range<Index,Coord>>
offset(
    Input &&input,
    real_coord_t<Coord> magnitude,
    Coord arc_step_size,
    point_tracker<Index> *pt=nullptr,
    std::pmr::memory_resource *contig_mem=nullptr)
{
    clipper<Index,Coord> n{pt,contig_mem};
    for(auto &&loop : input) add_offset_loop(n,std::forward<decltype(loop)>(loop),bool_cat::subject,magnitude,arc_step_size);
    n.execute(bool_op::union_);
    return std::move(n).template get_output<TreeOut>();
}

} // namespace poly_ops

#endif
