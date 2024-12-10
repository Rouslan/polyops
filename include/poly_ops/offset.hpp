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
#include "clip.hpp"


namespace poly_ops {

namespace detail {
template<typename Index> using original_sets_t = mini_set_proxy_vector<Index,std::pmr::polymorphic_allocator<Index>>;
template<typename Index> using original_set_replacements_t = indexed_mini_set_proxy_vector<Index,std::pmr::polymorphic_allocator<Index>>;
template<typename Index> using original_set_t = mini_flat_set<Index,std::pmr::polymorphic_allocator<Index>>;

template<typename Index,typename Coord>
void add_offset_point(
    typename clipper<Coord,Index>::point_sink &sink,
    real_coord_t<Coord> magnitude,
    Coord arc_step_size,
    Index orig_i1,
    Index orig_i2,
    point_t<Coord> p1,
    point_t<Coord> p2,
    point_t<Coord> p3)
{
    using real_t = real_coord_t<Coord>;

    POLY_OPS_ASSERT(p1 != p2);
    point_t<real_t> offset = perp_vector<Coord>(p1,p2,magnitude);

    /* add a point for the new end of the previous line segment */
    sink(p1 + vround<Coord>(offset),orig_i1);

    sink(p2 + vround<Coord>(offset),orig_i2);

    bool concave = triangle_winding(p1,p2,p3) > 0;
    if(magnitude < real_t(0)) concave = !concave;
    if(concave) {
        // we need to approximate an arc

        real_t angle = coord_ops<Coord>::pi() - vangle<Coord>(p1-p2,p3-p2);
        Coord steps = coord_ops<Coord>::to_coord(magnitude * angle / static_cast<real_t>(arc_step_size));
        long lsteps = std::abs(static_cast<long>(steps));
        if(lsteps > 1) {
            real_t s = coord_ops<Coord>::sin(angle / static_cast<real_t>(steps));
            real_t c = coord_ops<Coord>::cos(angle / static_cast<real_t>(steps));

            for(long i=1; i<lsteps; ++i) {
                offset = {c*offset[0] - s*offset[1],s*offset[0] + c*offset[1]};
                sink(p2 + vround<Coord>(offset),orig_i2);
            }
        }
    }
}

} // namespace detail

template<typename Coord,typename Index=std::size_t> struct point_and_origin {
    point_t<Coord> p;
    std::span<const Index> original_points;
};

template<typename Index=std::size_t> class origin_point_tracker final : public i_point_tracker<Index> {
    detail::original_sets_t<Index> original_sets;

public:
    origin_point_tracker()
        : original_sets(std::pmr::get_default_resource()) {}

    explicit origin_point_tracker(std::pmr::memory_resource *contig_mem)
        : original_sets(contig_mem ? contig_mem : std::pmr::get_default_resource()) {}
    
    origin_point_tracker(origin_point_tracker&&) noexcept = default;

    void point_added(Index original_i) override {
        original_sets.emplace_back();
        original_sets.back().insert(original_i);
    }

    void new_point_between(Index,Index) override {
        original_sets.emplace_back();
    }

    void point_merge(Index from,Index to) override {
        original_sets[to].merge(original_sets[from]);
    }

    void points_removed(Index n) override {
        original_sets.resize(original_sets.size()-static_cast<std::size_t>(n));
    }

    i_point_tracker<Index> *callbacks() { return this; }

    template<typename Coord> point_and_origin<Coord,Index> get_value(Index i,const point_t<Coord> &p) const {
        return {p,original_sets[i]};
    }

    void reset() override { original_sets.clear(); }
};

template<typename Coord,typename Index,typename Input>
void add_offset_loops(
    clipper<Coord,Index> &n,
    Input &&input,
    bool_set set,
    real_coord_t<Coord> magnitude,
    std::type_identity_t<Coord> arc_step_size,
    i_point_tracker<Index> *pt=nullptr)
{
    static_assert(point_range_or_range_range<Input,Coord>);
    static_assert(coordinate<Coord>);
    static_assert(std::integral<Index>);

    if constexpr(point_range_range<Input,Coord>) {
        for(auto &&loop: input) add_offset_loops(n,std::forward<decltype(loop)>(loop),set,magnitude,arc_step_size,pt);
    } else {
        /* for each point added, we need to know the previous and next points,
        thus the first point is added last, after the entire input iterator is
        traversed */

        Index orig_i;
        {
            auto sink = n.add_loop(set,pt);
            orig_i = sink.last_orig_i() + 1;
            auto itr = std::ranges::begin(input);
            auto end = std::ranges::end(input);
            if(itr == end) return;

            point_t<Coord> prev2(*itr);
            ++itr;
            point_t<Coord> first = prev2;

            point_t<Coord> prev1, second;
            
            do {
                if(itr == end) {
                    /* A size of one can be handled as a special case. The
                    result is a circle around the single point, but for now, we
                    just ignore such "loops". */
                    sink.last_orig_i() = orig_i;
                    return;
                }

                prev1 = second = *itr;
                ++itr;
                second = prev1;
            } while(prev1 == prev2);

            Index orig_i1 = orig_i;
            for(point_t<Coord> p : std::ranges::subrange(itr,end)) {
                if(p != prev1) {
                    detail::add_offset_point<Index,Coord>(sink,magnitude,arc_step_size,orig_i,orig_i+1,prev2,prev1,p);
                    prev2 = prev1;
                    prev1 = p;
                }
                ++orig_i;
            }
            if(prev2 != prev1) {
                detail::add_offset_point<Index,Coord>(sink,magnitude,arc_step_size,orig_i,orig_i+1,prev2,prev1,first);
            }
            ++orig_i;

            if(prev1 != first) {
                detail::add_offset_point<Index,Coord>(sink,magnitude,arc_step_size,orig_i,orig_i1,prev1,first,second);
            }
        }
        /* "last_orig_i" shouldn't be changed until "sink" goes out of scope,
        because it's used in sink's destructor */
        n.last_orig_i() = orig_i;
    }
}

template<typename Coord,typename Index,typename Tracker,typename Input>
void add_offset_loops(
    tclipper<Coord,Index,Tracker> &n,
    Input &&input,
    bool_set set,
    real_coord_t<Coord> magnitude,
    std::type_identity_t<Coord> arc_step_size)
{
    add_offset_loops(n.base,std::forward<Input>(input),set,magnitude,arc_step_size,n.tracker.callbacks());
}

template<typename Coord,typename Index,typename Input>
void add_offset_loops_subject(
    clipper<Coord,Index> &n,
    Input &&input,
    real_coord_t<Coord> magnitude,
    std::type_identity_t<Coord> arc_step_size)
{
    static_assert(point_range_or_range_range<Input,Coord>);
    static_assert(coordinate<Coord>);
    static_assert(std::integral<Index>);

    add_offset_loops(n,std::forward<Input>(input),bool_set::subject,magnitude,arc_step_size);
}

template<typename Coord,typename Index,typename Tracker,typename Input>
void add_offset_loops_subject(
    tclipper<Coord,Index,Tracker> &n,
    Input &&input,
    real_coord_t<Coord> magnitude,
    std::type_identity_t<Coord> arc_step_size)
{
    add_offset_loops(n,std::forward<Input>(input),bool_set::subject,magnitude,arc_step_size);
}

template<typename Coord,typename Index,typename Input>
void add_offset_loops_clip(
    clipper<Coord,Index> &n,
    Input &&input,
    real_coord_t<Coord> magnitude,
    std::type_identity_t<Coord> arc_step_size)
{
    static_assert(point_range_or_range_range<Input,Coord>);
    static_assert(coordinate<Coord>);
    static_assert(std::integral<Index>);

    add_offset_loops(n,std::forward<Input>(input),bool_set::clip,magnitude,arc_step_size);
}

template<typename Coord,typename Index,typename Tracker,typename Input>
void add_offset_loops_clip(
    tclipper<Coord,Index,Tracker> &n,
    Input &&input,
    real_coord_t<Coord> magnitude,
    std::type_identity_t<Coord> arc_step_size)
{
    add_offset_loops(n,std::forward<Input>(input),bool_set::clip,magnitude,arc_step_size);
}

template<bool TreeOut,typename Coord,typename Index=std::size_t,typename Input,typename Tracker>
std::conditional_t<TreeOut,
    temp_polygon_tree_range<Coord,Index,Tracker>,
    temp_polygon_range<Coord,Index,Tracker>>
offset(
    Input &&input,
    real_coord_t<Coord> magnitude,
    Coord arc_step_size,
    Tracker &&tracker,
    std::pmr::memory_resource *contig_mem=nullptr)
{
    static_assert(point_tracker<Tracker,Coord,Index>);
    static_assert(point_range_or_range_range<Input,Coord>);
    static_assert(coordinate<Coord>);
    static_assert(std::integral<Index>);

    tclipper<Coord,Index,Tracker> n{std::forward<Tracker>(tracker),contig_mem};
    add_offset_loops(n,std::forward<Input>(input),bool_set::subject,magnitude,arc_step_size);
    return std::move(n).template execute<TreeOut>(bool_op::union_);
}

template<bool TreeOut,typename Coord,typename Index=std::size_t,typename Input>
std::conditional_t<TreeOut,
    temp_polygon_tree_range<Coord,Index>,
    temp_polygon_range<Coord,Index>>
offset(
    Input &&input,
    real_coord_t<Coord> magnitude,
    Coord arc_step_size,
    std::pmr::memory_resource *contig_mem=nullptr)
{
    return offset<TreeOut,Coord,Index,Input,null_tracker<Coord,Index>>(std::forward<Input>(input),magnitude,arc_step_size,{},contig_mem);
}

template<coordinate Coord,std::integral Index=std::size_t> using origin_tracked_clipper = tclipper<Coord,Index,origin_point_tracker<Index>>;

} // namespace poly_ops

#endif
