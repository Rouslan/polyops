#include <type_traits>
#include <exception>
#include <span>
#include <vector>
#include <iostream>
#include <string_view>
#include <memory>
#include <cstring>

#include "server.hpp"

#ifndef DEBUG_STEP_BY_STEP
#define DEBUG_STEP_BY_STEP 0
#endif
#define POLY_OPS_GRAPHICAL_DEBUG 1
#define POLY_OPS_ASSERT(X) if(!(X)) throw assertion_failure{#X}
#define POLY_OPS_ASSERT_SLOW POLY_OPS_ASSERT

#if DEBUG_STEP_BY_STEP
#define DEBUG_STEP_BY_STEP_EVENT_F if(graphical_debug) { \
    { \
        auto out = mc__->console_line_stream(); \
        out << "event: FORWARD at " << e.ab << "  x: " << e.ab.a_x(lpoints); \
        emit_line_before_after(out, \
            (itr == sweep.begin() ? sweep.end() : std::prev(itr)), \
            std::next(itr), \
            sweep.end()); \
    } \
    delegate_drawing(lpoints,sweep,e,original_sets); } (void)0

#define DEBUG_STEP_BY_STEP_EVENT_B if(graphical_debug) { \
    { \
        auto out = mc__->console_line_stream(); \
        out << "event: BACKWARD at " << e.line_ba() << "  x: " << e.ab.a_x(lpoints); \
        emit_line_before_after(out, \
            (itr == sweep.begin() ? sweep.end() : std::prev(itr)), \
            itr, \
            sweep.end()); \
    } \
    delegate_drawing(lpoints,sweep,e,original_sets); } (void)0

#define DEBUG_STEP_BY_STEP_EVENT_CALC_BALANCE if(graphical_debug) { \
    std::vector<Index> hits1, hits2; \
    line_balance<Index,Coord> lb{lpoints.data(),e.ab.a,e.ab.b}; \
    for(auto s : sweep) { \
        auto [a,b] = lb.check(s); \
        Index i = s.a_is_main(lpoints) ? s.a : s.b; \
        if(a) hits1.push_back(i); \
        if(b) hits2.push_back(i); \
    } \
    for(auto s : events.touching_removed(lpoints)) { \
        auto [a,b] = lb.check(s); \
        Index i = s.a_is_main(lpoints) ? s.a : s.b; \
        if(a) hits1.push_back(i); \
        if(b) hits2.push_back(i); \
    } \
    std::tie( \
        lpoints[e.ab.a].line_bal, \
        lpoints[e.ab.b].line_bal) = lb.result(); \
    report_hits(e.ab.a,hits1,std::get<0>(lb.result())); \
    report_hits(e.ab.b,hits2,std::get<1>(lb.result())); \
    break; \
}

#define DEBUG_STEP_BY_STEP_LB_CHECK_RET_TYPE std::tuple<bool,bool>
#define DEBUG_STEP_BY_STEP_LB_CHECK_RETURN(A,B) return {A,B}
#define DEBUG_STEP_BY_STEP_LB_CHECK_FF {false,false}

#define DEBUG_STEP_BY_STEP_MISSED_INTR report_missed_intr(s1,s2)
#endif // DEBUG_STEP_BY_STEP

typedef int32_t coord_t;
typedef uint16_t index_t;

struct assertion_failure : std::logic_error {
    using logic_error::logic_error;
};

thread_local message_canvas *mc__ = nullptr;
bool graphical_debug = false;

namespace poly_ops::detail {
template<typename Index> struct event;
template<typename Index,typename Coord> struct loop_point;
template<typename Index> struct segment;
}
template<typename Sweep,typename OSet> void delegate_drawing(
    const std::pmr::vector<poly_ops::detail::loop_point<index_t,coord_t>> &lpoints,
    const Sweep &sweep,
    const poly_ops::detail::event<index_t> &e,
    const OSet &original_sets);
template<typename OSet> void delegate_drawing_trimmed(
    const std::pmr::vector<poly_ops::detail::loop_point<index_t,coord_t>> &lpoints,
    const OSet &original_sets);
std::ostream &operator<<(std::ostream &os,const poly_ops::detail::segment<index_t> &x);
template<typename Itr> void emit_line_before_after(std::ostream &out,Itr before,Itr after,Itr end) {
    out << "\n  line before: ";
    if(before == end) out << "none";
    else out << *before;
    out << "\n  line after: ";
    if(after == end) out << "none";
    else out << *after;
}
template<typename T> struct _pp;
template<typename T> _pp<T> pp(T &&x,unsigned int indent=0);

void report_hits(index_t index,const std::vector<index_t> &hits,int balance);
void report_missed_intr(poly_ops::detail::segment<index_t> s1,poly_ops::detail::segment<index_t> s2);

#include "poly_ops.hpp"
#include "stream_output.hpp"

using namespace poly_ops;
using namespace std::literals::string_view_literals;

thread_local coord_t (*input_coords)[2];
thread_local index_t input_sizes[1];

template<typename T> std::tuple<point_t<coord_t>,point_t<coord_t>> point_extent(const T &lpoints,coord_t padding=10) {
    point_t<coord_t> max_corner, min_corner;

    if(lpoints.size()) {
        max_corner = min_corner = lpoints[0].data;
        for(auto &p : lpoints) {
            max_corner[0] = std::max(max_corner[0],p.data[0]);
            max_corner[1] = std::max(max_corner[1],p.data[1]);

            min_corner[0] = std::min(min_corner[0],p.data[0]);
            min_corner[1] = std::min(min_corner[1],p.data[1]);
        }
    } else {
        max_corner = min_corner = {0,0};
    }

    min_corner[0] -= padding;
    min_corner[1] -= padding;
    max_corner[0] += padding;
    max_corner[1] += padding;
    return {max_corner,min_corner};
}

std::ostream &operator<<(std::ostream &os,const point_t<coord_t> &x) {
    return os << x[0] << ',' << x[1];
}

std::ostream &operator<<(std::ostream &os,const detail::loop_point<index_t,coord_t> &x) {
    return os << '{' << x.data << "}," << x.original_set << ',' << x.next;
}

std::ostream &operator<<(std::ostream &os,const detail::segment<index_t> &x) {
    return os << x.a << " - " << x.b;
}

namespace poly_ops {
    template<typename Coord> auto to_json_value(const point_t<Coord> &p) {
        return json::array_range(p);
    }
}

void report_hits(index_t index,const std::vector<index_t> &hits,int balance) {
    using namespace json;
    mc__->message(obj(
        attr("command") = "linebalance",
        attr("point") = index,
        attr("hits") = array_range(hits),
        attr("balance") = balance));
}

void report_missed_intr(detail::segment<index_t> s1,detail::segment<index_t> s2) {
    if(graphical_debug) {
        auto out = mc__->console_line_stream();
        out << s1 << " and " << s2 << " intersect";
    }
}

struct draw_point {
    point_t<coord_t> data;
    index_t next;
    enum state {NORMAL=0,SWEEP,INVERTED,NESTED,TO_ORIGINAL} type;
};

auto to_json_value(const draw_point &p) {
    using namespace json;

    return obj(
        attr("data") = p.data,
        attr("next") = p.next,
        attr("state") = static_cast<int>(p.type));
}

void dump_original_points(
    const std::pmr::vector<detail::loop_point<index_t,coord_t>> &lpoints,
    const auto &original_sets)
{
    std::vector<bool> emitted(original_sets.size(),false);

    auto out = mc__->console_line_stream();

    out << "original points:\n";
    for(size_t i=0; i<lpoints.size(); ++i) {
        index_t oset = lpoints[i].original_set;
        out << i << ": " << oset;
        if(!emitted[oset]) {
            emitted[oset] = true;
            out << " (" << delimited(original_sets[oset]) << ')';
        }
        out << '\n';
    }
}

template<typename Sweep,typename OSet> void delegate_drawing(
    const std::pmr::vector<detail::loop_point<index_t,coord_t>> &lpoints,
    const Sweep &sweep,
    const detail::event<index_t> &e,
    const OSet &original_sets)
{
    if(!graphical_debug) return;

    std::vector<draw_point> draw_lines;

    draw_lines.resize(lpoints.size());
    for(size_t i=0; i<lpoints.size(); ++i) {
        draw_point::state state = draw_point::NORMAL;
        if(sweep.count(detail::segment<index_t>(i,lpoints[i].next)) || sweep.count(detail::segment<index_t>(lpoints[i].next,i))) {
            state = draw_point::SWEEP;
        } else if(lpoints[i].line_bal != detail::loop_point<index_t,coord_t>::UNDEF_LINE_BAL) {
            if(lpoints[i].line_bal < 0) state = draw_point::INVERTED;
            else if(lpoints[i].line_bal > 0) state = draw_point::NESTED;
        }

        draw_lines[i] = {lpoints[i].data,lpoints[i].next,state};
    }

    mc__->message(json::obj(
        json::attr("command") = "draw",
        json::attr("points") = json::array_range(draw_lines),
        json::attr("currentPoint") = json::array_range(lpoints[e.ab.a].data),
        json::attr("indexedLineCount") = lpoints.size()));

    for(;;) {
        std::u8string_view msg = mc__->get_text();
        if(msg == u8"continue"sv) {
            break;
        } else if(msg == u8"dump_sweep"sv) {
            auto out = mc__->console_line_stream();
            out << "sweep items:\n";
            for(auto s : sweep) out << "  " << s << '\n';
        } else if(msg == u8"dump_orig_points"sv) {
            dump_original_points(lpoints,original_sets);
        } else throw std::runtime_error("unexpected command received from client");
    }
}

template<typename OSet> void delegate_drawing_trimmed(
    const std::pmr::vector<detail::loop_point<index_t,coord_t>> &lpoints,
    const OSet &original_sets)
{
    if(!graphical_debug) return;

    std::vector<draw_point> draw_lines;

    draw_lines.resize(lpoints.size());
    for(size_t i=0; i<lpoints.size(); ++i) {
        draw_point::state state = draw_point::NORMAL;
        if(lpoints[i].line_bal < 0) state = draw_point::INVERTED;
        else if(lpoints[i].line_bal > 0) state = draw_point::NESTED;
        else {
            for(index_t orig : original_sets[lpoints[i].original_set]) {
                const coord_t *op = input_coords[orig];
                draw_lines.emplace_back(point_t<coord_t>{op[0],op[1]},i,draw_point::TO_ORIGINAL);
            }
        }

        draw_lines[i] = {lpoints[i].data,lpoints[i].next,state};
    }

    mc__->message(json::obj(
        json::attr("command") = "draw",
        json::attr("points") = json::array_range(draw_lines),
        json::attr("currentPoint") = nullptr,
        json::attr("indexedLineCount") = lpoints.size()));

    for(;;) {
        std::u8string_view msg = mc__->get_text();
        if(msg == u8"continue") {
            break;
        } else if(msg == u8"dump_sweep") {
            // "sweep" not available. Do nothing.
        } else if(msg == u8"dump_orig_points") {
            dump_original_points(lpoints,original_sets);
        } else throw std::runtime_error("unexpected command received from client");
    }
}
