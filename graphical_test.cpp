
#include <type_traits>
#include <exception>
#include <span>
#include <vector>
#include <iostream>
#include <string_view>
#include <memory>
#include <cstring>

#include "server.hpp"

#define DEBUG_STEP_BY_STEP 1
#define POLY_OPS_GRAPHICAL_DEBUG 1

#if DEBUG_STEP_BY_STEP
#define DEBUG_STEP_BY_STEP_LINE_BALANCE_CHECK_1 \
    std::cout << "lines " << p << " - " << points[p].next << " and " << \
        (a_is_main ? s.a : s.b) << " - " << (a_is_main ? s.b : s.a)

#define DEBUG_STEP_BY_STEP_LINE_BALANCE_CHECK_2 \
    std::cout << " intersect\n"

#define DEBUG_STEP_BY_STEP_LINE_BALANCE_CHECK_3 \
    else std::cout << " don't intersect\n"

#define DEBUG_STEP_BY_STEP_ADD_EVENT do { \
    const char *type_str__; \
    switch(t) { \
    case event_type_t::forward: \
        type_str__ = "FORWARD"; \
        break; \
    case event_type_t::backward: \
        type_str__ = "BACKWARD"; \
        break; \
    case event_type_t::calc_balance: \
        type_str__ = "CALC_BALANCE"; \
        break; \
    } \
    mc__->console_line_stream() << "Event added: " << sa << " - " << sb \
        << ", " << type_str__; } while(false)

#define DEBUG_STEP_BY_STEP_CHECK_INTERSECTION_1 \
    mc__->console_line_stream() << "intersection between " << s1 << " and " \
        << s2 << ": yes"

#define DEBUG_STEP_BY_STEP_CHECK_INTERSECTION_2 \
    mc__->console_line_stream() << "intersection between " << s1 << " and " \
        << s2 << ": no"

#define DEBUG_STEP_BY_STEP_SELF_INTERSECTION_1 do { \
    mc__->console_line_stream() << "event: FORWARD at " << e.ab << "  x: " \
        << e.ab.a_x(lpoints); \
    delegate_drawing(lpoints,sweep,e,original_sets); } while(false)

#define DEBUG_STEP_BY_STEP_SELF_INTERSECTION_2 \
    emit_line_before_after("line before: ",before,sweep.end())

#define DEBUG_STEP_BY_STEP_SELF_INTERSECTION_3 \
    emit_line_before_after("line after: ",itr,sweep.end())

#define DEBUG_STEP_BY_STEP_SELF_INTERSECTION_4 do { \
    mc__->console_line_stream() << "event: BACKWARD at " << e.ab.a << "  x: " \
        << e.ab.a_x(lpoints); \
    delegate_drawing(lpoints,sweep,e,original_sets); } while(false)

#define DEBUG_STEP_BY_STEP_SELF_INTERSECTION_5 \
    emit_line_before_after("line after: ",itr,sweep.end())

#define DEBUG_STEP_BY_STEP_SELF_INTERSECTION_6 \
    emit_line_before_after("line before: ",before,sweep.end())

#define DEBUG_STEP_BY_STEP_SELF_INTERSECTION_8 do { \
    mc__->console_line_stream() << "event: CALC_BALANCE at " << e.ab.a \
        << " - " << lpoints[e.ab.a].next << "  x: " << e.ab.a_x(lpoints); \
    delegate_drawing(lpoints,sweep,e,original_sets); } while(false)

#define DEBUG_STEP_BY_STEP_SELF_INTERSECTION_9 \
    mc__->console_line_stream() << "line balance of " << e.ab.a << " - " \
        << lpoints[e.ab.a].next << " is " << lpoints[e.ab.a].line_bal
#endif // DEBUG_STEP_BY_STEP

typedef int32_t icoord_t;
typedef uint16_t index_t;

thread_local message_canvas *mc__;

namespace poly_ops::detail {
template<typename Index> struct event;
template<typename Index,typename Coord> struct loop_point;
template<typename Index> struct segment;
}
template<typename Sweep,typename OSet> void delegate_drawing(
    const std::pmr::vector<poly_ops::detail::loop_point<index_t,icoord_t>> &lpoints,
    const Sweep &sweep,
    const poly_ops::detail::event<index_t> &e,
    const OSet &original_sets);
template<typename OSet> void delegate_drawing_trimmed(
    const std::pmr::vector<poly_ops::detail::loop_point<index_t,icoord_t>> &lpoints,
    const OSet &original_sets);
std::ostream &operator<<(std::ostream &os,const poly_ops::detail::segment<index_t> &x);
void emit_line_before_after(const char *pre,auto itr,auto end) {
    auto out = mc__->console_line_stream();
    out << pre;
    if(itr == end) out << "none";
    else out << *itr;
}

#include "poly_ops.hpp"
#include "stream_output.hpp"

using namespace poly_ops;
using namespace std::literals::string_view_literals;

thread_local icoord_t (*input_coords)[2];
thread_local index_t input_sizes[1];

template<typename T> std::tuple<point_t<icoord_t>,point_t<icoord_t>> point_extent(const T &lpoints,icoord_t padding=10) {
    point_t<icoord_t> max_corner, min_corner;

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

std::ostream &operator<<(std::ostream &os,const point_t<icoord_t> &x) {
    return os << x[0] << ',' << x[1];
}

std::ostream &operator<<(std::ostream &os,const detail::loop_point<index_t,icoord_t> &x) {
    return os << '{' << x.data << "}," << x.original_set << ',' << x.next;
}

std::ostream &operator<<(std::ostream &os,const detail::segment<index_t> &x) {
    return os << x.a << " - " << x.b;
}

struct draw_point {
    point_t<icoord_t> data;
    index_t next;
    enum state {NORMAL=0,SWEEP,INVERTED,NESTED,TO_ORIGINAL} type;
};

auto to_json_value(const draw_point &p) {
    using namespace json;

    return obj(
        attr("data") = array_range(p.data),
        attr("next") = p.next,
        attr("state") = static_cast<int>(p.type));
}

void dump_original_points(
    const std::pmr::vector<detail::loop_point<index_t,icoord_t>> &lpoints,
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
    const std::pmr::vector<detail::loop_point<index_t,icoord_t>> &lpoints,
    const Sweep &sweep,
    const detail::event<index_t> &e,
    const OSet &original_sets)
{
    std::vector<draw_point> draw_lines;

    draw_lines.resize(lpoints.size());
    for(size_t i=0; i<lpoints.size(); ++i) {
        draw_point::state state = draw_point::NORMAL;
        if(sweep.count(detail::segment<index_t>(i,lpoints[i].next)) || sweep.count(detail::segment<index_t>(lpoints[i].next,i))) {
            state = draw_point::SWEEP;
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
            for(detail::segment<index_t> s : sweep) out << "  " << s << '\n';
        } else if(msg == u8"dump_orig_points"sv) {
            dump_original_points(lpoints,original_sets);
        } else throw std::runtime_error("unexpected command received from client");
    }
}

template<typename OSet> void delegate_drawing_trimmed(
    const std::pmr::vector<detail::loop_point<index_t,icoord_t>> &lpoints,
    const OSet &original_sets)
{
    std::vector<draw_point> draw_lines;

    draw_lines.resize(lpoints.size());
    for(size_t i=0; i<lpoints.size(); ++i) {
        draw_point::state state = draw_point::NORMAL;
        if(lpoints[i].line_bal < 0) state = draw_point::INVERTED;
        else if(lpoints[i].line_bal > 0) state = draw_point::NESTED;
        else {
            for(index_t orig : original_sets[lpoints[i].original_set]) {
                const icoord_t *op = input_coords[orig];
                draw_lines.emplace_back(point_t<icoord_t>{op[0],op[1]},i,draw_point::TO_ORIGINAL);
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

int main(int argc,char **argv) {
    if(argc != 2) {
        std::cerr << "Usage: inset_test FILENAME" << std::endl;
        return 1;
    }

    try {
        run_message_server(argv[1],[](message_canvas &mc) {
            cblob data = mc.get_binary();
            if(data.size % sizeof(icoord_t)) throw std::runtime_error("invalid data received");
            mc__ = &mc;
            input_sizes[0] = data.size / sizeof(icoord_t[2]);
            std::unique_ptr<icoord_t[2]> _input_coords(new icoord_t[input_sizes[0]][2]);
            std::memcpy(_input_coords.get(),data.data,data.size);
            input_coords = _input_coords.get();
            offset_stroke_triangulate<index_t,icoord_t>(
                basic_polygon<index_t,icoord_t,icoord_t[2]>(
                    input_coords,
                    std::span(input_sizes)),
                -140,
                80);
        });
    } catch(const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
