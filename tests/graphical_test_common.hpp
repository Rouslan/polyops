#include <type_traits>
#include <exception>
#include <span>
#include <vector>
#include <iostream>
#include <string_view>
#include <memory>
#include <cstring>
#include <chrono>
#include <charconv>

#include "server.hpp"

#ifndef DEBUG_STEP_BY_STEP
#define DEBUG_STEP_BY_STEP 0
#endif
#define POLY_OPS_GRAPHICAL_DEBUG 1
#define POLY_OPS_ASSERT(X) if(!(X)) throw assertion_failure{#X}
#define POLY_OPS_ASSERT_SLOW POLY_OPS_ASSERT

#if DEBUG_STEP_BY_STEP
#define POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_F if(graphical_debug) { \
    emit_forward_backward( \
        true, \
        e.ab, \
        (itr == sweep.begin() ? sweep.end() : std::prev(itr)), \
        std::next(itr), \
        sweep, \
        events); \
    delegate_drawing(lpoints,sweep,e); \
}

#define POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_FR if(graphical_debug) { \
    auto out = mc__->console_line_stream(); \
    out << "undoing: FORWARD at " << e.ab; \
}

#define POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_B if(graphical_debug) { \
    emit_forward_backward( \
        false, \
        e.ab, \
        (itr == sweep.begin() ? sweep.end() : std::prev(itr)), \
        itr, \
        sweep, \
        events); \
    delegate_drawing(lpoints,sweep,e); \
}

#define POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_BR if(graphical_debug) { \
    auto out = mc__->console_line_stream(); \
    out << "undoing: BACKWARD at " << e.line_ba(); \
}

#define POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_CALC_BALANCE if(graphical_debug) { \
    report_hits(e.ab.a,intrs_tmp[0],std::get<0>(lb.result())); \
    report_hits(e.ab.b,intrs_tmp[1],std::get<1>(lb.result())); \
}

#define POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_CALC_SAMPLE if(graphical_debug) { \
    report_hits(i,samples.back().hits,lb.result()); \
    auto out = mc__->console_line_stream(); \
    out << "sweep: " << delimited(sweep); \
    out << "\nbefore: " << delimited(events.touching_removed(lpoints)); \
    out << "\nafter: " << delimited(events.touching_pending(lpoints)); \
}

#define POLY_OPS_DEBUG_STEP_BY_STEP_MISSED_INTR report_missed_intr(s1,s2)

#define POLY_OPS_DEBUG_ITERATION if(timeout) { \
    if(std::chrono::steady_clock::now() > timeout_expiry) { \
        timeout = false; \
        throw timed_out{}; \
    } \
}
#endif // DEBUG_STEP_BY_STEP

typedef int32_t coord_t;
typedef uint16_t index_t;

struct test_failure : std::exception {
    virtual std::ostream &emit(std::ostream &os) const = 0;
    const char *what() const noexcept override {
        return "test failure";
    }
};
struct assertion_failure : test_failure {
    const char *assertion_str;
    explicit assertion_failure(const char *assertion) : assertion_str(assertion) {}

    std::ostream &emit(std::ostream &os) const override {
        return os << "assertion failure: " << assertion_str;
    }
};
struct timed_out : test_failure {
    std::ostream &emit(std::ostream &os) const override {
        return os << "test timed out";
    }
};
std::ostream &operator<<(std::ostream &os,const test_failure &f) {
    return f.emit(os);
}

thread_local size_t increment_amount;
thread_local message_canvas *mc__ = nullptr;
bool graphical_debug = false;
bool timeout = false;
std::chrono::steady_clock::time_point timeout_expiry;

namespace poly_ops::detail {
template<typename Index> struct event;
template<typename Index,typename Coord> struct loop_point;
template<typename Index> struct segment;
}
template<typename Sweep> void delegate_drawing(
    const std::pmr::vector<poly_ops::detail::loop_point<index_t,coord_t>> &lpoints,
    const Sweep &sweep,
    const poly_ops::detail::event<index_t> &e);
void delegate_drawing_trimmed(
    const std::pmr::vector<poly_ops::detail::loop_point<index_t,coord_t>> &lpoints);
std::ostream &operator<<(std::ostream &os,const poly_ops::detail::segment<index_t> &x);
template<typename Sweep,typename Events> void emit_forward_backward(
    bool forward,
    poly_ops::detail::segment<index_t> seg,
    std::ranges::iterator_t<const Sweep> before,
    std::ranges::iterator_t<const Sweep> after,
    const Sweep &sweep,
    const Events &events);
template<typename T> struct _pp;
template<typename T> _pp<T> pp(T &&x,unsigned int indent);
template<typename T> struct delimited_t;
template<typename R> delimited_t<R> delimited(R &&items);

void report_hits(index_t index,const std::pmr::vector<index_t> &hits,int balance);
void report_missed_intr(poly_ops::detail::segment<index_t> s1,poly_ops::detail::segment<index_t> s2);

#include "../include/poly_ops/poly_ops.hpp"
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
    return os << '{' << x.data << "}," << ',' << x.next;
}

std::ostream &operator<<(std::ostream &os,const detail::segment<index_t> &x) {
    return os << x.a << " - " << x.b;
}

namespace poly_ops {
    template<typename Coord> auto to_json_value(const point_t<Coord> &p) {
        return json::array_range(p);
    }

    namespace detail {
        template<typename Index> auto to_json_value(const segment_common<Index> &x) {
            return json::array_tuple(x.a,x.b);
        }

        template<typename Index> auto to_json_value(const event<Index> &x) {
            using namespace json;

            return obj(
                attr("ab") = x.ab,
                attr("deleted") = x.status == event_status_t::deleted,
                attr("type") = static_cast<int>(x.type) - static_cast<int>(event_type_t::vforward));
        }
    }
}

template<typename Sweep,typename Events> void emit_forward_backward(
    bool forward,
    poly_ops::detail::segment<index_t> seg,
    std::ranges::iterator_t<const Sweep> before,
    std::ranges::iterator_t<const Sweep> after,
    const Sweep &sweep,
    const Events &events)
{
    using namespace json;

    mc__->message(obj(
        attr("command") = "event",
        attr("type") = forward ? "forward" : "backward",
        attr("segment") = seg,
        attr("before") = before != sweep.end() ? just(*before) : std::nullopt,
        attr("after") = after != sweep.end() ? just(*after) : std::nullopt,
        attr("sweep") = array_range(sweep),
        attr("events") = array_range(events)));
}

void report_hits(index_t index,const std::pmr::vector<index_t> &hits,int balance) {
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

/*void dump_original_points(
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
}*/

class tokenizer {
    std::u8string_view input;

public:
    tokenizer(std::u8string_view input) : input(input) {}

    std::u8string_view operator()() {
        for(;;) {
            if(input.empty()) return input;
            char8_t c = input.front();
            if(!(c == ' ' || c == '\t' || c == '\n' || c == '\r')) break;
            input.remove_prefix(1);
        }
        for(size_t s=1; s<input.size(); ++s) {
            char8_t c = input[s];
            if(c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                std::u8string_view r{input.data(),s};
                input.remove_prefix(s);
                return r;
            }
        }
        auto r = input;
        input.remove_prefix(input.size());
        return r;
    }
};

template<typename Sweep> void delegate_drawing(
    const std::pmr::vector<detail::loop_point<index_t,coord_t>> &lpoints,
    const Sweep &sweep,
    const detail::event<index_t> &e)
{
    if(!graphical_debug) return;

    std::vector<draw_point> draw_lines;

    draw_lines.resize(lpoints.size());
    for(size_t i=0; i<lpoints.size(); ++i) {
        draw_point::state state = draw_point::NORMAL;
        if(sweep.count(detail::cached_segment<index_t,coord_t>(index_t(i),lpoints[i].next,lpoints))
                || sweep.count(detail::cached_segment<index_t,coord_t>(lpoints[i].next,index_t(i),lpoints))) {
            state = draw_point::SWEEP;
        } else if(lpoints[i].line_bal != detail::UNDEF_LINE_BAL) {
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

    if(!increment_amount) {
        for(;;) {
            tokenizer tok(mc__->get_text());
            std::u8string_view msg = tok();
            if(msg == u8"continue"sv) {
                auto inc_str = tok();
                if(inc_str.empty()) increment_amount = 1;
                else {
                    const char *start = reinterpret_cast<const char*>(inc_str.data());
                    const char *end = start+inc_str.size();
                    auto r = std::from_chars(start,end,increment_amount);
                    if(r.ptr != end || !tok().empty())
                        throw std::runtime_error("invalid command arguments");
                }
                if(increment_amount) break;
            } else throw std::runtime_error("unexpected command received from client");
        }
    }
    --increment_amount;
}

void delegate_drawing_trimmed(
    const std::pmr::vector<detail::loop_point<index_t,coord_t>> &lpoints)
{
    if(!graphical_debug) return;

    std::vector<draw_point> draw_lines;

    draw_lines.resize(lpoints.size());
    for(size_t i=0; i<lpoints.size(); ++i) {
        draw_point::state state = draw_point::NORMAL;
        if(lpoints[i].line_bal < 0) state = draw_point::INVERTED;
        else if(lpoints[i].line_bal > 0) state = draw_point::NESTED;
        /*else {
            for(index_t orig : original_sets[lpoints[i].original_set]) {
                const coord_t *op = input_coords[orig];
                draw_lines.emplace_back(point_t<coord_t>{op[0],op[1]},i,draw_point::TO_ORIGINAL);
            }
        }*/

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
        } else throw std::runtime_error("unexpected command received from client");
    }
}
