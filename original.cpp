#include <unordered_set>
#include <cstring>
#include <bit>
#include <random>
#include <iostream>
#include <fstream>
#include <string_view>
#include <ranges>
#include <stdexcept>

#include "poly_ops.hpp"
#include "server.hpp"
#include "stream_output.hpp"

typedef int32_t coord_t;
typedef uint16_t index_t;
bool step_by_step = false;
thread_local message_canvas *mc__;

using namespace poly_ops;
using namespace std::literals::string_view_literals;

struct assertion_failure : std::logic_error {
    using logic_error::logic_error;
};

void emit_line_before_after(const char *pre,auto itr,auto end) {
    auto out = mc__->console_line_stream();
    out << pre;
    if(itr == end) out << "none";
    else out << *itr;
}

struct draw_point {
    point_t<coord_t> data;
    index_t next;
    enum state {NORMAL=0,SWEEP} type;
};

auto to_json_value(const draw_point &p) {
    using namespace json;

    return obj(
        attr("data") = array_range(p.data),
        attr("next") = p.next,
        attr("state") = static_cast<int>(p.type));
}

std::ostream &operator<<(std::ostream &os,const detail::segment<index_t> &x) {
    return os << x.a << " - " << x.b;
}

std::ostream &operator<<(std::ostream &os,const point_t<coord_t> &x) {
    return os << x[0] << ',' << x[1];
}

void dump_sweep(
    const std::pmr::vector<detail::loop_point<index_t,coord_t>> &lpoints,
    const auto &sweep)
{
    auto out = mc__->console_line_stream();
    out << "sweep items:\n";
    for(detail::segment<index_t> s : sweep)
        out << "  " << s << " : " << lpoints[s.a].data << " - " << lpoints[s.b].data << '\n';
}

template<typename Sweep> void delegate_drawing(
    const std::pmr::vector<detail::loop_point<index_t,coord_t>> &lpoints,
    const Sweep &sweep,
    const detail::event<index_t> &e)
{
    std::vector<draw_point> draw_lines;

    draw_lines.resize(lpoints.size());
    for(size_t i=0; i<lpoints.size(); ++i) {
        draw_lines[i] = {lpoints[i].data,lpoints[i].next,draw_point::NORMAL};
    }
    for(auto &s : sweep) {
        /* the "lpoints[s.a].next != 0" check is needed for the first line,
        where both points are "main" points */
        draw_lines[s.a_is_main(lpoints) && lpoints[s.a].next != 0 ? s.a : s.b].type = draw_point::SWEEP;
    }

    mc__->message(json::obj(
        json::attr("command") = "draw",
        json::attr("points") = json::array_range(draw_lines),
        json::attr("currentPoint") = json::array_range(lpoints[e.ab.a].data)));

    for(;;) {
        std::u8string_view msg = mc__->get_text();
        if(msg == u8"continue"sv) {
            break;
        } else if(msg == u8"dump_sweep"sv) {
            dump_sweep(lpoints,sweep);
        } else throw std::runtime_error("unexpected command received from client");
    }
}

template<> struct std::hash<point_t<coord_t>> {
    std::size_t operator()(const point_t<coord_t> &p) const noexcept {
        if constexpr(sizeof(std::size_t) == 8) {
            static_assert(sizeof(p) == 8);
            std::size_t r;
            std::memcpy(&r,&p,8);
            return r;
        } else {
            return static_cast<std::size_t>(uint32_t(p[0]) ^ std::rotl(uint32_t(p[1]),1));
        }
    }
};

point_t<coord_t> unique_point(auto &rand_gen,std::unordered_set<point_t<coord_t>> &seen,std::uniform_int_distribution<coord_t> &dist,bool unique) {
    for(;;) {
        point_t<coord_t> c{dist(rand_gen),dist(rand_gen)};
        if(!unique) return c;
        auto [itr,inserted] = seen.insert(c);
        if(inserted) return c;
    }
}

void random_lines(auto &rand_gen,std::pmr::vector<detail::loop_point<index_t,coord_t>> &lpoints,index_t size,bool unique) {
    const int c_range = 1000;

    /* All points must be unique, so the number of possible point values should
    be much greater than the number of points we request */
    assert(size <= c_range);

    std::uniform_int_distribution<coord_t> dist(0,c_range);
    lpoints.resize(size*2);
    std::unordered_set<point_t<coord_t>> seen;
    for(index_t i=0; i<size; ++i) {
        lpoints[i].data = unique_point(rand_gen,seen,dist,unique);
        lpoints[i].next = size+i;
        lpoints[i].original_set = 0;
        lpoints[size+i].data = unique_point(rand_gen,seen,dist,unique);
        lpoints[size+i].next = 0;
        lpoints[size+i].original_set = 0;
    }
}

struct lpoint_renumber {
    std::pmr::vector<detail::loop_point<index_t,coord_t>> &lpoints;
    std::vector<index_t> old_indices;

    lpoint_renumber(std::pmr::vector<detail::loop_point<index_t,coord_t>> &lpoints) : lpoints(lpoints) {}

    void set(index_t i) {
        if(lpoints[i].line_bal == -1) {
            lpoints[i].line_bal = static_cast<int>(old_indices.size());
            old_indices.push_back(i);
        }
    }

    void set(detail::segment<index_t> s) {
        set(s.a);
        set(s.b);
    }

    index_t operator()(index_t i) const {
        assert(lpoints[i].line_bal != -1);
        return static_cast<index_t>(lpoints[i].line_bal);
    }

    detail::segment<index_t> operator()(detail::segment<index_t> s) const {
        return {(*this)(s.a),(*this)(s.b)};
    }
};

std::pmr::vector<index_t> self_intersection_orig(
    std::pmr::vector<detail::loop_point<index_t,coord_t>> &lpoints,
    detail::original_sets_t<index_t> &original_sets)
{
    using namespace poly_ops::detail;

    std::pmr::vector<event<index_t>> lines;
    std::pmr::vector<index_t> intrs;

    lines.reserve(lpoints.size() + 10);

    for(index_t i=0; i<lpoints.size()/2; ++i) {
        index_t j1 = i;
        index_t j2 = lpoints[i].next;

        if(lpoints[j1].data[0] > lpoints[j2].data[0]) std::swap(j1,j2);
        lines.emplace_back(segment<index_t>{j1,j2},event_type_t::forward);
        lines.emplace_back(segment<index_t>{j2,j1},event_type_t::backward);
    }

    events_t<index_t,coord_t> events(event_cmp<index_t,coord_t>{lpoints},std::move(lines));
    sweep_t<index_t,coord_t> sweep(sweep_cmp<index_t,coord_t>{lpoints});

    coord_t last_x = std::numeric_limits<coord_t>::lowest();
    std::pmr::vector<segment<index_t>> sweep_removed;

    while(!events.empty()) {
        event<index_t> e = events.top();
        events.pop();

        if(e.ab.a_x(lpoints) > last_x) {
            last_x = e.ab.a_x(lpoints);
            sweep_removed.clear();
        }

        /*if(!std::ranges::all_of(sweep,[=,&lpoints](segment<index_t> s) {
            return last_x >= s.a_x(lpoints) && last_x <= s.b_x(lpoints);
        })) throw assertion_failure{};*/

        switch(e.type) {
        case event_type_t::forward:
            {
                auto [itr,inserted] = sweep.insert(e.ab);
                if(!inserted) {
                    if(!step_by_step && !std::ranges::any_of(sweep,[new_s=e.ab](auto s){ return s == new_s; })) {
                        for(auto &p : lpoints) p.line_bal = -1;
                        lpoint_renumber renumber(lpoints);

                        for(auto s : sweep) renumber.set(s);
                        renumber.set(e.ab);

                        std::cout << "void test_sweep_cmp() {";

                        std::cout << "\n    std::pmr::vector<detail::loop_point<index_t,coord_t>> lpoints{\n        ";
                        bool started = false;
                        for(index_t i : renumber.old_indices) {
                            if(started) std::cout << ",\n        ";
                            started = true;
                            int next = lpoints[lpoints[i].next].line_bal;
                            std::cout << "detail::loop_point<index_t,coord_t>("
                                << pp(lpoints[i].data) << ",0,"
                                << (static_cast<size_t>(next) < renumber.old_indices.size() ? static_cast<index_t>(next) : index_t(0))
                                << ")";
                        }
                        std::cout << "};";

                        std::cout << "\n    sweep_t<index_t,coord_t> sweep({\n        "
                            << delimited(
                                std::views::transform(sweep,[&](segment<index_t> s) {
                                    return pp(segment<index_t>(renumber(s.a),renumber(s.b)));
                                }),
                                ",\n        ")
                            << "},detail::sweep_cmp<index_t,coord_t>{lpoints});";

                        std::cout << "\n    BOOST_CHECK(!sweep.count(" << pp(renumber(e.ab)) << "));";
                        std::cout << "\n}" << std::endl;
                    }
                    throw assertion_failure{"segment appears to have been inserted into sweep twice"};
                }
                auto before = line_before(sweep,itr,e.ab.a,lpoints.data());

                if(step_by_step) {
                    mc__->console_line_stream() << "event: FORWARD at "
                        << e.ab << "  x: " << e.ab.a_x(lpoints);
                    delegate_drawing(lpoints,sweep,e);
                }

                if(step_by_step)
                    emit_line_before_after("line before: ",before,sweep.end());

                if(before != sweep.end() && check_intersection(events,intrs,original_sets,sweep,lpoints,e.ab,*before)) continue;
                ++itr;
                itr = line_at_or_after(sweep,itr,e.ab.a,lpoints.data());

                if(step_by_step)
                    emit_line_before_after("line after: ",itr,sweep.end());

                if(itr != sweep.end()) check_intersection(events,intrs,original_sets,sweep,lpoints,e.ab,*itr);
            }
            break;
        case event_type_t::backward:
            {
                if(step_by_step) {
                    mc__->console_line_stream() << "event: BACKWARD at "
                        << e.ab << "  x: " << e.ab.a_x(lpoints);
                    delegate_drawing(lpoints,sweep,e);
                }

                auto itr = sweep.find(e.line_ba());

                /* if it's not in here, the line was split and no longer exists
                 */
                if(itr != sweep.end()) {
                    sweep_removed.push_back(*itr);
                    itr = sweep.erase(itr);

                    if(step_by_step)
                        emit_line_before_after("line after: ",itr,sweep.end());

                    if(itr != sweep.end()) {
                        auto before = line_before(sweep,itr,itr->a,lpoints.data());

                        if(step_by_step)
                            emit_line_before_after("line before: ",before,sweep.end());

                        if(before != sweep.end()) {
                            /*itr = line_at_or_after(sweep,itr,e.ab.b,lpoints.data());
                            if(itr != sweep.end())*/ check_intersection(events,intrs,original_sets,sweep,lpoints,*before,*itr);
                        }
                    }

                    if(intersects_any(e.line_ba(),sweep,lpoints.data()))
                        throw assertion_failure{"missed intersection"};
                } else {
                    for(auto s : sweep) {
                        if(s == e.line_ba()) {
                            if(step_by_step) dump_sweep(lpoints,sweep);
                            throw assertion_failure{"sweep sorting failure"};
                        }
                    }
                }
            }
            break;
        case event_type_t::calc_balance:
            {
                line_balance<index_t,coord_t> lb{lpoints.data(),e.ab.a,e.ab.b};
                for(const segment<index_t> &s : sweep) lb.check(s);
                for(const segment<index_t> &s : sweep_removed) lb.check(s);
                lpoints[e.ab.a].line_bal = std::get<0>(lb.result());
                lpoints[e.ab.b].line_bal = std::get<1>(lb.result());

                break;
            }
        }
    }

    std::ranges::sort(intrs);
    intrs.resize(std::unique(intrs.begin(),intrs.end()) - intrs.begin());
    return intrs;
}

void do_one(auto &rand_gen,std::pmr::vector<detail::loop_point<index_t,coord_t>> &lpoints) {
    random_lines(rand_gen,lpoints,5,false);
    detail::original_sets_t<index_t> original_sets(std::pmr::get_default_resource());
    original_sets.emplace_back();
    self_intersection_orig(lpoints,original_sets);
}

void do_one(auto &rand_gen) {
    std::pmr::vector<detail::loop_point<index_t,coord_t>> lpoints;
    do_one(rand_gen,lpoints);
}

int run_my_message_server(const char *htmlfile,const std::mt19937 &rand_gen,int i) {
    step_by_step = true;
    try {
        run_message_server(htmlfile,[&,i](message_canvas &mc) {
            mc__ = &mc;
            /* the RNG state is copied so each invocation of this callback
            is the same */
            auto rand_gen_ = rand_gen;
            mc.console_line_stream() << "failure on iteration " << i;
            try {
                do_one(rand_gen_);
            } catch(const std::exception &e) {
                mc.console_line(e.what());
            }
        });
    } catch(const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}

int main(int argc,char **argv) {
    if(argc != 2 && argc != 3) {
        std::cerr << "Usage: inset_test FILENAME [RANDSTATEFILE]" << std::endl;
        return 1;
    }

    std::mt19937 rand_gen;
    int i=0;
    if(argc == 2) {
        // try a bunch, until one fails, then redo it interactively
        try {
            std::pmr::vector<detail::loop_point<index_t,coord_t>> lpoints;
            for(; i<1000000; ++i) {
                std::mt19937 r_state = rand_gen;
                try {
                    do_one(rand_gen,lpoints);
                } catch(...) {
                    rand_gen = r_state;
                    throw;
                }
            }
        } catch(const assertion_failure&) {
            { std::ofstream("last_rand_state") << i << '\n' << rand_gen; }
            return run_my_message_server(argv[1],rand_gen,i);
        }
        std::cout << "all succeeded\n";
    } else {
        { std::ifstream(argv[2]) >> i >> rand_gen; }
        return run_my_message_server(argv[1],rand_gen,i);
    }
    return 0;
}
