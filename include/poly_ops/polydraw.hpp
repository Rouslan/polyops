/* A very basic polygon rasterizer */

#ifndef POLY_OPS_POLYDRAW_HPP
#define POLY_OPS_POLYDRAW_HPP

#include <vector>
#include <span>
#include <bit>
#include <cassert>

#include "base.hpp"
#include "sweep_set.hpp"
#include "large_ints.hpp"


#ifndef POLY_OPS_DRAW_DEBUG_LOG
#define POLY_OPS_DRAW_DEBUG_LOG(...) (void)0
#endif

namespace poly_ops::draw {

enum class fill_rule_t {non_zero,even_odd,positive,negative};

namespace detail {

template<typename T,unsigned int SizeNum=1,unsigned int SizeDenom=SizeNum>
struct fraction {
    using numerator_t = large_ints::sized_int<sizeof(T)*SizeNum>;
    using denominator_t = large_ints::sized_int<sizeof(T)*SizeDenom>;

    numerator_t num;
    denominator_t denom;

    fraction() noexcept = default;
    fraction(const fraction&) noexcept = default;
    fraction(const numerator_t &_num,const denominator_t &_denom=1) noexcept : num{_num}, denom{_denom} {
        assert(denom != 0);
        if(large_ints::negative(denom)) {
            num = -num;
            denom = -denom;
        }
    }

    fraction &operator=(const fraction&) noexcept = default;

    template<typename Num> fraction &operator=(const Num &value) noexcept {
        num = value;
        denom = 1;
        return *this;
    }

    friend auto operator<=>(const fraction &a,const fraction &b) noexcept {
        assert(a.denom > 0 && b.denom > 0);
        return large_ints::mul(a.num,b.denom) <=> large_ints::mul(b.num,a.denom);
    }
    friend auto operator<=>(const fraction &a,T b) noexcept {
        assert(a.denom > 0);
        return a.num <=> large_ints::mul(b,a.denom);
    }
    friend auto operator==(const fraction &a,const fraction &b) noexcept {
        assert(a.denom > 0 && b.denom > 0);
        return large_ints::mul(a.num,b.denom) == large_ints::mul(b.num,a.denom);
    }
    friend auto operator==(const fraction &a,T b) noexcept {
        assert(a.denom > 0);
        return a.num <=> large_ints::mul(b,a.denom);
    }
};

template<typename T,unsigned int Size> struct whole_and_frac {
    T whole;
    fraction<T,Size> frac;

    whole_and_frac() noexcept = default;
    explicit whole_and_frac(const T &whole) noexcept : whole{whole}, frac{0} {}
    whole_and_frac(const T &whole,const fraction<T,Size> &frac) noexcept : whole{whole}, frac{frac} {}

    T round() const noexcept {
        return whole + ((frac.num + frac.denom - 1) >= frac.denom/2);
    }

    friend auto operator<=>(const whole_and_frac &a,const whole_and_frac &b) noexcept {
        auto r = a.whole <=> b.whole;
        if(r != 0) return r;
        return a.frac <=> b.frac;
    }
    friend auto operator<=>(const whole_and_frac &a,T b) noexcept {
        auto r = a.whole <=> b;
        if(r != 0) return r;
        return a.frac <=> 0;
    }
    friend auto operator==(const whole_and_frac &a,const whole_and_frac &b) noexcept {
        return a.whole == b.whole && a.frac == b.frac;
    }
    friend auto operator==(const whole_and_frac &a,T b) noexcept {
        return a.frac == 0 && a.whole == b;
    }
};


template<typename Coord> struct loop_point {
    poly_ops::point_t<Coord> data;
    std::size_t next;

    loop_point() = default;
    loop_point(const loop_point &b) = default;
    loop_point(poly_ops::point_t<Coord> data,std::size_t next)
        noexcept(std::is_nothrow_copy_constructible_v<poly_ops::point_t<Coord>>)
        : data{data}, next{next} {}

    friend void swap(loop_point &a,loop_point &b) noexcept(std::is_nothrow_swappable_v<poly_ops::point_t<Coord>>) {
        using std::swap;

        swap(a.data,b.data);
        swap(a.next,b.next);
    }
};

using segment = poly_ops::detail::segment<std::size_t>;
template<typename Coord> using cached_segment = poly_ops::detail::cached_segment<std::size_t,Coord>;

template<typename T> constexpr size_t compound_int_size
    = (sizeof(T) + sizeof(large_ints::full_uint) - 1) / sizeof(large_ints::full_uint);

template<typename Coord>
bool intersects(
    const cached_segment<Coord> &s1,
    const cached_segment<Coord> &s2,
    whole_and_frac<Coord,2> &intr_y)
{
    using namespace large_ints;

    const Coord x1 = s1.pa.x();
    const Coord y1 = s1.pa.y();
    const Coord x2 = s1.pb.x();
    const Coord y2 = s1.pb.y();
    const Coord x3 = s2.pa.x();
    const Coord y3 = s2.pa.y();
    const Coord x4 = s2.pb.x();
    const Coord y4 = s2.pb.y();

    if(s1.a == s2.a || s1.a == s2.b || s1.b == s2.a || s1.b == s2.b) return false;

    auto d = mul<Coord,Coord>(x1-x2,y3-y4) - mul<Coord,Coord>(y1-y2,x3-x4);
    if(d == 0) return false;

    auto t_i = mul<Coord,Coord>(x1-x3,y3-y4) - mul<Coord,Coord>(y1-y3,x3-x4);
    auto u_i = mul<Coord,Coord>(x1-x3,y1-y2) - mul<Coord,Coord>(y1-y3,x1-x2);

    if(d > 0) {
        if(t_i <= 0 || t_i >= d || u_i <= 0 || u_i >= d) return false;
    } else if(t_i >= 0 || t_i <= d || u_i >= 0 || u_i <= d) return false;

    auto qr = unmul<compound_int_size<Coord>>(mul(t_i,y2-y1) + mul(d,y1),d,modulo_t::euclid);
    intr_y.whole = static_cast<Coord>(qr.quot);
    intr_y.frac.num = static_cast<typename fraction<Coord,2>::numerator_t>(qr.rem);
    intr_y.frac.denom = abs(d);

    return true;
}

template<typename T> fraction<T,2,1> x_intercept(
    const poly_ops::point_t<T> &start,
    const poly_ops::point_t<T> &end,
    std::type_identity_t<T> y) noexcept
{
    using namespace large_ints;
    return {mul(start.x(),static_cast<T>(y - end.y())) + mul(end.x(),static_cast<T>(start.y() - y)),
        static_cast<T>(start.y() - end.y())};
}

template<typename Coord> struct sweep_cmp {
    Coord current_y;
    
    bool operator()(const cached_segment<Coord> &s1,const cached_segment<Coord> &s2) const {
        auto cmp = x_intercept(s1.pa,s1.pb,current_y) <=> x_intercept(s2.pa,s2.pb,current_y);
        if(cmp != 0) return cmp > 0;
        cmp = x_intercept(s1.pa,s1.pb,current_y+1) <=> x_intercept(s2.pa,s2.pb,current_y+1);
        if(cmp != 0) return cmp > 0;

        return s1.b == s2.b ? (s1.a > s2.a) : (s1.b > s2.b);
    }
};

template<typename Coord>
using sweep_t = poly_ops::detail::sweep_set<
    cached_segment<Coord>,
    std::size_t,
    sweep_cmp<Coord>,
    std::vector<poly_ops::detail::set_node<cached_segment<Coord>,std::size_t>>>;

template<typename Coord>
using sweep_node = poly_ops::detail::set_node<cached_segment<Coord>,std::size_t>;

/* This is only used for debugging */
template<typename Coord>
bool consistent_order(const sweep_t<Coord> &sweep) {
    auto &cmp = sweep.node_comp();
    for(auto itr_a = sweep.begin(); itr_a != sweep.end(); ++itr_a) {
        auto itr_b = sweep.begin();
        for(; itr_b != itr_a; ++itr_b) {
            if(cmp(*itr_a,*itr_b)) {
                POLY_OPS_ASSERT(false);
                return false;
            }
        }
        if(cmp(*itr_a,*itr_b++)) {
            POLY_OPS_ASSERT(false);
            return false;
        }
        for(; itr_b != sweep.end(); ++itr_b) {
            if(cmp(*itr_b,*itr_a)) {
                POLY_OPS_ASSERT(false);
                return false;
            }
        }
    }
    return true;
}

enum class event_type_t {backward,forward,swap};
struct swap_type {};
template<typename Coord> struct event {
    event_type_t type;

    /* if type == event_type_t::swap, ab.a must be less than or equal to ab.b */
    segment ab;

    union {
        std::size_t sweep_node;

        /* the y coordinate of the intersection point */
        whole_and_frac<Coord,2> intr_y;
    };
    

    event() = default;
    event(event_type_t type,segment ab,std::size_t sweep_node) noexcept
        : type{type}, ab{ab}, sweep_node{sweep_node} {}
    event(swap_type,segment ab,const whole_and_frac<Coord,2> &intr_y)
        : type{event_type_t::swap}, ab{ab}, intr_y{intr_y} {}
    event(const event &b) = default;

    event &operator=(const event&) = default;

    segment line_ba() const { return {ab.b,ab.a}; }

    friend bool operator==(const event &a,const event &b) {
        if(a.type != b.type || a.ab != b.ab) return false;
        return a.type != event_type_t::swap || a.intr_y == b.intr_y;
    }
};

template<typename Coord> struct event_cmp {
    const std::vector<loop_point<Coord>> &lpoints;

    bool operator()(const event<Coord> &l1,const event<Coord> &l2) const {
        std::weak_ordering ord{std::weak_ordering::equivalent};
        if(l1.type == event_type_t::swap) [[unlikely]] {
            if(l2.type == event_type_t::swap) [[unlikely]] {
                ord = l1.intr_y <=> l2.intr_y;
            } else {
                ord = l1.intr_y <=> l2.ab.a_y(lpoints);
            }
        } else if(l2.type == event_type_t::swap) [[unlikely]] {
            ord = l1.ab.a_y(lpoints) <=> l2.intr_y;
        } else [[likely]] {
            ord = l1.ab.a_y(lpoints) <=> l2.ab.a_y(lpoints);
        }
        if(ord != 0) return ord < 0;
        if(l1.type != l2.type) return l1.type < l2.type;
        if(l1.ab.a != l2.ab.a) return l1.ab.a < l2.ab.a;
        return l1.ab.b < l2.ab.b;
    }
};

template<typename Coord>
bool less_than_or_equal(const std::vector<loop_point<Coord>> &lpoints,const event<Coord> &a,const Coord b) {
    if(a.type == event_type_t::swap) [[unlikely]] {
        return a.intr_y <= b;
    }

    return a.ab.a_y(lpoints) <= b;
}

template<typename Coord> class events_t {
    using points_ref = const std::vector<loop_point<Coord>> &;
    using cmp = event_cmp<Coord>;

    struct to_insert {
        event<Coord> e;
        std::size_t i;
    };

    std::vector<event<Coord>> events;
    std::vector<to_insert> new_events;
    std::ptrdiff_t current_i;
    std::size_t last_size;

    void incorporate_new() {
        POLY_OPS_ASSERT(events.size());

        std::size_t new_count = events.size() - last_size;
        for(std::size_t i=new_count, j=events.size()-1; i>0; --i) {
            while(j > new_events[i-1].i) {
                events[j] = std::move(events[j-i]);
                --j;
            }
            events[j--] = std::move(new_events[i-1].e);
        }

        new_events.clear();
        last_size = events.size();
    }

public:
    events_t()
        : events(), new_events(), current_i(-1), last_size(0) {}

    void clear() {
        events.clear();
        new_events.clear();
        current_i = -1;
        last_size = 0;
    }

    bool at_end() const {
        return current_i >= static_cast<std::ptrdiff_t>(events.size());
    }

    void next(points_ref points) {
        POLY_OPS_ASSERT(current_i < static_cast<std::ptrdiff_t>(events.size()));

        if(events.size() > last_size) {
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
                        events.begin()+static_cast<std::ptrdiff_t>(last_size),
                        events[last_size+i],
                        cmp{points});
                    POLY_OPS_ASSERT(itr == (events.begin()+static_cast<std::ptrdiff_t>(last_size))
                        || events[last_size+i] != *itr
                        || itr->type == event_type_t::swap);

                    new_events.emplace_back(std::move(events[last_size+i]),static_cast<std::size_t>(itr - events.begin()));
                }
                events.resize(last_size + new_events.size());

                if(!new_events.empty()) {
                    std::ranges::sort(
                        new_events,
                        [&](const to_insert &a,const to_insert &b) {
                            if(a.i != b.i) return a.i < b.i;
                            return cmp{points}(a.e,b.e);
                        });
                    for(std::size_t i=0; i<new_events.size(); ++i) new_events[i].i += i;

                    POLY_OPS_ASSERT(static_cast<std::ptrdiff_t>(new_events[0].i) > current_i);

                    incorporate_new();
                }
            }
        }

        ++current_i;
    }

    void add_event(event_type_t t,std::size_t sa,std::size_t sb,std::size_t sweep_node) {
        events.emplace_back(t,segment{sa,sb},sweep_node);
    }

    void add_fb_events(std::size_t sa,std::size_t sb,std::size_t sweep_node) {
        add_event(event_type_t::forward,sa,sb,sweep_node);
        add_event(event_type_t::backward,sb,sa,sweep_node);
    }

    void add_swap_event(std::size_t sa,std::size_t sb,const detail::whole_and_frac<Coord,2> &intr_y) {
        events.emplace_back(swap_type{},segment{sa,sb},intr_y);
    }

    // this is only used for debugging
    bool check_last_event(points_ref points) const {
        return cmp{points}(current(),events.back());
    }

    event<Coord> &current() { return events[static_cast<std::size_t>(current_i)]; }
    const event<Coord> &current() const { return events[static_cast<std::size_t>(current_i)]; }
};

template<typename T> T clamp(T x,std::type_identity_t<T> lo,std::type_identity_t<T> hi) {
    POLY_OPS_ASSERT(lo <= hi);
    if(x < lo) x = lo;
    else if(x > hi) x = hi;
    return x;
}

template<typename T,unsigned int NumSize,unsigned int DenomSize> T to_coord(const fraction<T,NumSize,DenomSize> &x,std::type_identity_t<T> lo,std::type_identity_t<T> hi) {
    return clamp(static_cast<T>(large_ints::unmul<compound_int_size<T>>(x.num,x.denom).quot),lo,hi);
}

bool should_fill(fill_rule_t rule,long winding) {
    switch(rule) {
    case fill_rule_t::non_zero:
        return winding != 0;
    case fill_rule_t::even_odd:
        return winding % 2;
    case fill_rule_t::positive:
        return winding > 0;
    case fill_rule_t::negative:
        return winding < 0;
    }
    POLY_OPS_ASSERT(false);
    return false;
}

} // namespace detail


template<typename Coord> struct scan_line_sweep_state;

template<typename Coord> class rasterizer {
    friend scan_line_sweep_state<Coord>;

    std::vector<detail::loop_point<Coord>> lpoints;
    detail::events_t<Coord> events;

    /* The first value in this array is reserved for the red black tree in
    "sweep_set" */
    std::vector<detail::sweep_node<Coord>> sweep_nodes;

    bool ran;

    void check_intersection(std::size_t s1,std::size_t s2,Coord y);
    void check_intersection_after_swap(
        detail::sweep_t<Coord> sweep,
        std::size_t s1,
        std::size_t s2,
        const detail::whole_and_frac<Coord,2> &y);
    void add_fb_events(std::size_t sa,std::size_t sb);

    template<poly_ops::point_range<Coord> R> void _add_loop(R &&loop) {
        auto sink = add_loop();
        for(poly_ops::point_t<Coord> p : loop) sink(p);
    }

    void swap_and_check(detail::sweep_t<Coord> sweep,std::size_t s1,std::size_t s2,const detail::whole_and_frac<Coord,2> &y);

public:
    class point_sink {
        friend class rasterizer;

        rasterizer &n;
        poly_ops::point_t<Coord> prev;
        std::size_t first_i;
        bool started;

        point_sink(rasterizer &n) : n(n), started(false) {}
        point_sink(const point_sink&) = delete;
        point_sink(point_sink &&b) : n(b.n), prev(b.prev), first_i{b.first_i}, started(b.started) {
            b.started = false;
        }

    public:
        void operator()(const poly_ops::point_t<Coord> &p);
        ~point_sink();
    };

    rasterizer() : ran(false) {
        sweep_nodes.emplace_back();
    }

    rasterizer(rasterizer &&b) = default;

    template<poly_ops::point_range_or_range_range<Coord> R> void add_loops(R &&loops) {
        if constexpr(poly_ops::point_range_range<R,Coord>) {
            for(auto &&loop : loops) _add_loop(std::forward<decltype(loop)>(loop));
        } else {
            _add_loop(std::forward<R>(loops));
        }
    }

    point_sink add_loop() {
        if(ran) clear();
        return {*this};
    }

    scan_line_sweep_state<Coord> scan_lines(unsigned int width,unsigned int height);

    template<typename F> void draw(unsigned int width,unsigned int height,F &&fill_row,fill_rule_t fill_rule=fill_rule_t::non_zero);

    void clear() {
        lpoints.clear();
        ran = false;
    }
};

template<typename Coord>
void rasterizer<Coord>::point_sink::operator()(const poly_ops::point_t<Coord> &p) {
    if(started) [[likely]] {
        if(prev == n.lpoints.back().data) goto skip;
    } else {
        prev = p;
        first_i = n.lpoints.size();
        started = true;
    }

    /* Normally, points aren't added until this is called with the next point or
    the destructor is called, but duplicate points aren't added anyway and
    adding it on the first call means the "prev != n.lpoints.back().data" checks
    above and in the destructor are always safe. */
    n.lpoints.emplace_back(prev,n.lpoints.size()+1);

skip:
    prev = p;
}

template<typename Coord>
rasterizer<Coord>::point_sink::~point_sink() {
    if(started) [[likely]] {
        if(prev != n.lpoints.back().data && prev != n.lpoints[first_i].data) [[likely]] {
            n.lpoints.emplace_back(prev,0);
        }

        std::size_t new_points = n.lpoints.size() - static_cast<std::size_t>(first_i);
        if(new_points < 3) [[unlikely]] {
            while(new_points-- > 0) {
                n.lpoints.pop_back();
            }
        } else {
            n.lpoints.back().next = first_i;
        }
    }
}

template<typename Coord>
void rasterizer<Coord>::add_fb_events(std::size_t sa,std::size_t sb) {
    events.add_fb_events(sa,sb,sweep_nodes.size());
    sweep_nodes.emplace_back(detail::cached_segment<Coord>(sa,sb,lpoints));
}

template<typename Coord>
void rasterizer<Coord>::check_intersection(std::size_t s1,std::size_t s2,Coord y) {
    detail::whole_and_frac<Coord,2> intr_y;
    if(intersects(sweep_nodes[s1].value,sweep_nodes[s2].value,intr_y) && intr_y >= y) {
        POLY_OPS_DRAW_DEBUG_LOG("checking intersection of {} and {}: {}",sweep_nodes[s1].value,sweep_nodes[s2].value,intr_y);
        events.add_swap_event(s1,s2,intr_y);
        POLY_OPS_ASSERT(events.check_last_event(lpoints));
    } else {
        POLY_OPS_DRAW_DEBUG_LOG("checking intersection of {} and {}: none",sweep_nodes[s1].value,sweep_nodes[s2].value);
    }
}

/* When lines are coincident, intersecting lines will have to swap with all the
coincident lines */
template<typename Coord>
void rasterizer<Coord>::check_intersection_after_swap(
    detail::sweep_t<Coord> sweep,
    std::size_t s1,
    std::size_t s2,
    const detail::whole_and_frac<Coord,2> &y)
{
    detail::whole_and_frac<Coord,2> intr_y;
    if(intersects(sweep_nodes[s1].value,sweep_nodes[s2].value,intr_y) && intr_y >= y) {
        POLY_OPS_DRAW_DEBUG_LOG("checking intersection of {} and {}: {}",sweep_nodes[s1].value,sweep_nodes[s2].value,intr_y);
        if(intr_y == y) {
            POLY_OPS_DRAW_DEBUG_LOG(
                "IMMEDIATE SWAP {} and {}",
                sweep_nodes[s1].value,
                sweep_nodes[s2].value);
            
            swap_and_check(sweep,s1,s2,y);
        } else {
            events.add_swap_event(s1,s2,intr_y);
            POLY_OPS_ASSERT(events.check_last_event(lpoints));
        }
    } else {
        POLY_OPS_DRAW_DEBUG_LOG("checking intersection of {} and {}: none",sweep_nodes[s1].value,sweep_nodes[s2].value);
    }
}

template<typename Coord> void rasterizer<Coord>::swap_and_check(
    detail::sweep_t<Coord> sweep,
    std::size_t s1,
    std::size_t s2,
    const detail::whole_and_frac<Coord,2> &y)
{
    if(std::next(sweep.iterator_to(s1)) != sweep.iterator_to(s2)) return;

    sweep.erase(s2);
    sweep.insert_before(s1,s2);

    auto itr1 = sweep.iterator_to(s2);
    if(itr1 != sweep.begin()) check_intersection_after_swap(sweep,std::prev(itr1).index(),s2,y);
    itr1 = sweep.iterator_to(s1);
    auto itr2 = std::next(itr1);
    if(itr2 != sweep.end()) check_intersection_after_swap(sweep,s1,itr2.index(),y);
}

struct scan_line {
    unsigned int x1;
    unsigned int x2;
    unsigned int y;
    long winding;
};

template<typename Coord> struct scan_line_sweep_state {
    unsigned int width;
    unsigned int height;
    detail::sweep_t<Coord> sweep;
    Coord y;
    typename detail::sweep_t<Coord>::const_iterator s_itr;
    Coord x;
    long winding;
    bool suspended;

    scan_line_sweep_state(rasterizer<Coord> &rast,unsigned int width,unsigned int height,Coord y)
        : width{width}, height{height}, sweep{rast.sweep_nodes}, y{y}, s_itr{sweep.end()}, suspended{false} {}

public:
    bool operator()(rasterizer<Coord> &rast,scan_line &sc) {
        using namespace detail;

        if(suspended) {
            suspended = false;
            goto continue_point;
        }

        for(; !rast.events.at_end() && std::cmp_less(y,height); ++y) {
            sweep.key_comp().current_y = y;
            for(; !rast.events.at_end()
                    && less_than_or_equal(rast.lpoints,rast.events.current(),y)
                    && rast.events.current().type == event_type_t::swap; rast.events.next(rast.lpoints))
            {
                auto &e = rast.events.current();

                POLY_OPS_DRAW_DEBUG_LOG(
                    "SWAP {} and {} at {}",
                    rast.sweep_nodes[e.ab.a].value,
                    rast.sweep_nodes[e.ab.b].value,
                    e.intr_y);

                rast.swap_and_check(sweep,e.ab.a,e.ab.b,e.intr_y);

                POLY_OPS_DRAW_DEBUG_LOG("sweep: {}",sweep);
            }
            for(; !rast.events.at_end() && less_than_or_equal(rast.lpoints,rast.events.current(),y) && rast.events.current().type == event_type_t::backward; rast.events.next(rast.lpoints)) {
                auto &e = rast.events.current();

                POLY_OPS_DRAW_DEBUG_LOG(
                    "BACKWARD {} at {}",
                    e.ab,
                    e.ab.a_y(rast.lpoints));

                auto itr = sweep.erase(e.sweep_node);

                if(itr != sweep.end() && itr != sweep.begin()) {
                    rast.check_intersection(std::prev(itr).index(),itr.index(),e.ab.a_y(rast.lpoints));
                }

                POLY_OPS_DRAW_DEBUG_LOG("sweep: {}",sweep);
            }
            for(; !rast.events.at_end() && less_than_or_equal(rast.lpoints,rast.events.current(),y) && rast.events.current().type == event_type_t::forward; rast.events.next(rast.lpoints)) {
                auto &e = rast.events.current();

                POLY_OPS_DRAW_DEBUG_LOG(
                    "FORWARD {} at {}",
                    e.ab,
                    e.ab.a_y(rast.lpoints));

                auto [itr,inserted] = sweep.insert(e.sweep_node);
                POLY_OPS_ASSERT(inserted);

                if(itr != sweep.begin()) rast.check_intersection(std::prev(itr).index(),e.sweep_node,e.ab.a_y(rast.lpoints));
                ++itr;
                if(itr != sweep.end()) rast.check_intersection(e.sweep_node,itr.index(),e.ab.a_y(rast.lpoints));

                POLY_OPS_DRAW_DEBUG_LOG("sweep: {}",sweep);
            }

            POLY_OPS_ASSERT(rast.events.at_end() || !less_than_or_equal(rast.lpoints,rast.events.current(),y));
            POLY_OPS_ASSERT(consistent_order(sweep));

            if(y >= 0) {
                winding = 0;
                s_itr = sweep.begin();

                for(; s_itr != sweep.end(); ++s_itr) {
                    {
                        long prev_winding = winding;
                        winding += s_itr->value.a_is_main(rast.lpoints) ? 1 : -1;

                        Coord x_start = x;
                        x = to_coord(x_intercept(s_itr->value.pa,s_itr->value.pb,y),0,static_cast<Coord>(width-1));
   
                        if(prev_winding) {
                            sc = {static_cast<unsigned int>(x),static_cast<unsigned int>(x_start),static_cast<unsigned int>(y),prev_winding};
                            suspended = true;
                            return true;
                        }
                    }
                  continue_point: ;
                }
            }
        }
        return false;
    }
};

template<typename Coord>
scan_line_sweep_state<Coord> rasterizer<Coord>::scan_lines(unsigned int width,unsigned int height) {
    using namespace detail;

    events.clear();
    sweep_nodes.resize(1);

    ran = true;

    for(std::size_t i=0; i<lpoints.size(); ++i) {
        std::size_t j1 = i;
        std::size_t j2 = lpoints[i].next;

        if(lpoints[j1].data.y() != lpoints[j2].data.y()) {
            if(lpoints[j1].data.y() > lpoints[j2].data.y()) std::swap(j1,j2);
            add_fb_events(j1,j2);
        }
    }

    events.next(lpoints);

    return {*this,width,height,events.at_end() ? static_cast<Coord>(0) : events.current().ab.a_y(lpoints)};
}

template<typename Coord> template<typename F> 
void rasterizer<Coord>::draw(unsigned int width,unsigned int height,F &&fill_row,fill_rule_t fill_rule) {
    auto state = scan_lines(width,height);
    scan_line sc;
    while(state(*this,sc)) {
        if(detail::should_fill(fill_rule,sc.winding)) fill_row(sc.x1,sc.x2,sc.y);
    }
}

} // namespace poly_ops::draw

#endif
