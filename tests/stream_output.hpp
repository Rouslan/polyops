/* The methods here assume the Coord template parameter from objects defined in
poly_ops.hpp is always "coord_t" and Index is always "index_t".
*/

#ifndef stream_output_hpp
#define stream_output_hpp

#include <tuple>
#include <istream>
#include <ostream>
#include <type_traits>
#include <map>
#include <string_view>
#include <ranges>
#include <vector>

#if __has_include (<format>)
#include <format>
#define POLY_OPS_HAS_STD_FORMAT
#endif

#include "../include/poly_ops/base.hpp"

template<typename T> inline std::ostream &operator<<(std::ostream &os,const poly_ops::point_t<T> &x) {
    return os << '{' << x.x() << ',' << x.y() << '}';
}

/*
template<typename T> struct _pp_type {};
template<typename T> inline _pp_type<T> pp_type;

template<typename T> std::ostream &operator<<(std::ostream &os,_pp_type<T>) {
    return _pp_type<T>::name;
}

#define DEFINE_PP_TYPE(T) \
template<> struct _pp_type<T> { static constexpr const char *name = #T; }

DEFINE_PP_TYPE(bool);
DEFINE_PP_TYPE(char);
DEFINE_PP_TYPE(unsigned char);
DEFINE_PP_TYPE(signed char);
DEFINE_PP_TYPE(int);
DEFINE_PP_TYPE(unsigned int);
DEFINE_PP_TYPE(short);
DEFINE_PP_TYPE(unsigned short);
DEFINE_PP_TYPE(long);
DEFINE_PP_TYPE(unsigned long);
DEFINE_PP_TYPE(long long);
DEFINE_PP_TYPE(unsigned long long);
*/

template<typename T> struct _pp {
    T value;
    unsigned int indent;
};

struct indent_t {
    unsigned int amount;
};
inline indent_t operator+(indent_t a,unsigned int b) {
    return {a.amount+b};
}

inline std::ostream &operator<<(std::ostream &os,indent_t indent) {
    os << '\n';
    for(unsigned int i=0; i<indent.amount; ++i) os << "  ";
    return os;
}

template<typename T> _pp<T> pp(T &&x,unsigned int indent) { return _pp<T>{std::forward<T>(x),indent}; }
template<typename T> _pp<T> pp(T &&x,indent_t indent) { return _pp<T>{std::forward<T>(x),indent.amount}; }

template<typename T> struct pp_printer {
    void operator()(std::ostream &os,indent_t,const T &x) const {
        os << x;
    }
};

template<> struct pp_printer<bool> {
    void operator()(std::ostream &os,indent_t,bool x) const {
        os << (x ? "true" : "false");
    }
};

template<typename Coord> struct pp_printer<poly_ops::point_t<Coord>> {
    void operator()(std::ostream &os,indent_t,const poly_ops::point_t<Coord> &x) const {
        os << "point_t<coord_t>(" << x[0] << ',' << x[1] << ')';
    }
};

template<typename Index> struct pp_printer<poly_ops::detail::segment<Index>> {
    void operator()(std::ostream &os,indent_t,const poly_ops::detail::segment<Index> &x) const {
        os << "detail::segment<index_t>(" << x.a << ',' << x.b << ')';
    }
};

template<typename T1,typename... T> struct arg_delimited {
    const T1 &value;
    std::tuple<const T&...> values;

    arg_delimited(const T1 &value,const T&... values) : value{value}, values{values...} {}

    template<typename U> static arg_delimited from_tuple(const U &x) {
        return std::make_from_tuple<arg_delimited>(x);
    }
};

template<typename T> std::ostream &operator<<(std::ostream &os,arg_delimited<T> x) {
    return os << pp(x.value,0);
}

template<typename T1,typename... T> std::ostream &operator<<(std::ostream &os,arg_delimited<T1,T...> x) {
    return os << pp(x.value,0) << ',' << arg_delimited<T...>::from_tuple(x.values);
}

template<typename... T> struct pp_printer<std::tuple<T...>> {
    void operator()(std::ostream &os,indent_t,const std::tuple<T...> &x) const {
        os << "std::tuple(" << arg_delimited<T...>::from_tuple(x) << ')';
    }
};

template<typename T> struct delimited_t {
    T items;
    std::string_view delimiter;

    explicit constexpr delimited_t(T &&items,std::string_view delimiter=", ")
        : items(std::forward<T>(items)), delimiter(delimiter) {}
};
template<typename R> constexpr delimited_t<R> delimited(R &&items) {
    return delimited_t<R>{std::forward<R>(items)};
}
template<typename T> std::ostream &operator<<(std::ostream &os,delimited_t<T> &&d) {
    bool started = false;
    for(auto &&item : d.items) {
        if(started) os << d.delimiter;
        os << item;
        started = true;
    }
    return os;
}

template<typename Key,typename Value,typename Compare,typename Alloc> struct pp_printer<std::map<Key,Value,Compare,Alloc>> {
    void operator()(std::ostream &os,indent_t indent,const std::map<Key,Value,Compare,Alloc> &x) const {
        os << "std::map{";
        bool started = false;
        indent = indent + 1;
        for(const auto& [key,value] : x) {
            if(started) os << ',';
            started = true;
            if(x.size() > 1) os << indent;
            os << '{' << pp(key,indent) << ',' << pp(value,indent) << '}';
        }
        os << '}';
    }
};

template<typename T> struct pp_printer<delimited_t<T>> {
    void operator()(std::ostream &os,indent_t indent,delimited_t<T> &d) const {
        bool started = false;
        indent = indent + 1;
        for(auto &&item : d.items) {
            if(started) os << ',';
            started = true;
            os << indent;
            os << pp(item,indent);
        }
    }
};

template<typename T,typename Alloc> struct pp_printer<std::vector<T,Alloc>> {
    void operator()(std::ostream &os,indent_t indent,const std::vector<T,Alloc> &x) const {
        os << "std::vector{";
        bool started = false;
        indent = indent + 1;
        for(const auto &item : x) {
            if(started) os << ',';
            started = true;
            if(x.size() > 1) os << indent;
            os << pp(item,indent);
        }
        os << '}';
    }
};

template<typename T> std::ostream &operator<<(std::ostream &os,const _pp<T> &x) {
    pp_printer<std::remove_cvref_t<T>>{}(os,indent_t{x.indent},x.value);
    return os;
}

template<typename Coord,typename R> void write_loops(std::ostream &os,R &&loops) {
    bool first = true;
    for(auto &&loop : loops) {
        if(!first) os << "===\n";
        first = false;
        for(auto &&raw_p : loop) {
            poly_ops::point_t<Coord> p(raw_p);
            os << p[0] << ' ' << p[1] << '\n';
        }
    }
}

void odd_coord_count() {
    throw std::runtime_error("input has odd number of coordinates");
}

struct stream_exceptions_saver {
    std::ios &ios;
    std::ios_base::iostate state;

    stream_exceptions_saver(std::ios &ios) : ios(ios), state(ios.exceptions()) {}
    ~stream_exceptions_saver() {
        ios.exceptions(state);
    }
};

template<typename Coord> void read_loops(std::istream &is,std::vector<std::vector<poly_ops::point_t<Coord>>> &loops) {
    loops.emplace_back();
    poly_ops::point_t<Coord> p;
    unsigned int c_count = 0;

    stream_exceptions_saver ses(is);

    is.exceptions(is.badbit);
    for(;;) {
        is >> std::ws;
        if(is.eof()) {
            if(c_count) odd_coord_count();
            break;
        }
        auto c = is.peek();
        if(c == '=') {
            if(c_count) odd_coord_count();
            is.get();
            if(!loops.back().empty()) loops.emplace_back();
        } else {
            is >> p[c_count++];
            if(is.fail()) {
                throw std::runtime_error("invalid input format");
            }
            if(c_count > 1) {
                c_count = 0;
                loops.back().push_back(p);
            }
        }
    }
}

#ifdef POLY_OPS_HAS_STD_FORMAT

#include "../include/poly_ops/sweep_set.hpp"

template<typename Index> struct segment_common_formatter {
    template<typename ParseContext>
    constexpr ParseContext::iterator parse(ParseContext &ctx) const {
        return ctx.begin();
    }
    template<typename FmtContext>
    FmtContext::iterator format(const poly_ops::detail::segment_common<Index> &s,FmtContext &ctx) const {
        return std::format_to(ctx.out(),"{} - {}",s.a,s.b);
    }
};

template<typename Index>
struct std::formatter<poly_ops::detail::segment<Index>,char> : segment_common_formatter<Index> {};
template<typename Index,typename Coord>
struct std::formatter<poly_ops::detail::cached_segment<Index,Coord>,char> : segment_common_formatter<Index> {};

/* a poor man's substitute for C++23 range formatting */
template<typename R>
struct std::formatter<delimited_t<R>,char> {
    template<typename ParseContext>
    constexpr ParseContext::iterator parse(ParseContext &ctx) const {
        return ctx.begin();
    }
    template<typename FmtContext>
    FmtContext::iterator format(delimited_t<R> &d,FmtContext &ctx) const {
        auto out = ctx.out();
        bool started = false;
        for(auto &&item : d.items) {
            if(started) {
                out = std::format_to(out,"{}",d.delimiter);
            }
            out = std::format_to(out,"{}",item);
            started = true;
        }
        return out;
    }
};

template<typename T,typename Index,typename Compare,typename Vec>
struct std::formatter<poly_ops::detail::sweep_set<T,Index,Compare,Vec>,char> {
    template<typename ParseContext>
    constexpr ParseContext::iterator parse(ParseContext &ctx) const {
        return ctx.begin();
    }
    template<typename FmtContext>
    FmtContext::iterator format(const poly_ops::detail::sweep_set<T,Index,Compare,Vec> &s,FmtContext &ctx) const {
        return std::format_to(
            ctx.out(),
            "{{{}}}",
            delimited(s | std::views::transform([](const auto &node) -> decltype(auto) { return node.value; })));
    }
};

template<typename T>
struct std::formatter<poly_ops::point_t<T>,char> {
    template<typename ParseContext>
    constexpr ParseContext::iterator parse(ParseContext &ctx) const {
        return ctx.begin();
    }
    template<typename FmtContext>
    FmtContext::iterator format(const poly_ops::point_t<T> &p,FmtContext &ctx) const {
        return std::format_to(ctx.out(),"{{{},{}}}",p.x(),p.y());
    }
};

namespace detail {
template<unsigned int N>
int format_large_int(poly_ops::large_ints::compound_uint<N> x,char *buffer) {
    int digits = 0;
    while(x != 0) {
        auto dm = poly_ops::large_ints::divmod(x,10u);
        x = dm.quot;
        buffer[digits++] = "0123456789"[static_cast<unsigned int>(dm.rem)];
    }
    if(digits == 0) {
        *buffer = '0';
        digits = 1;
    }
    return digits;
}
}

template<unsigned int N,bool Signed>
struct std::formatter<poly_ops::large_ints::compound_xint<N,Signed>,char> {
    template<typename ParseContext>
    constexpr ParseContext::iterator parse(ParseContext &ctx) const {
        return ctx.begin();
    }
    template<typename FmtContext>
    FmtContext::iterator format(const poly_ops::large_ints::compound_xint<N,Signed> &x,FmtContext &ctx) const {
        char buffer[(N*sizeof(poly_ops::large_ints::full_uint)*8 + 2) / 3];
        int digits;
        auto out = ctx.out();

        if constexpr(Signed) {
            if(x.negative()) {
                *out++ = '-';
                digits = detail::format_large_int(static_cast<poly_ops::large_ints::compound_uint<N>>(-x),buffer);
            } else {
                digits = detail::format_large_int(static_cast<poly_ops::large_ints::compound_uint<N>>(x),buffer);
            }
        } else {
            digits = detail::format_large_int(x,buffer);
        }

        while(digits--) {
            *out++ = buffer[digits];
        }

        return out;
    }
};

#ifdef POLY_OPS_POLYDRAW_HPP
template<typename T,unsigned int SizeNum,unsigned int SizeDenom>
struct std::formatter<poly_ops::draw::detail::fraction<T,SizeNum,SizeDenom>,char> {
    template<typename ParseContext>
    constexpr ParseContext::iterator parse(ParseContext &ctx) const {
        return ctx.begin();
    }
    template<typename FmtContext>
    FmtContext::iterator format(const poly_ops::draw::detail::fraction<T,SizeNum,SizeDenom> &f,FmtContext &ctx) const {
        return std::format_to(ctx.out(),"{}/{}",f.num,f.denom);
    }
};

template<typename T,unsigned int Size>
struct std::formatter<poly_ops::draw::detail::whole_and_frac<T,Size>,char> {
    template<typename ParseContext>
    constexpr ParseContext::iterator parse(ParseContext &ctx) const {
        return ctx.begin();
    }
    template<typename FmtContext>
    FmtContext::iterator format(const poly_ops::draw::detail::whole_and_frac<T,Size> &f,FmtContext &ctx) const {
        return std::format_to(ctx.out(),"{} + {}",f.whole,f.frac);
    }
};
#endif

#endif

#endif
