/* The methods here assume the Coord template parameter from objects defined in
poly_ops.hpp is always "coord_t" and Index is always "index_t".
*/

#ifndef stream_output_hpp
#define stream_output_hpp

#include <tuple>
#include <iostream>
#include <type_traits>

#include "poly_ops.hpp"

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
    return os << pp(x.value);
}

template<typename T1,typename... T> std::ostream &operator<<(std::ostream &os,arg_delimited<T1,T...> x) {
    return os << pp(x.value) << ',' << arg_delimited<T...>::from_tuple(x.values);
}

template<typename... T> struct pp_printer<std::tuple<T...>> {
    void operator()(std::ostream &os,indent_t,const std::tuple<T...> &x) const {
        os << "std::tuple(" << arg_delimited<T...>::from_tuple(x) << ')';
    }
};

template<typename T> struct delimited {
    const T &items;
    const char *delimiter;

    explicit delimited(const T &items,const char *delimiter=",")
        : items(items), delimiter(delimiter) {}
};
template<typename T> std::ostream &operator<<(std::ostream &os,delimited<T> d) {
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

#endif
