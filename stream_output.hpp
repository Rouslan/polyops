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
};

template<typename T> auto pp(T &&x) { return _pp<T>{std::forward<T>(x)}; }

template<typename T> struct pp_printer {
    void operator()(std::ostream &os,const T &x) const {
        os << x;
    }
};

template<typename T> std::ostream &operator<<(std::ostream &os,const _pp<T> &x) {
    pp_printer<std::remove_reference_t<T>>{}(os,x.value);
    return os;
}

template<typename T> std::ostream &operator<<(std::ostream &os,_pp<T> &&x) {
    pp_printer<std::remove_reference_t<T>>{}(os,x.value);
    return os;
}

template<> struct pp_printer<bool> {
    void operator()(std::ostream &os,bool x) const {
        os << (x ? "true" : "false");
    }
};

template<typename Coord> struct pp_printer<poly_ops::point_t<Coord>> {
    void operator()(std::ostream &os,const poly_ops::point_t<Coord> &x) const {
        os << "point_t<coord_t>(" << x[0] << ',' << x[1] << ')';
    }
};

template<typename Index> struct pp_printer<poly_ops::detail::segment<Index>> {
    void operator()(std::ostream &os,const poly_ops::detail::segment<Index> &x) const {
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
    void operator()(std::ostream &os,const std::tuple<T...> &x) const {
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

#endif
