#ifndef poly_ops_strided_itr_hpp
#define poly_ops_strided_itr_hpp

#include <cstddef>
#include <concepts>
#include <ranges>
#include <type_traits>

#include "base.hpp"

namespace poly_ops {

namespace detail {
/* this iterator deliberately yields its own pointer */
template<typename T>
struct strided_itr {
    using value_type = T*;

    T *ptr;
    std::ptrdiff_t stride;

    strided_itr() = default;
    strided_itr(T *ptr,std::ptrdiff_t stride) : ptr(ptr), stride(stride) {}
    strided_itr(const strided_itr&) = default;

    strided_itr &operator=(const strided_itr&) = default;

    T *operator*() const { return ptr; }

    T *operator[](std::ptrdiff_t i) const {
        return ptr+i*stride;
    }

    strided_itr new_ptr(T *x) const {
        return {x,stride};
    }

    strided_itr &operator++() {
        ptr += stride;
        return *this;
    }
    strided_itr &operator--() {
        ptr -= stride;
        return *this;
    }

    strided_itr operator++(int) {
        return new_ptr(std::exchange(ptr,ptr+stride));
    }
    strided_itr operator--(int) {
        return new_ptr(std::exchange(ptr,ptr-stride));
    }

    strided_itr &operator+=(std::ptrdiff_t n) {
        ptr += n * stride;
        return *this;
    }
    strided_itr &operator-=(std::ptrdiff_t n) {
        ptr -= n * stride;
        return *this;
    }

    friend strided_itr operator+(const strided_itr &a,std::ptrdiff_t b) {
        return a.new_ptr(a.ptr + b*a.stride);
    }
    friend strided_itr operator+(std::ptrdiff_t a,const strided_itr &b) {
        return b.new_ptr(b.ptr + a*b.stride);
    }

    friend strided_itr operator-(const strided_itr &a,std::ptrdiff_t b) {
        return a.new_ptr(a.ptr - b*a.stride);
    }
    friend strided_itr operator-(std::ptrdiff_t a,const strided_itr &b) {
        return b.new_ptr(b.ptr - a*b.stride);
    }

    friend std::ptrdiff_t operator-(const strided_itr &a,const strided_itr &b) {
        return a.ptr - b.ptr;
    }

    friend auto operator<=>(const strided_itr &a,const strided_itr &b) {
        return a.ptr <=> b.ptr;
    }

    friend auto operator==(const strided_itr &a,const strided_itr &b) {
        return a.ptr == b.ptr;
    }
};

template<typename T,typename F>
auto _make_strided_range(T *start,std::size_t length,std::ptrdiff_t stride,F &&f) {
    return std::ranges::subrange(
        strided_itr<T>{start,stride},
        strided_itr<T>{start,stride}+length) | std::views::transform(std::forward<F>(f));
}
} // namespace detail

template<std::regular_invocable<const char*> F>
auto make_strided_range(const char *start,std::size_t length,std::ptrdiff_t stride,F &&f) {
    return detail::_make_strided_range(start,length,stride,std::forward<F>(f));
}

template<std::regular_invocable<char*> F>
auto make_strided_range(char *start,std::size_t length,std::ptrdiff_t stride,F &&f) {
    return detail::_make_strided_range(start,length,stride,std::forward<F>(f));
}

template<typename Coord,typename SourceT=Coord> auto blob_to_point_range(
    const void *data,
    std::ptrdiff_t coord_stride,
    std::ptrdiff_t point_stride,
    std::size_t points_length)
{
    return make_strided_range(
        reinterpret_cast<const char*>(data),
        points_length,
        point_stride,
        [=](const char *data) {
            return point_t(
                static_cast<Coord>(*reinterpret_cast<const SourceT*>(data)),
                static_cast<Coord>(*reinterpret_cast<const SourceT*>(data+coord_stride)));
        });
}

template<typename Coord,typename SourceT=Coord> auto blob_to_point_range_range(
    const void *data,
    std::ptrdiff_t coord_stride,
    std::ptrdiff_t point_stride,
    std::ptrdiff_t loop_stride,
    std::size_t points_length,
    std::size_t loops_length)
{
    return make_strided_range(
        reinterpret_cast<const char*>(data),
        loops_length,
        loop_stride,
        [=](const char *data) {
            return blob_to_point_range<Coord,SourceT>(data,coord_stride,point_stride,points_length);
        });
}

}

#endif
