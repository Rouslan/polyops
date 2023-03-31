/* Define a 128 bit signed integer on platforms with a native 64 bit integer.

This integer only supports a handful of operations. */

#ifndef poly_ops_int128_hpp
#define poly_ops_int128_hpp

#include <concepts>

#ifdef __SIZEOF_INT128__
#  define POLY_OPS_128BIT_IMPL_BUILTIN 1
#  define POLY_OPS_HAVE_128BIT_INT 1
#elif defined(_MSC_VER)
#  if defined(_M_AMD64) || defined(_M_ARM64) || defined(_M_ARM64EC)
#    define POLY_OPS_128BIT_IMPL_MSVC 1
#    define POLY_OPS_HAVE_128BIT_INT 1
#  endif
#else
#  define POLY_OPS_128BIT_IMPL_OTHER 1
#endif

#ifndef POLY_OPS_128BIT_IMPL_BUILTIN
#define POLY_OPS_128BIT_IMPL_BUILTIN 0
#endif
#ifndef POLY_OPS_128BIT_IMPL_MSVC
#define POLY_OPS_128BIT_IMPL_MSVC 0
#endif
#ifndef POLY_OPS_128BIT_IMPL_OTHER
#define POLY_OPS_128BIT_IMPL_OTHER 0
#endif
#ifndef POLY_OPS_HAVE_128BIT_INT
#define POLY_OPS_HAVE_128BIT_INT 0
#endif



#if POLY_OPS_128BIT_IMPL_MSVC

#include <cmath>
#include <intrin.h>
#include <immintrin.h>

#pragma intrinsic(_mul128)
#pragma intrinsic(_umul128)
#pragma intrinsic(_addcarry_u64)
#pragma intrinsic(_subborrow_u64)

#else

#include <cstdint>

#endif

namespace poly_ops {

namespace detail {
template<typename T> concept builtin_number
    = std::integral<T> || std::floating_point<T>;

template<typename T> concept small_sints
    = std::signed_integral<T> && sizeof(T) < 64;
template<typename T> concept small_uints
    = std::unsigned_integral<T> && sizeof(T) < 64;

}

#if POLY_OPS_128BIT_IMPL_BUILTIN

namespace detail {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-pedantic"
using int128_t = __int128;
#pragma GCC diagnostic pop
}

class basic_int128 {
    detail::int128_t base;

public:
    basic_int128() noexcept = default;
    basic_int128(detail::int128_t x) noexcept : base(x) {}
    basic_int128(std::integral auto x) noexcept : base(x) {}
    explicit basic_int128(std::floating_point auto x) noexcept : base(static_cast<detail::int128_t>(x)) {}
    basic_int128(std::uint64_t hi,std::uint64_t lo) noexcept
        : base(detail::int128_t(lo) + (detail::int128_t(hi) << 64)) {}
    basic_int128(const basic_int128&) noexcept = default;

    basic_int128 &operator=(const basic_int128&) noexcept = default;
    basic_int128 &operator=(std::integral auto b) noexcept { base = b; return *this; }

    basic_int128 &operator+=(basic_int128 b) noexcept { base += b.base; return *this; }
    basic_int128 &operator+=(std::integral auto b) noexcept { base += b; return *this; }

    basic_int128 &operator-=(basic_int128 b) noexcept { base -= b.base; return *this; }
    basic_int128 &operator-=(std::integral auto b) noexcept { base -= b; return *this; }

    basic_int128 operator-() const noexcept { return -base; }

    operator detail::int128_t() const noexcept { return base; }
    template<detail::builtin_number T> explicit operator T() const noexcept { return static_cast<T>(base); }

    explicit operator bool() const noexcept { return base != 0; }

    friend basic_int128 operator+(basic_int128 a,basic_int128 b) noexcept {
        return a.base + b.base;
    }
    friend basic_int128 operator+(std::integral auto a,basic_int128 b) noexcept {
        return a + b.base;
    }
    friend basic_int128 operator+(basic_int128 a,std::integral auto b) noexcept {
        return a.base + b;
    }
    friend basic_int128 operator-(basic_int128 a,basic_int128 b) noexcept {
        return a.base - b.base;
    }
    friend basic_int128 operator-(std::integral auto a,basic_int128 b) noexcept {
        return a - b.base;
    }
    friend basic_int128 operator-(basic_int128 a,std::integral auto b) noexcept {
        return a.base - b;
    }

    friend bool operator==(basic_int128 a,basic_int128 b) noexcept {
        return a.base == b.base;
    }
    friend bool operator==(std::integral auto a,basic_int128 b) noexcept {
        return a == b.base;
    }
    friend bool operator==(basic_int128 a,std::integral auto b) noexcept {
        return a.base == b;
    }

    friend auto operator<=>(basic_int128 a,basic_int128 b) noexcept {
        return a.base <=> b.base;
    }
    friend auto operator<=>(std::integral auto a,basic_int128 b) noexcept {
        return a <=> b.base;
    }
    friend auto operator<=>(basic_int128 a,std::integral auto b) noexcept {
        return a.base <=> b;
    }

    std::uint64_t lo() const noexcept { return static_cast<std::uint64_t>(base); }
    std::uint64_t hi() const noexcept { return static_cast<std::uint64_t>(base >> 64); }

    static basic_int128 mul(std::int64_t a,std::int64_t b) noexcept {
        return static_cast<detail::int128_t>(a) * b;
    }
};

#elif POLY_OPS_128BIT_IMPL_MSVC

class basic_int128 {
    unsigned __int64 _lo;
    unsigned __int64 _hi;

public:
    basic_int128() noexcept = default;
    basic_int128(unsigned __int64 hi,unsigned __int64 lo) noexcept : _lo(lo), _hi(hi) {}
    basic_int128(__int64 b) noexcept
        : _lo(static_cast<unsigned __int64>(b)), _hi(static_cast<unsigned __int64>(b >> 63)) {}
    basic_int128(unsigned __int64 b) noexcept : _lo(b), _hi(0) {}
    basic_int128(detail::small_sints auto b) noexcept : basic_int128(static_cast<__int64>(b)) {}
    basic_int128(detail::small_uints auto b) noexcept : basic_int128(static_cast<unsigned __int64>(b)) {}
    explicit basic_int128(std::floating_point auto b) noexcept
        : _lo(static_cast<unsigned __int64>(static_cast<__int64>(std::fmod(b,T(0x1p64))))),
          _hi(static_cast<unsigned __int64>(static_cast<__int64>(b/T(0x1p64)))) {}
    basic_int128(const basic_int128&) noexcept = default;

    basic_int128 &operator=(const basic_int128&) noexcept = default;
    basic_int128 &operator=(__int64 b) noexcept {
        _hi = static_cast<unsigned __int64>(b >> 63);
        _lo = static_cast<unsigned __int64>(b);
        return *this;
    }
    basic_int128 &operator=(unsigned __int64 b) noexcept {
        _hi = 0;
        _lo = b;
        return *this;
    }
    basic_int128 &operator=(detail::small_sints auto b) noexcept {
        return *this = static_cast<__int64>(b);
    }
    basic_int128 &operator=(detail::small_uints auto b) noexcept {
        return *this = static_cast<unsigned __int64>(b);
    }

    basic_int128 operator-() const noexcept { return 0ul - *this; }

    explicit operator __int64() const noexcept {
        return static_cast<__int64>(_lo);
    }
    explicit operator unsigned __int64() const noexcept {
        return _lo;
    }
    template<std::integral T> explicit operator T() const noexcept {
        return static_cast<T>(static_cast<__int64>(_lo));
    }
    template<std::floating_point T> explicit operator T() const noexcept {
        return std::ldexp(static_cast<T>(static_cast<__int64>(_hi)),64)
            + static_cast<T>(static_cast<__int64>(_lo));
    }
    template<detail::builtin_number T> explicit operator T() const noexcept { return static_cast<T>(base); }

    explicit operator bool() const noexcept {
        return _lo != 0 || _hi != 0;
    }

    friend basic_int128 operator+(basic_int128 a,basic_int128 b) noexcept {
        basic_int128 r;
        _addcarry_u64(
            _addcarry_u64(0,a._lo,b._lo,&r._lo),
            a._hi,
            b._hi,
            &r._hi);
        return r;
    }
    friend basic_int128 operator+(std::integral auto a,basic_int128 b) noexcept {
        return b + a;
    }
    friend basic_int128 operator+(basic_int128 a,__int64 b) noexcept {
        basic_int128 r;
        _addcarry_u64(
            _addcarry_u64(0,a._lo,static_cast<unsigned __int64>(b),&r._lo),
            a._hi,
            static_cast<unsigned __int64>(b >> 63),
            &r._hi);
        return r;
    }
    friend basic_int128 operator+(basic_int128 a,unsigned __int64 b) noexcept {
        basic_int128 r;
        _addcarry_u64(
            _addcarry_u64(0,a._lo,b,&r._lo),
            a._hi,
            0,
            &r._hi);
        return r;
    }
    friend basic_int128 operator+(basic_int128 a,detail::small_sints auto b) noexcept {
        return a + static_cast<__int64>(b);
    }
    friend basic_int128 operator+(basic_int128 a,detail::small_uints auto b) noexcept {
        return a + static_cast<unsigned __int64>(b);
    }

    friend basic_int128 operator-(basic_int128 a,basic_int128 b) noexcept {
        basic_int128 r;
        _subborrow_u64(
            _subborrow_u64(0,a._lo,b._lo,&r._lo),
            a._hi,
            b._hi,
            &r._hi);
        return r;
    }
    friend basic_int128 operator-(__int64 a,basic_int128 b) noexcept {
        basic_int128 r;
        _subborrow_u64(
            _subborrow_u64(0,static_cast<unsigned __int64>(a),b._lo,&r._lo),
            static_cast<unsigned __int64>(a >> 63),
            b._hi,
            &r._hi);
        return r;
    }
    friend basic_int128 operator-(unsigned __int64 a,basic_int128 b) noexcept {
        basic_int128 r;
        _subborrow_u64(
            _subborrow_u64(0,a,b._lo,&r._lo),
            0,
            b._hi,
            &r._hi);
        return r;
    }
    friend basic_int128 operator-(detail::small_sints auto a,basic_int128 b) noexcept {
        return static_cast<__int64>(a) - b;
    }
    friend basic_int128 operator-(detail::small_uints auto a,basic_int128 b) noexcept {
        return static_cast<unsigned __int64>(a) - b;
    }
    friend basic_int128 operator-(basic_int128 a,__int64 b) noexcept {
        basic_int128 r;
        _subborrow_u64(
            _subborrow_u64(0,a._lo,static_cast<unsigned __int64>(b),&r._lo),
            a._hi,
            static_cast<unsigned __int64>(b >> 63),
            &r._hi);
        return r;
    }
    friend basic_int128 operator-(basic_int128 a,unsigned __int64 b) noexcept {
        basic_int128 r;
        _subborrow_u64(
            _subborrow_u64(0,a._lo,b,&r._lo),
            a._hi,
            0,
            &r._hi);
        return r;
    }
    friend basic_int128 operator-(basic_int128 a,detail::small_sints auto b) noexcept {
        return a - static_cast<__int64>(b);
    }
    friend basic_int128 operator-(basic_int128 a,detail::small_uints auto b) noexcept {
        return a - static_cast<unsigned __int64>(b);
    }

    friend bool operator==(basic_int128 a,basic_int128 b) noexcept {
        return a._lo == b._lo && a._hi == b._hi;
    }
    friend bool operator==(basic_int128 a,__int64 b) noexcept {
        return a._lo == b && a._hi == (b >> 63);
    }
    friend bool operator==(basic_int128 a,unsigned __int64 b) noexcept {
        return a._hi == 0 && a._lo == b;
    }
    friend bool operator==(basic_int128 a,detail::small_sints auto b) noexcept {
        return a == static_cast<__int64>(b);
    }
    friend bool operator==(basic_int128 a,detail::small_uints auto b) noexcept {
        return a == static_cast<unsigned __int64>(b);
    }

    friend bool operator<(basic_int128 a,basic_int128 b) noexcept {
        unsigned __int64 x;
        return _subborrow_u64(
            _subborrow_u64(0,a._lo,b._lo,&x),
            a._hi,
            b._hi,
            &x) != 0;
    }
    friend bool operator<(basic_int128 a,__int64 b) noexcept {
        __int64 x;
        return _subborrow_u64(
            _subborrow_u64(0,a._lo,static_cast<unsigned __int64>(b),&x),
            a._hi,
            static_cast<unsigned __int64>(b >> 63),
            &x) != 0;
    }
    friend bool operator<(basic_int128 a,unsigned __int64 b) noexcept {
        __int64 x;
        return _subborrow_u64(
            _subborrow_u64(0,a._lo,b,&x),
            a._hi,
            0,
            &x) != 0;
    }
    friend bool operator<(basic_int128 a,detail::small_sints auto b) noexcept {
        return a < static_cast<__int64>(b);
    }
    friend bool operator==(basic_int128 a,detail::small_uints auto b) noexcept {
        return a < static_cast<unsigned __int64>(b);
    }

    unsigned __int64 lo() const noexcept { return lo; }
    unsigned __int64 hi() const noexcept { return hi; }

    static basic_int128 mul(__int64 a,__int64 b) noexcept {
        basic_int128 r;
        r._lo = _mul128(a,b,&r._hi);
        return r;
    }

    static basic_int128 mul(unsigned __int64 a,unsigned __int64 b) noexcept {
        basic_int128 r;
        r._lo = _umul128(a,b,&r._hi);
        return r;
    }

    static basic_int128 mul(unsigned __int64 a,__int64 b) = delete;
    static basic_int128 mul(__int64 a,unsigned __int64 b) = delete;
};

#endif

}

#endif
