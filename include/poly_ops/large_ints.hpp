/* Define integers bigger than 64 bits.

This code checks for compiler extensions. To disable these checks and use only
standard C++, define POLY_OPS_NO_COMPILER_EXTENSIONS. */

#ifndef poly_ops_large_ints_hpp
#define poly_ops_large_ints_hpp

#include <concepts>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <algorithm>
#include <compare>
#include <bit>


/* The __int128 type is available */
#define _POLY_OPS_IMPL_HAVE_INT128BIT 0

/* The _mul128 intrinsic is available */
#define _POLY_OPS_IMPL_HAVE_MUL128 0

/* the _div64/_div128 intrinsic is available */
#define _POLY_OPS_IMPL_HAVE_DIVX 0

/* The _addcarry_X and _subborrow_X intrinsics are available */
#define _POLY_OPS_IMPL_INTEL_INTR 0

/* The __builtin_addcX and __builtin_subcX intrinsics are available */
#define _POLY_OPS_IMPL_CLANG_INTR 0

/* The __builtin_add_overflow and __builtin_sub_overflow intrinsics are
available */
#define _POLY_OPS_IMPL_GCC_INTR 0

/* GCC's extended assembly is available and the target architecture is x86 or
x86-64 */
#define _POLY_OPS_IMPL_GCC_X86_ASM 0


#define _POLY_OPS_FORCE_INLINE inline
#define _POLY_OPS_ARTIFICIAL inline
#define _POLY_OPS_RESTRICT


#ifndef POLY_OPS_NO_COMPILER_EXTENSIONS

#ifdef __SIZEOF_INT128__
#  undef _POLY_OPS_IMPL_HAVE_INT128BIT
#  define _POLY_OPS_IMPL_HAVE_INT128BIT 1
#endif

#if defined(_MSC_VER) || defined(__INTEL_LLVM_COMPILER)
#  include <immintrin.h>
#  if defined(_MSC_VER)
#    include <intrin.h>
#    ifdef _WIN64
#      if _MSC_VER >= 1920
#        undef _POLY_OPS_IMPL_HAVE_DIVX
#        define _POLY_OPS_IMPL_HAVE_DIVX 1
#        pragma intrinsic(_div128)
#        define _POLY_OPS_MSVC_DIV _div128
#        define _POLY_OPS_MSVC_UDIV _udiv128
#      endif
#      pragma intrinsic(_mul128)
#      pragma intrinsic(_umul128)
#      pragma intrinsic(_addcarry_u64)
#      pragma intrinsic(_subborrow_u64)
#      define _POLY_OPS_INTEL_ADDCARRY _addcarry_u64
#      define _POLY_OPS_INTEL_SUBBORROW _subborrow_u64
#      undef _POLY_OPS_IMPL_HAVE_MUL128
#      define _POLY_OPS_IMPL_HAVE_MUL128 1
#    else
#      if _MSC_VER >= 1920
#        undef _POLY_OPS_IMPL_HAVE_DIVX
#        define _POLY_OPS_IMPL_HAVE_DIVX 1
#        pragma intrinsic(_div64)
#        define _POLY_OPS_MSVC_DIV _div64
#        define _POLY_OPS_MSVC_UDIV _udiv64
#      endif
#      pragma intrinsic(_addcarry_u32)
#      pragma intrinsic(_subborrow_u32)
#      define _POLY_OPS_INTEL_ADDCARRY _addcarry_u32
#      define _POLY_OPS_INTEL_SUBBORROW _subborrow_u32
#    endif
#    undef _POLY_OPS_FORCE_INLINE
#    define _POLY_OPS_FORCE_INLINE __forceinline
#    undef _POLY_OPS_ARTIFICIAL
#    define _POLY_OPS_ARTIFICIAL __forceinline
#    undef _POLY_OPS_RESTRICT
#    define _POLY_OPS_RESTRICT __restrict
#  endif
#  undef _POLY_OPS_IMPL_INTEL_INTR
#  define _POLY_OPS_IMPL_INTEL_INTR 1
#elif defined(__GNUC__)
#  if (defined(__amd64__) || defined(__i386__))
#    undef _POLY_OPS_IMPL_GCC_X86_ASM
#    define _POLY_OPS_IMPL_GCC_X86_ASM 1
#  endif
#  if __has_builtin(__builtin_addc)
#    undef _POLY_OPS_IMPL_CLANG_INTR
#    define _POLY_OPS_IMPL_CLANG_INTR 1
#  endif
#  undef _POLY_OPS_IMPL_GCC_INTR
#  define _POLY_OPS_IMPL_GCC_INTR 1
#  undef _POLY_OPS_FORCE_INLINE
#  define _POLY_OPS_FORCE_INLINE __attribute__((always_inline)) inline
#  undef _POLY_OPS_ARTIFICIAL
#  define _POLY_OPS_ARTIFICIAL __attribute__((always_inline,artificial)) inline
#  undef _POLY_OPS_RESTRICT
#  define _POLY_OPS_RESTRICT __restrict__
#endif

#endif


namespace poly_ops_new {

#if _POLY_OPS_IMPL_HAVE_INT128BIT
namespace detail {
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wpragmas"
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-pedantic"
using int128_t = __int128;
using uint128_t = unsigned __int128;
#  pragma GCC diagnostic pop
}
#endif

/* full_int and full_uint should be the biggest integer types that the target
CPU can hold in a general-purpose register. As far as I know, std::intptr_t
should be it. If there are any platforms that a C++20 compiler targets, where
this is not the case, these definitions may need to be updated. */
using full_int = std::intptr_t;
using full_uint = std::uintptr_t;
template<bool Signed> using full_xint = std::conditional_t<Signed,full_int,full_uint>;


namespace detail {
#if _POLY_OPS_IMPL_CLANG_INTR
    _POLY_OPS_ARTIFICIAL unsigned int builtin_addc(unsigned int a,unsigned int b,unsigned int carry_in,unsigned int *carry_out) noexcept {
        return __builtin_addc(a,b,carry_in,carry_out);
    }
    _POLY_OPS_ARTIFICIAL unsigned long builtin_addc(unsigned long a,unsigned long b,unsigned long carry_in,unsigned long *carry_out) noexcept {
        return __builtin_addcl(a,b,carry_in,carry_out);
    }
    _POLY_OPS_ARTIFICIAL unsigned long long builtin_addc(unsigned long long a,unsigned long long b,unsigned long long carry_in,unsigned long long *carry_out) noexcept {
        return __builtin_addcll(a,b,carry_in,carry_out);
    }
    _POLY_OPS_ARTIFICIAL unsigned int builtin_subc(unsigned int a,unsigned int b,unsigned int carry_in,unsigned int *carry_out) noexcept {
        return __builtin_subc(a,b,carry_in,carry_out);
    }
    _POLY_OPS_ARTIFICIAL unsigned long builtin_subc(unsigned long a,unsigned long b,unsigned long carry_in,unsigned long *carry_out) noexcept {
        return __builtin_subcl(a,b,carry_in,carry_out);
    }
    _POLY_OPS_ARTIFICIAL unsigned long long builtin_subc(unsigned long long a,unsigned long long b,unsigned long long carry_in,unsigned long long *carry_out) noexcept {
        return __builtin_subcll(a,b,carry_in,carry_out);
    }
#endif

    /* whether __int128 counts as integral can vary depending on compiler
    settings, so we need to define our own versions of these concepts: */
#if _POLY_OPS_IMPL_HAVE_INT128BIT
    template<typename T> concept integral = std::integral<T>
        || std::same_as<std::remove_cv_t<T>,int128_t> || std::same_as<std::remove_cv_t<T>,uint128_t>;
    template<typename T> concept signed_integral
        = std::signed_integral<T> || std::same_as<std::remove_cv_t<T>,int128_t>;
    template<typename T> concept unsigned_integral
        = std::unsigned_integral<T> || std::same_as<std::remove_cv_t<T>,uint128_t>;
#else
    template<typename T> concept integral = std::integral<T>;
    template<typename T> concept signed_integral = std::signed_integral<T>;
    template<typename T> concept unsigned_integral = std::unsigned_integral<T>;
#endif

    template<typename T> concept builtin_number
        = integral<T> || std::floating_point<T>;

    template<typename T> concept small_sints
        = signed_integral<T> && sizeof(T) < sizeof(full_int);
    template<typename T> concept small_uints
        = unsigned_integral<T> && sizeof(T) < sizeof(full_int);
    template<typename T> concept small_xints
        = integral<T> && sizeof(T) < sizeof(full_int);

    template<bool Signed> constexpr full_uint extended_word(full_uint x) noexcept {
        if constexpr(Signed) {
            return static_cast<full_uint>(static_cast<full_int>(x) >> (sizeof(full_int)*8 - 1));
        } else {
            return 0;
        }
    }

    template<typename T> struct type_val { using type=T; };
    template<std::size_t Size> auto _bltin_sized_int() noexcept {
        if constexpr(Size <= 1) { return type_val<std::int8_t>{}; }
        if constexpr(Size == 2) { return type_val<std::int16_t>{}; }
        else if constexpr(Size <= 4) { return type_val<std::int32_t>{}; }
        else if constexpr(Size <= 8) { return type_val<std::int64_t>{}; }
        else { return type_val<void>{}; }
    }
    template<std::size_t Size> using bltin_sized_int = typename decltype(_bltin_sized_int<Size>())::type;

    template<bool Signed,typename T> struct _int_count {};

    /* Unsigned values are counted as being one bit larger than signed values */
    template<bool Signed,integral T> struct _int_count<Signed,T>
        : std::integral_constant<
            unsigned int,
            (sizeof(T) + sizeof(full_int) - (signed_integral<T> || !Signed)) / sizeof(full_int)> {};

    /* The value represented by this value is the number of full_uint instances
    the would be required to store all the bits of the integer-like T type. If
    Signed is true, one bit is reserved for the sign bit, which means if T is
    unsigned, one more bit is required.

    e.g.:
        _int_count<false,full_uint>::value == 1
        _int_count<true,full_uint>::value == 2
        _int_count<false,full_int>::value == 1
        _int_count<true,full_int>::value == 1
    */
    template<bool Signed,typename T> inline constexpr unsigned int int_count = _int_count<Signed,std::decay_t<T>>::value;

    template<bool Signed,typename T1,typename... T> inline constexpr unsigned int max_int_count = std::max({int_count<Signed,T1>,int_count<Signed,T>...});

    template<unsigned int HighestN,bool Signed=false,signed_integral T> constexpr auto compi_hi(T x) noexcept {
        if constexpr(HighestN > int_count<false,T>) {
            return static_cast<full_xint<Signed>>(x >> (sizeof(T) * 8 - 1));
        } else {
            return static_cast<full_xint<Signed>>(x >> (sizeof(full_int) * (HighestN-1) * 8));
        }
    }
    template<unsigned int HighestN,bool Signed=false,unsigned_integral T> constexpr auto compi_hi(T x) noexcept {
        if constexpr(HighestN > int_count<false,T>) {
            return full_xint<Signed>(0);
        } else {
            return static_cast<full_xint<Signed>>(x >> (sizeof(full_int) * (HighestN-1) * 8));
        }
    }

    template<unsigned int HighestN,integral T> _POLY_OPS_FORCE_INLINE constexpr auto compi_lo(T x) noexcept;

    template<typename T> struct _is_signed {};
    template<typename T> concept signed_type = signed_integral<T> || _is_signed<std::decay_t<T>>::value;
    template<typename T> concept unsigned_type = unsigned_integral<T> || !_is_signed<std::decay_t<T>>::value;
} // namespace detail

template<typename T> concept comp_or_single_i = requires {
    { detail::_int_count<false,T>::value };
};

namespace detail {
    template<typename T> concept has_hi_fun = requires(T value) {
        { value.hi() } -> std::convertible_to<full_uint>;
    };

    /* Here 'Signed' only affects whether the output type is signed. Whether 'x'
    is sign extended or zero extended depends on whether 'x' fulfills the
    'signed_type' concept. */
    template<unsigned int HighestN,bool Signed=false,has_hi_fun T> constexpr auto compi_hi(const T &x) noexcept {
        if constexpr(HighestN == int_count<false,T>) {
            return static_cast<full_xint<Signed>>(x.hi());
        } else if constexpr(HighestN < int_count<false,T>) {
            return compi_hi<HighestN,Signed>(x.lo());
        } else {
            return static_cast<full_xint<Signed>>(extended_word<signed_type<T>>(x.hi()));
        }
    }

    template<unsigned int HighestN,has_hi_fun T> constexpr auto compi_lo(const T &x) noexcept {
        if constexpr(HighestN == int_count<false,T>) {
            return x.lo();
        } else if constexpr(HighestN < int_count<false,T>) {
            return compi_lo<HighestN>(x.lo());
        } else {
            return x;
        }
    }

    template<integral T,typename U> T cast(const U &x) noexcept {
        constexpr unsigned int N = int_count<false,U>;
        if constexpr(N == 1) {
            return static_cast<T>(compi_hi<1>(x));
        } else if constexpr(int_count<false,T> >= N) {
            return (static_cast<T>(compi_hi<N>(x)) << ((N-1) * sizeof(full_int))) | cast<T>(compi_lo<N>(x));
        } else {
            return cast<T>(compi_lo<N>(x));
        }
    }
}

template<unsigned int N,bool Const> class _compound_xint_ref {
public:
    using value_type = std::conditional_t<Const,const full_uint,full_uint>;

private:
    value_type *_data;

public:
    explicit _POLY_OPS_ARTIFICIAL constexpr _compound_xint_ref(value_type *_data) noexcept : _data{_data} {}

    constexpr _compound_xint_ref(const _compound_xint_ref&) noexcept = default;
    _POLY_OPS_ARTIFICIAL constexpr _compound_xint_ref(const _compound_xint_ref<N,false> &b) noexcept requires Const : _data{b.data()} {}

    template<bool ConstB>
    requires (!Const)
    constexpr const _compound_xint_ref &operator=(const _compound_xint_ref<N,ConstB> &b) const noexcept {
        for(unsigned int i=0; i<N; ++i) _data[i] = b[i];
        return *this;
    }

    template<typename T>
    requires (!Const)
    constexpr void set(const T &b) const noexcept {
        if constexpr(N > 1) {
            lo().set(detail::compi_lo<N>(b));
        }
        hi() = detail::compi_hi<N>(b);
    }

    _POLY_OPS_ARTIFICIAL constexpr value_type &hi() const noexcept { return _data[N-1]; }
    _POLY_OPS_ARTIFICIAL constexpr auto lo() const noexcept requires (N > 1) {
        return _compound_xint_ref<N-1,Const>{_data};
    }

    _POLY_OPS_FORCE_INLINE constexpr value_type &operator[](unsigned int i) const noexcept {
        assert(i < N);
        return _data[i];
    }

    _POLY_OPS_ARTIFICIAL value_type *data() const noexcept { return _data; }
    _POLY_OPS_ARTIFICIAL value_type *begin() const noexcept { return _data; }
    _POLY_OPS_ARTIFICIAL value_type *end() const noexcept { return _data + N; }
};

template<bool Const> class _compound_xint_ref<0,Const> {
public:
    explicit constexpr _compound_xint_ref(full_uint*) noexcept {}
};

template<unsigned int N> using compound_xint_ref = _compound_xint_ref<N,false>;
template<unsigned int N> using const_compound_xint_ref = _compound_xint_ref<N,true>;

template<typename T,unsigned int N,bool Signed> concept safe_comp_or_single_i
    = detail::int_count<Signed,T> <= N && (Signed || detail::unsigned_type<T>);

template<unsigned int N,bool Signed> class compound_xint {
public:
    using value_type = full_uint;

private:
    full_uint _data[N];

public:
    compound_xint() noexcept = default;
    template<comp_or_single_i T> constexpr compound_xint(full_uint _hi,T _lo) noexcept {
        lo().set(_lo);
        hi() = _hi;
    }

    /* SFINAE doesn't seem to hold when using detail::int_count instead of the longer equivalent, under Clang 16.0.6 */
    template<comp_or_single_i T> explicit(detail::_int_count<Signed,std::decay_t<T>>::value > N || (detail::signed_type<T> && !Signed)) constexpr
    compound_xint(const T &b) : compound_xint{detail::compi_hi<N>(b),detail::compi_lo<N>(b)} {}

    /*template<std::floating_point T> explicit constexpr compound_int(T b) noexcept
        : _lo(static_cast<full_uint>(static_cast<full_int>(std::fmod(b,T(0x1p64))))),
          _hi(static_cast<full_uint>(static_cast<full_int>(b/T(0x1p64)))) {}*/
    constexpr compound_xint(const compound_xint&) noexcept = default;

    constexpr compound_xint &operator=(const compound_xint&) noexcept = default;

    template<safe_comp_or_single_i<N,Signed> T> compound_xint &operator=(const T &b) noexcept {
        lo() = detail::compi_lo<N>(b);
        hi() = detail::compi_hi<N>(b);
        return *this;
    }

    compound_xint operator-() const noexcept requires Signed { return full_uint(0) - *this; }

    template<detail::integral T> explicit operator T() const noexcept {
        return detail::cast<T>(*this);
    }
    /*template<std::floating_point T> explicit operator T() const noexcept {
        return std::ldexp(static_cast<T>(static_cast<full_int>(hi())),32*N)
            + static_cast<T>(lo());
    }*/

    constexpr bool negative() const noexcept {
        return Signed && static_cast<full_int>(hi()) < 0;
    }

    explicit constexpr operator bool() const noexcept {
        for(auto i : *this) {
            if(i != 0) return true;
        }
        return false;
    }

    _POLY_OPS_ARTIFICIAL constexpr operator compound_xint_ref<N>() { return compound_xint_ref<N>{_data}; }
    _POLY_OPS_ARTIFICIAL constexpr operator const_compound_xint_ref<N>() const { return const_compound_xint_ref<N>{_data}; }

    _POLY_OPS_ARTIFICIAL constexpr full_uint hi() const noexcept { return _data[N-1]; }
    _POLY_OPS_ARTIFICIAL constexpr full_uint &hi() noexcept { return _data[N-1]; }
    _POLY_OPS_ARTIFICIAL constexpr auto lo() const noexcept { return const_compound_xint_ref<N-1>{_data}; }
    _POLY_OPS_ARTIFICIAL constexpr auto lo() noexcept { return compound_xint_ref<N-1>{_data}; }

    _POLY_OPS_FORCE_INLINE constexpr full_uint operator[](unsigned int i) const noexcept {
        assert(i < N);
        return _data[i];
    }
    _POLY_OPS_FORCE_INLINE constexpr full_uint &operator[](unsigned int i) noexcept {
        assert(i < N);
        return _data[i];
    }

    std::size_t size() const { return N; }

    _POLY_OPS_ARTIFICIAL const full_uint *data() const noexcept { return _data; }
    _POLY_OPS_ARTIFICIAL full_uint *data() noexcept { return _data; }
    _POLY_OPS_ARTIFICIAL const full_uint *begin() const noexcept { return _data; }
    _POLY_OPS_ARTIFICIAL full_uint *begin() noexcept { return _data; }
    _POLY_OPS_ARTIFICIAL const full_uint *end() const noexcept { return _data + N; }
    _POLY_OPS_ARTIFICIAL full_uint *end() noexcept { return _data + N; }
};

template<bool Signed> class compound_xint<1,Signed> {
public:
    using value_type = full_uint;

private:
    full_uint _data[1];

public:
    compound_xint() noexcept = default;
    constexpr compound_xint(full_uint b) noexcept : _data{b} {}

    explicit(!Signed) constexpr compound_xint(full_int b) noexcept
        : _data{static_cast<full_uint>(b)} {}

    template<comp_or_single_i T> explicit(detail::int_count<Signed,T> > 1 || (detail::signed_type<T> && !Signed)) constexpr
    compound_xint(const T &b) : _data{detail::compi_hi<1>(b)} {}

    template<std::floating_point T> explicit constexpr compound_xint(T b) noexcept
        : _data{static_cast<full_uint>(static_cast<full_int>(b))} {}

    constexpr compound_xint(const compound_xint&) noexcept = default;

    compound_xint &operator=(const compound_xint &b) noexcept = default;

    template<safe_comp_or_single_i<1,Signed> T> compound_xint &operator=(const T &b) noexcept {
        hi() = detail::compi_hi<1>(b);
        return *this;
    }

    compound_xint operator-() const noexcept requires Signed { return compound_xint(-static_cast<full_int>(hi())); }

    template<detail::builtin_number T> explicit operator T() const noexcept {
        return static_cast<T>(static_cast<full_int>(hi()));
    }

    constexpr bool negative() const noexcept {
        return Signed && static_cast<full_int>(hi()) < 0;
    }

    explicit constexpr operator bool() const noexcept {
        return hi() != 0;
    }

    _POLY_OPS_ARTIFICIAL constexpr operator compound_xint_ref<1>() { return compound_xint_ref<1>{_data}; }
    _POLY_OPS_ARTIFICIAL constexpr operator const_compound_xint_ref<1>() const { return const_compound_xint_ref<1>{_data}; }

    _POLY_OPS_ARTIFICIAL constexpr full_uint hi() const noexcept { return _data[0]; }
    _POLY_OPS_ARTIFICIAL constexpr full_uint &hi() noexcept { return _data[0]; }

    _POLY_OPS_FORCE_INLINE constexpr full_uint operator[](unsigned int i) const noexcept {
        assert(i == 0);
        return _data[i];
    }
    _POLY_OPS_FORCE_INLINE constexpr full_uint &operator[](unsigned int i) noexcept {
        assert(i == 0);
        return _data[i];
    }

    std::size_t size() const { return 1; }

    _POLY_OPS_ARTIFICIAL const full_uint *data() const noexcept { return _data; }
    _POLY_OPS_ARTIFICIAL full_uint *data() noexcept { return _data; }
    _POLY_OPS_ARTIFICIAL const full_uint *begin() const noexcept { return _data; }
    _POLY_OPS_ARTIFICIAL full_uint *begin() noexcept { return _data; }
    _POLY_OPS_ARTIFICIAL const full_uint *end() const noexcept { return _data + 1; }
    _POLY_OPS_ARTIFICIAL full_uint *end() noexcept { return _data + 1; }
};

namespace detail {
    template<unsigned int HighestN,integral T> _POLY_OPS_FORCE_INLINE constexpr auto compi_lo(T x) noexcept {
        if constexpr(HighestN >= int_count<false,T>) {
            return x;
        } else {
            return compound_xint<HighestN-1,false>{x};
        }
    }

    template<bool DestSigned,unsigned int N,bool SourceSigned>
    struct _int_count<DestSigned,compound_xint<N,SourceSigned>> : std::integral_constant<unsigned int,N+(DestSigned && !SourceSigned)> {};
    template<bool DestSigned,unsigned int N,bool Const>
    struct _int_count<DestSigned,_compound_xint_ref<N,Const>> : std::integral_constant<unsigned int,N> {};

    template<unsigned int N,bool Signed> struct _is_signed<compound_xint<N,Signed>> : std::integral_constant<bool,Signed> {};
    template<unsigned int N,bool Const> struct _is_signed<_compound_xint_ref<N,Const>> : std::false_type {};
}

template<unsigned int N> using compound_uint = compound_xint<N,false>;
template<unsigned int N> using compound_int = compound_xint<N,true>;

namespace detail {
    /* The __builtin_add_overflow/__builtin_sub_overflow intrinsics often
    produce inferior machine code on GCC and ICC (Clang seems to do well), thus
    they are low on the list of fall-backs */
#if _POLY_OPS_IMPL_INTEL_INTR
    using carry_t = unsigned char;
    _POLY_OPS_FORCE_INLINE carry_t _addcarry(full_uint a,full_uint b,full_uint &out,carry_t carry=0) noexcept {
        return _POLY_OPS_INTEL_ADDCARRY(carry,a,b,&out);
    }
    _POLY_OPS_FORCE_INLINE carry_t _addcarry_to(full_uint &a,full_uint b,carry_t carry=0) noexcept {
        return _POLY_OPS_INTEL_ADDCARRY(carry,a,b,&a);
    }
    _POLY_OPS_FORCE_INLINE carry_t _subborrow(full_uint a,full_uint b,full_uint &out,carry_t carry=0) noexcept {
        return _POLY_OPS_INTEL_SUBBORROW(carry,a,b,&out);
    }
    _POLY_OPS_FORCE_INLINE carry_t _subborrow_to(full_uint &a,full_uint b,carry_t carry=0) noexcept {
        return _POLY_OPS_INTEL_SUBBORROW(carry,a,b,&a);
    }
#elif _POLY_OPS_IMPL_CLANG_INTR
    using carry_t = full_uint;
    _POLY_OPS_FORCE_INLINE carry_t _addcarry(full_uint a,full_uint b,full_uint &out,carry_t carry_i=0) noexcept {
        full_uint carry_o;
        out = builtin_addc(a,b,carry_i,&carry_o);
        return carry_o;
    }
    _POLY_OPS_FORCE_INLINE carry_t _addcarry_to(full_uint &a,full_uint b,carry_t carry_i=0) noexcept {
        full_uint carry_o;
        a = builtin_addc(a,b,carry_i,&carry_o);
        return carry_o;
    }
    _POLY_OPS_FORCE_INLINE carry_t _subborrow(full_uint a,full_uint b,full_uint &out,carry_t carry_i=0) noexcept {
        full_uint carry_o;
        out = builtin_subc(a,b,carry_i,&carry_o);
        return carry_o;
    }
    _POLY_OPS_FORCE_INLINE carry_t _subborrow_to(full_uint &a,full_uint b,carry_t carry_i=0) noexcept {
        full_uint carry_o;
        a = builtin_subc(a,b,carry_i,&carry_o);
        return carry_o;
    }
#elif _POLY_OPS_IMPL_GCC_INTR
    using carry_t = full_uint;
    _POLY_OPS_FORCE_INLINE carry_t _addcarry(full_uint a,full_uint b,full_uint &out,carry_t carry) noexcept {
        carry_t carry2 = __builtin_add_overflow(a,b,&out);
        return carry2 | __builtin_add_overflow(out,carry,&out);
    }
    _POLY_OPS_FORCE_INLINE carry_t _addcarry(full_uint a,full_uint b,full_uint &out) noexcept {
        return __builtin_add_overflow(a,b,&out);
    }
    _POLY_OPS_FORCE_INLINE carry_t _addcarry_to(full_uint &a,full_uint b,carry_t carry) noexcept {
        carry_t carry2 = __builtin_add_overflow(a,b,&a);
        return carry2 | __builtin_add_overflow(a,carry,&a);
    }
    _POLY_OPS_FORCE_INLINE carry_t _addcarry_to(full_uint &a,full_uint b) noexcept {
        return __builtin_add_overflow(a,b,&a);
    }
    _POLY_OPS_FORCE_INLINE carry_t _subborrow(full_uint a,full_uint b,full_uint &out,carry_t carry) noexcept {
        carry_t carry2 = __builtin_sub_overflow(a,b,&out);
        return carry2 | __builtin_sub_overflow(out,carry,&out);
    }
    _POLY_OPS_FORCE_INLINE carry_t _subborrow(full_uint a,full_uint b,full_uint &out) noexcept {
        return __builtin_sub_overflow(a,b,&out);
    }
    _POLY_OPS_FORCE_INLINE carry_t _subborrow_to(full_uint &a,full_uint b,carry_t carry) noexcept {
        carry_t carry2 = __builtin_sub_overflow(a,b,&a);
        return carry2 | __builtin_sub_overflow(a,carry,&a);
    }
    _POLY_OPS_FORCE_INLINE carry_t _subborrow_to(full_uint &a,full_uint b) noexcept {
        return __builtin_sub_overflow(a,b,&a);
    }
#else
    using carry_t = full_uint;
    _POLY_OPS_FORCE_INLINE carry_t _addcarry(full_uint a,full_uint b,full_uint &out,carry_t carry=0) noexcept {
        out = a + b;
        carry_t cr = out < a;
        out += carry;
        return (out < a) | cr;
    }
    _POLY_OPS_FORCE_INLINE carry_t _addcarry_to(full_uint &a,full_uint b,carry_t carry=0) noexcept {
        a = a + b;
        carry_t cr = a < b;
        a += carry;
        return (a < b) | cr;
    }
    _POLY_OPS_FORCE_INLINE carry_t _subborrow(full_uint a,full_uint b,full_uint &out,carry_t carry=0) noexcept {
        out = a - b;
        carry_t cr = out > a;
        out -= carry;
        return (out > a) | cr;
    }
    _POLY_OPS_FORCE_INLINE carry_t _subborrow_to(full_uint &a,full_uint b,carry_t carry=0) noexcept {
        full_uint prev_a = a;
        a = a - b;
        carry_t cr = a > prev_a;
        a -= carry;
        return (a > prev_a) | cr;
    }
#endif

/* Clang already transforms the C++ expressions into shrd/shld instructions,
making inline assembly unnecessary */
#if _POLY_OPS_IMPL_GCC_X86_ASM && !defined(__clang__)
    _POLY_OPS_FORCE_INLINE void _ext_shift_right_to(full_uint &a,full_uint fill,unsigned char amount) noexcept {
        if constexpr(__builtin_constant_p(a) && __builtin_constant_p(fill) && __builtin_constant_p(amount)) {
            if(amount == 0) return;
            a = (a >> amount) | (fill << (sizeof(full_uint)*8 - amount));
        } else {
            __asm__("shrd {%b[cnt], %[src], %[dest]|%[dest], %[src], %b[cnt]}" : [dest]"+rm"(a) : [src]"r"(fill), [cnt]"cJ"(amount) : "cc");
        }
    }
    _POLY_OPS_FORCE_INLINE void _ext_shift_left_to(full_uint &a,full_uint fill,unsigned char amount) noexcept {
        if constexpr(__builtin_constant_p(a) && __builtin_constant_p(fill) && __builtin_constant_p(amount)) {
            if(amount == 0) return;
            a = (a << amount) | (fill >> (sizeof(full_uint)*8 - amount));
        } else {
            __asm__("shld {%b[cnt], %[src], %[dest]|%[dest], %[src], %b[cnt]}" : [dest]"+rm"(a) : [src]"r"(fill), [cnt]"cJ"(amount) : "cc");
        }
    }
#else
    _POLY_OPS_FORCE_INLINE void _ext_shift_right_to(full_uint &a,full_uint fill,unsigned char amount) noexcept {
        if(amount == 0) return;
        a = (a >> amount) | (fill << (sizeof(full_uint)*8 - amount));
    }
    _POLY_OPS_FORCE_INLINE void _ext_shift_left_to(full_uint &a,full_uint fill,unsigned char amount) noexcept {
        if(amount == 0) return;
        a = (a << amount) | (fill >> (sizeof(full_uint)*8 - amount));
    }
#endif

    struct always_zero {
        _POLY_OPS_ARTIFICIAL operator full_uint() const noexcept { return 0; }
    };

    template<bool> always_zero extended_word(always_zero x) noexcept { return x; }

    template<unsigned int N> struct compound_zero {
        static_assert(N > 0);

        _POLY_OPS_ARTIFICIAL always_zero hi() const noexcept { return {}; }
        _POLY_OPS_ARTIFICIAL compound_zero<N-1> lo() const noexcept { return {}; }
    };
    template<> struct compound_zero<1> {
        _POLY_OPS_ARTIFICIAL always_zero hi() const noexcept { return {}; };
    };

    template<bool Signed,unsigned int N>
    struct _int_count<Signed,compound_zero<N>> { static constexpr unsigned int value = N; };

    /* This is like compound_int of size N+Shift where the bottom 'Shift'
    components are zero. */
    template<unsigned int Shift,unsigned int N> class padded_compound_int {
        padded_compound_int<Shift,N-1> _lo;
        full_uint _hi;

    public:
        explicit padded_compound_int(const_compound_xint_ref<N> b) noexcept : _lo{b.lo()}, _hi{b.hi()} {}
        padded_compound_int(const padded_compound_int&) noexcept = default;

        padded_compound_int &operator=(const padded_compound_int&) noexcept = default;

        full_uint hi() const noexcept { return _hi; }
        const padded_compound_int<Shift,N-1> &lo() const noexcept { return _lo; }
    };
    template<unsigned int Shift> class padded_compound_int<Shift,1> {
        full_uint _hi;

    public:
        explicit padded_compound_int(const_compound_xint_ref<1> b) noexcept : _hi{b.hi()} {}
        padded_compound_int(const padded_compound_int&) noexcept = default;

        padded_compound_int &operator=(const padded_compound_int&) noexcept = default;

        full_uint hi() const noexcept { return _hi; }
        const compound_zero<Shift> lo() const noexcept { return {}; }
    };

    template<bool Signed,unsigned int Shift,unsigned int N>
    struct _int_count<Signed,padded_compound_int<Shift,N>> { static constexpr unsigned int value = N + Shift + Signed; };

    template<unsigned int Shift,unsigned int N> struct _is_signed<padded_compound_int<Shift,N>> : std::false_type {};

    template<unsigned int Shift,typename T>
    _POLY_OPS_ARTIFICIAL padded_compound_int<Shift,int_count<false,T>> shift(const T &x) noexcept {
        return padded_compound_int<Shift,int_count<false,T>>{x};
    }

    _POLY_OPS_FORCE_INLINE carry_t _addcarry(always_zero,always_zero,full_uint &out,carry_t carry=0) noexcept {
        out = carry;
        return 0;
    }
    _POLY_OPS_FORCE_INLINE carry_t _addcarry(full_uint a,always_zero,full_uint &out) noexcept {
        out = a;
        return 0;
    }
    _POLY_OPS_FORCE_INLINE carry_t _addcarry(always_zero,full_uint b,full_uint &out) noexcept {
        out = b;
        return 0;
    }
    _POLY_OPS_FORCE_INLINE carry_t _addcarry_to(full_uint&,always_zero) noexcept {
        return 0;
    }
    _POLY_OPS_FORCE_INLINE carry_t _subborrow(always_zero,always_zero,full_uint &out) noexcept {
        out = 0;
        return 0;
    }
    _POLY_OPS_FORCE_INLINE carry_t _subborrow(full_uint a,always_zero,full_uint &out) noexcept {
        out = a;
        return 0;
    }
    _POLY_OPS_FORCE_INLINE carry_t _subborrow_to(full_uint&,always_zero) noexcept {
        return 0;
    }
    _POLY_OPS_FORCE_INLINE void _ext_shift_right_to(full_uint &a,always_zero,unsigned char amount) noexcept {
        a = a >> amount;
    }
    template<typename T> _POLY_OPS_FORCE_INLINE full_uint _ext_shift_right(full_uint a,T fill,unsigned char amount) noexcept {
        full_uint r = a;
        _ext_shift_right_to(r,fill,amount);
        return r;
    }
    _POLY_OPS_FORCE_INLINE full_uint _ext_shift_right(always_zero,full_uint fill,unsigned char amount) noexcept {
        return amount ? fill << (sizeof(full_uint)*8 - amount) : 0;
    }
    _POLY_OPS_FORCE_INLINE full_uint _ext_shift_right(always_zero,always_zero,unsigned char) noexcept {
        return 0;
    }
    _POLY_OPS_FORCE_INLINE void _ext_shift_left_to(full_uint &a,always_zero,unsigned char amount) noexcept {
        a = a << amount;
    }
    template<typename T> _POLY_OPS_FORCE_INLINE full_uint _ext_shift_left(full_uint a,T fill,unsigned char amount) noexcept {
        full_uint r = a;
        _ext_shift_left_to(r,fill,amount);
        return r;
    }
    _POLY_OPS_FORCE_INLINE full_uint _ext_shift_left(always_zero,full_uint fill,unsigned char amount) noexcept {
        return amount ? fill >> (sizeof(full_uint)*8 - amount) : 0;
    }
    _POLY_OPS_FORCE_INLINE full_uint _ext_shift_left(always_zero,always_zero,unsigned char) noexcept {
        return 0;
    }

    template<unsigned int N,typename T,typename U> carry_t addcarry(const T &a,const U &b,compound_xint_ref<N> out) noexcept {
        if constexpr(N > 1) {
            return _addcarry(
                compi_hi<N>(a),
                compi_hi<N>(b),
                out.hi(),
                addcarry(compi_lo<N>(a),compi_lo<N>(b),out.lo()));
        } else {
            return _addcarry(compi_hi<N>(a),compi_hi<N>(b),out.hi());
        }
    }
    template<unsigned int N,typename T> carry_t addcarry(compound_xint_ref<N> a,const T &b) noexcept {
        if constexpr(N > 1) {
            return _addcarry_to(
                a.hi(),
                compi_hi<N>(b),
                addcarry(a.lo(),compi_lo<N>(b)));
        } else {
            return _addcarry_to(a.hi(),compi_hi<N>(b));
        }
    }
    template<unsigned int N,typename T,typename U> carry_t subborrow(const T &a,const U &b,compound_xint_ref<N> out) noexcept {
        if constexpr(N > 1) {
            return _subborrow(
                compi_hi<N>(a),
                compi_hi<N>(b),
                out.hi(),
                subborrow(compi_lo<N>(a),compi_lo<N>(b),out.lo()));
        } else {
            return _subborrow(compi_hi<N>(a),compi_hi<N>(b),out.hi());
        }
    }
    template<unsigned int N,typename T> carry_t subborrow(compound_xint_ref<N> a,const T &b) noexcept {
        if constexpr(N > 1) {
            return _subborrow_to(
                a.hi(),
                compi_hi<N>(b),
                subborrow(a.lo(),compi_lo<N>(b)));
        } else {
            return _subborrow_to(a.hi(),compi_hi<N>(b));
        }
    }

    template<unsigned int N,typename T,typename U> carry_t subborrow_or(const T &a,const U &b,full_uint &out) noexcept {
        full_uint x;
        carry_t r;
        if constexpr(N > 1) {
            r = _subborrow(
                compi_hi<N>(a),
                compi_hi<N>(b),
                x,
                subborrow_or<N-1>(compi_lo<N>(a),compi_lo<N>(b),out));
        } else {
            r = _subborrow(compi_hi<N>(a),compi_hi<N>(b),x);
        }
        out |= x;
        return r;
    }

    template<bool Signed,unsigned int N> void _shift_right(compound_xint_ref<N> r,const_compound_xint_ref<N> a,unsigned int amount) noexcept {
        unsigned int word = amount / (sizeof(full_int)*8);
        unsigned char bit = amount % (sizeof(full_int)*8);

        unsigned int i=0;
        for(; i<(N-1-word); ++i) {
            r[i] = _ext_shift_right(a[i+word],a[i+1+word],bit);
        }
        if(i < (N-word)) {
            r[i] = static_cast<full_uint>(static_cast<full_xint<Signed>>(a[i+word]) >> bit);
        }
        for(++i; i<N; ++i) {
            r[i] = extended_word<Signed>(a[N-1]);
        }
    }

    template<unsigned int N> void _shift_left(compound_xint_ref<N> r,const_compound_xint_ref<N> a,unsigned int amount) noexcept {
        unsigned int word = amount / (sizeof(full_int)*8);
        unsigned char bit = amount % (sizeof(full_int)*8);

        unsigned int i=N;
        while(i > word+1) {
            --i;
            r[i] = _ext_shift_left(a[i-word],a[i-1-word],bit);
        }
        if(i > word) {
            --i;
            r[i] = a[i-word] << bit;
        }
        while(i > 0) {
            --i;
            r[i] = 0;
        }
    }
}

template<typename T,unsigned int N=detail::int_count<false,T>>
void copy(full_uint to[],const T &from) noexcept {
    copy(to,compi_lo<N>(from));
    to[N-1] = compi_hi<N>(from);
}

template<typename T,typename U> auto add(const T &a,const U &b) noexcept {
    constexpr bool signed_ = detail::signed_type<T> || detail::signed_type<U>;
    constexpr unsigned int N = detail::max_int_count<signed_,T,U>;
    compound_xint<N,signed_> r;
    detail::addcarry<N>(a,b,r);
    return r;
}
template<typename T,typename U> auto sub(const T &a,const U &b) noexcept {
    constexpr bool signed_ = detail::signed_type<T> || detail::signed_type<U>;
    constexpr unsigned int N = detail::max_int_count<signed_,T,U>;
    compound_xint<N,signed_> r;
    detail::subborrow<N>(a,b,r);
    return r;
}

template<typename T> auto abs(const T &x) noexcept {
    using namespace detail;

    constexpr unsigned int N = int_count<false,T>;
    constexpr bool Signed = signed_type<T>;

    if constexpr(!Signed) {
        return x;
    } else if constexpr(builtin_number<T>) {
        return std::abs(x);
    } else if constexpr(N == 1) {
        return compound_xint<N,true>(std::abs(compi_hi<1,true>(x)));
    } else {
        return compi_hi<N,true>(x) < 0 ? -x : x;
    }
}

void ar_add(unsigned int n,const full_uint *a,const full_uint *b,full_uint *out) noexcept {
    detail::carry_t carry = 0;
    for(unsigned int i=0; i<n; ++i) {
        carry = detail::_addcarry(a[i],b[i],out[i],carry);
    }
}
void ar_sub(unsigned int n,const full_uint *a,const full_uint *b,full_uint *out) noexcept {
    detail::carry_t carry = 0;
    for(unsigned int i=0; i<n; ++i) {
        carry = detail::_subborrow(a[i],b[i],out[i],carry);
    }
}

template<typename T> int countl_zero(const T &x) noexcept {
    constexpr unsigned int N = detail::int_count<false,T>;

    if constexpr(N > 1) {
        if(detail::compi_hi<N>(x) == 0)
            return countl_zero(detail::compi_lo<N>(x)) + static_cast<int>(sizeof(full_uint)*8);
    }
    return std::countl_zero(detail::compi_hi<N>(x));
}

template<typename T> auto shift_right(const T &a,unsigned int amount) noexcept {
    constexpr unsigned int N = detail::int_count<false,T>;
    constexpr bool Signed = detail::signed_type<T>;
    compound_xint<N,Signed> r;
    detail::_shift_right<Signed,N>(r,a,amount);
    return r;
}

template<typename T> auto shift_left(const T &a,unsigned char amount) noexcept {
    constexpr unsigned int N = detail::int_count<false,T>;
    constexpr bool Signed = detail::signed_type<T>;
    compound_xint<N,Signed> r;
    detail::_shift_left<N>(r,a,amount);
    return r;
}

template<typename T,typename U,unsigned int N = detail::max_int_count<detail::signed_type<T> || detail::signed_type<U>,T,U>>
bool eq(const T &a,const U &b) noexcept {
    if constexpr(N > 1) {
        if(!eq<T,U,N-1>(detail::compi_lo<N>(a),detail::compi_lo<N>(b))) return false;
    }
    return detail::compi_hi<N>(a) == detail::compi_hi<N>(b);
}

template<typename T,typename U> std::strong_ordering cmp(const T &a,const U &b) noexcept {
    using namespace detail;

    constexpr bool Signed = signed_type<T> || signed_type<U>;
    constexpr unsigned int N = max_int_count<Signed,T,U>;

    if constexpr(N > 1) {
        full_uint diff = 0;
        carry_t c;
        if constexpr(Signed) {
            carry_t c = subborrow_or<N-1>(compi_lo<N>(a),compi_lo<N>(b),diff);

            auto a_hi = compi_hi<N,true>(a);
            auto b_hi = compi_hi<N,true>(b);
            return a_hi != b_hi
                ? (a_hi <=> b_hi)
                : (diff
                    ? (c ? std::strong_ordering::less : std::strong_ordering::greater)
                    : std::strong_ordering::equal);
        } else {
            c = subborrow_or<N>(a,b,diff);
            return c ? std::strong_ordering::less : (diff ? std::strong_ordering::greater : std::strong_ordering::equal);
        }
    } else {
        return compi_hi<N,Signed>(a) <=> compi_hi<N,Signed>(b);
    }
}

template<unsigned int N,bool Signed> bool negative(const compound_xint<N,Signed> &x) noexcept {
    return x.negative();
}
template<detail::integral T> bool negative(T x) noexcept {
    return x < 0;
}

template<unsigned int N,bool Signed,comp_or_single_i T> bool operator==(const compound_xint<N,Signed> &a,const T &b) noexcept {
    return eq(a,b);
}
template<detail::integral T,unsigned int N,bool Signed> bool operator==(const T &a,const compound_xint<N,Signed> &b) noexcept {
    return eq(a,b);
}

template<unsigned int N,bool Signed,comp_or_single_i T> auto operator+(const compound_xint<N,Signed> &a,const T &b) noexcept {
    return add(a,b);
}
template<detail::integral T,unsigned int N,bool Signed> auto operator+(const T &a,const compound_xint<N,Signed> &b) noexcept {
    return add(b,a);
}

template<unsigned int N,bool Signed,comp_or_single_i T> auto operator-(const compound_xint<N,Signed> &a,const T &b) noexcept {
    return sub(a,b);
}
template<detail::integral T,unsigned int N,bool Signed> auto operator-(const T &a,const compound_xint<N,Signed> &b) noexcept {
    return sub(a,b);
}

template<unsigned int N,bool Signed,safe_comp_or_single_i<N,Signed> T> compound_xint<N,Signed> &operator+=(compound_xint<N,Signed> &a,const T &b) noexcept {
    detail::addcarry<N>(a,b);
    return a;
}
template<unsigned int N,bool Signed,safe_comp_or_single_i<N,Signed> T> compound_xint<N,Signed> &operator-=(compound_xint<N,Signed> &a,const T &b) noexcept {
    detail::subborrow<N>(a,b);
    return a;
}

template<unsigned int N,bool Signed> compound_xint<N,Signed> operator>>(const compound_xint<N,Signed> &a,unsigned char amount) noexcept {
    return shift_right(a,amount);
}

template<unsigned int N,bool Signed> compound_xint<N,Signed> &operator>>=(compound_xint<N,Signed> &a,unsigned char amount) noexcept {
    detail::_shift_right<Signed,N>(a,a,amount);
    return a;
}

template<unsigned int N,bool Signed> compound_xint<N,Signed> operator<<(const compound_xint<N,Signed> &a,unsigned char amount) noexcept {
    return shift_left(a,amount);
}

template<unsigned int N,bool Signed> compound_xint<N,Signed> &operator<<=(compound_xint<N,Signed> &a,unsigned char amount) noexcept {
    detail::_shift_left<N>(a,a,amount);
    return a;
}

template<unsigned int N,bool Signed,comp_or_single_i T> auto operator<=>(const compound_xint<N,Signed> &a,const T &b) noexcept {
    return cmp(a,b);
}

namespace detail {
    template<bool Signed> auto mul1x1(full_xint<Signed> a,full_xint<Signed> b) noexcept {
        static_assert(sizeof(full_int) <= 4 || sizeof(full_int) == 8);

        if constexpr(sizeof(full_int) == 8) {
#if _POLY_OPS_IMPL_HAVE_INT128BIT
            return compound_xint<2,Signed>{
                static_cast<std::conditional_t<Signed,int128_t,uint128_t>>(a) * b};
#elif _POLY_OPS_IMPL_HAVE_MUL128
            full_xint<Signed> hi, lo;
            if constexpr(Signed) {
                lo = _mul128(a,b,&hi);
            } else {
                lo = _umul128(a,b,&hi);
            }
            return compound_xint<2,Signed>{static_cast<full_uint>(hi),static_cast<full_uint>(lo)};
#else
            std::uint64_t au = static_cast<std::uint64_t>(a);
            std::uint64_t bu = static_cast<std::uint64_t>(b);
            std::uint64_t lo_lo = (au & 0xffffffff) * (bu & 0xffffffff);
            std::uint64_t hi_lo = (au >> 32) * (bu & 0xffffffff);
            std::uint64_t lo_hi = (au & 0xffffffff) * (bu >> 32);
            std::uint64_t hi_hi = (au >> 32) * (bu >> 32);

            std::uint64_t cross = (lo_lo >> 32) + (hi_lo & 0xffffffff) + lo_hi;

            std::uint64_t r_hi = (hi_lo >> 32) + (cross >> 32) + hi_hi;
            std::uint64_t r_lo = (cross << 32) | (lo_lo & 0xffffffff);

            if constexpr(Signed) {
                r_hi += bu * static_cast<std::uint64_t>(a >> 63) + au * static_cast<std::uint64_t>(b >> 63);
            }

            return compound_xint<2,Signed>{r_hi,r_lo};
#endif
        } else {
            auto value = static_cast<std::conditional_t<Signed,std::int64_t,std::uint64_t>>(a) * b;
            return {static_cast<full_uint>(value >> sizeof(full_int)*8),static_cast<full_uint>(value)};
        }
    }

    template<unsigned int N> auto umulnx1(const_compound_xint_ref<N> a,full_uint b) noexcept {
        if constexpr(N > 1) {
            return add(umulnx1<N-1>(a.lo(),b),shift<N-1>(mul1x1<false>(a.hi(),b)));
        } else {
            return mul1x1<false>(a.hi(),b);
        }
    }

    template<unsigned int N,unsigned int Shift> void offset_add(compound_uint<N> &r,const_compound_xint_ref<N-Shift> x) noexcept {
        r += detail::shift<Shift>(x);
        if constexpr(N-Shift > 1) {
            offset_add<N,Shift+1>(r,x.lo());
        }
    }

    template<unsigned int N> auto offset_mul_add(const_compound_xint_ref<N> a,const_compound_xint_ref<N> b) noexcept {
        compound_uint<1> hi{a.hi()*b[0]};
        if constexpr(N > 1) {
            return add(umulnx1<N-1>(a.lo(),b[0]),shift<N-1>(hi))
                + shift<1>(offset_mul_add<N-1>(a.lo(),const_compound_xint_ref<N-1>{b.data()+1}));
        } else {
            return hi;
        }
    }

    template<typename T> T sign(T x) noexcept {
        return static_cast<T>(static_cast<std::make_signed_t<T>>(x) >> (sizeof(T)*8 - 1));
    }
}

template<typename T,typename U> auto mul(const T &a,const U &b) noexcept {
    using namespace detail;

    if constexpr(small_xints<T> && small_xints<U>) {
        return static_cast<detail::bltin_sized_int<sizeof(T) + sizeof(U)>>(a) * b;
    } else {
        constexpr bool Signed = signed_type<T> || signed_type<U>;
        constexpr unsigned int Na = int_count<Signed,T>;
        constexpr unsigned int Nb = int_count<Signed,U>;
        if constexpr(Nb > Na) {
            return mul(b,a);
        } else if constexpr(Na == 1) {
            return mul1x1<Signed>(compi_hi<1,Signed,T>(a),compi_hi<1,Signed>(b));
        } else if constexpr(Nb == 1) {
            auto r = add(umulnx1<Na-1>(compi_lo<Na>(a),compi_hi<1>(b)),shift<Na-1>(mul1x1<Signed>(compi_hi<Na,Signed>(a),compi_hi<1,Signed>(b))));
            if constexpr(Signed) {
                offset_add<Na+1,1>(r,umulnx1<Na-1>(compi_lo<Na>(a),compi_hi<Nb+1>(b)));
                return compound_int<Na+1>{r};
            }
        } else {
            compound_uint<Na+Nb> a_tmp{a};
            compound_uint<Na+Nb> b_tmp{b};

            auto r = offset_mul_add<Na+Nb>(a_tmp,b_tmp);
            if constexpr(Signed) {
                return compound_int<Na+Nb>{r};
            } else {
                return r;
            }
        }
    }
}

template<typename Q,typename R=Q> struct quot_rem {
    Q quot;
    R rem;
};

namespace modulo_t {
    enum value_t {truncate_v,euclid_v};
    template<value_t Mod> using type = std::integral_constant<value_t,Mod>;
    inline constexpr type<truncate_v> truncate = {};
    inline constexpr type<euclid_v> euclid = {};
};

template<comp_or_single_i T,comp_or_single_i U,modulo_t::value_t Mod=modulo_t::truncate_v>
auto divmod(const T &a,const U &b,modulo_t::type<Mod> = {}) noexcept;

/* "unmul" is short for un-multiply. It is like a normal division function
except it assumes a/b fits inside compound_xint<Nr,Signed>. */
template<unsigned int Nr,typename T,typename U,modulo_t::value_t Mod=modulo_t::truncate_v,bool Signed = detail::signed_type<T> || detail::signed_type<U>>
auto unmul(const T &a,const U &b,modulo_t::type<Mod> = {}) noexcept {
    using namespace detail;

    constexpr unsigned int Na = detail::int_count<Signed,T>;
    constexpr unsigned int Nb = detail::int_count<Signed,U>;

    quot_rem<compound_xint<Nr,Signed>,compound_xint<Nb,Signed>> r;

#if _POLY_OPS_IMPL_HAVE_DIVX || _POLY_OPS_IMPL_GCC_X86_ASM
    if constexpr(Nr == 1 && Na == 2 && Nb == 1) {
#  if _POLY_OPS_IMPL_HAVE_DIVX
        if constexpr(Signed) {
            r.quot = _POLY_OPS_MSVC_DIV(
                compi_hi<2,true>(a),
                compi_hi<1,true>(a),
                compi_hi<1,true>(b),
                &static_cast<full_int&>(r.rem.hi()));
        } else {
            r.quot = _POLY_OPS_MSVC_UDIV(
                compi_hi<2>(a),
                compi_hi<1>(a),
                compi_hi<1>(b),
                &r.rem.hi());
        }
#  else
        if constexpr(Signed) {
            if constexpr(sizeof(full_int) == 8) {
                __asm__("idiv{q} %[c]" : [a]"=a"(r.quot.hi()), [b]"=d"(r.rem.hi()) : "[a]"(compi_hi<1>(a)), "[b]"(compi_hi<2>(a)), [c]"rm"(compi_hi<1>(b)));
            } else if constexpr(__builtin_constant_p(a) && __builtin_constant_p(b)) {
                std::int64_t a_tmp = static_cast<std::int64_t>(a);
                return {static_cast<full_int>(a_tmp / compi_hi<1,true>(b)),static_cast<full_int>(a_tmp % compi_hi<1,true>(b))};
            } else {
                __asm__("idiv{l} %[c]" : [a]"=a"(r.quot.hi()), [b]"=d"(r.rem.hi()) : "[a]"(compi_hi<1>(a)), "[b]"(compi_hi<2>(a)), [c]"rm"(compi_hi<1>(b)));
            }
        } else {
            if constexpr(sizeof(full_int) == 8) {
                __asm__("div{q} %[c]" : [a]"=a"(r.quot.hi()), [b]"=d"(r.rem.hi()) : "[a]"(compi_hi<1>(a)), "[b]"(compi_hi<2>(a)), [c]"rm"(compi_hi<1>(b)));
            } else if constexpr(__builtin_constant_p(a) && __builtin_constant_p(b)) {
                std::uint64_t a_tmp = static_cast<std::uint64_t>(a);
                return {static_cast<full_uint>(a_tmp / compi_hi<1>(b)),static_cast<full_uint>(a_tmp % compi_hi<1>(b))};
            } else {
                __asm__("div{l} %[c]" : [a]"=a"(r.quot.hi()), [b]"=d"(r.rem.hi()) : "[a]"(compi_hi<1>(a)), "[b]"(compi_hi<2>(a)), [c]"rm"(compi_hi<1>(b)));
            }
        }
#  endif
        if constexpr(Signed && Mod == modulo_t::euclid_v) {
            if((compi_hi<3>(a) ^ compi_hi<2>(b)) & compi_hi<1>(r.rem)) {
                r.quot -= 1;
                r.rem = abs(b) - abs(r.rem);
            }
        }
    } else
#endif
    {
        auto tmp = divmod(a,b,modulo_t::type<Mod>{});
        r.quot = compound_xint<Nr,Signed>{tmp.quot};
        r.rem = tmp.rem;
    }
    return r;
}

namespace detail {
#if _POLY_OPS_IMPL_HAVE_DIVX || _POLY_OPS_IMPL_GCC_X86_ASM
    template<unsigned int N>
    full_uint divmod_each_word_long(
        full_uint *_POLY_OPS_RESTRICT a,
        full_uint b,
        full_uint hi,
        full_uint *_POLY_OPS_RESTRICT quot) noexcept
    {
        auto qr = unmul<1>(compound_uint<2>{hi,a[N-1]},b);
        quot[N-1] = qr.quot[0];
        if constexpr(N > 1) {
            return divmod_each_word_long<N-1>(a,b,qr.rem[0],quot);
        } else {
            return qr.rem[0];
        }
    }
    template<unsigned int N> void udivmod_long(
        full_uint *_POLY_OPS_RESTRICT a,
        full_uint b,
        full_uint *_POLY_OPS_RESTRICT quot) noexcept
    {
        if constexpr(N > 1) {
            a[0] = divmod_each_word_long<N>(a,b,0,quot);
            for(unsigned int i=1; i<N; ++i) a[i] = 0;
        } else {
            *quot = *a / b;
            *a = *a % b;
        }
    }
    /*template<unsigned int Na,unsigned int Nb> void udivmod_long(
        compound_xint_base<Na> &_POLY_OPS_RESTRICT a,
        compound_xint_base<Nb> &_POLY_OPS_RESTRICT b,
        compound_xint_base<Na> &_POLY_OPS_RESTRICT quot) noexcept
    {
        if constexpr(N > 1) {
            if(a.hi() == 0) {
                if(b.hi() == 0) {
                    quot.hi() = 0;
                    udivmod_long(a.lo(),b.lo(),quot.lo(),rem.lo());
                }
            } else {
                divmod_each_word_long(a,b,quot);
            }
        } else {
            quot.hi() = a.hi() / b.hi();
            a.hi() = a.hi() % b.hi();
        }
    }*/
#endif
    template<unsigned int N> void udivmod_shift_sub(
        full_uint *_POLY_OPS_RESTRICT a,
        full_uint *_POLY_OPS_RESTRICT b,
        full_uint *_POLY_OPS_RESTRICT quot) noexcept
    {
        if constexpr(N > 1) {
            if(a[N-1] == 0 && b[N-1] == 0) {
                quot[N-1] = 0;
                udivmod_shift_sub<N-1>(a,b,quot);
            } else {
                compound_xint_ref<N> a_ref{a};
                compound_xint_ref<N> b_ref{b};
                int shift = countl_zero(b_ref) - countl_zero(a_ref);
                if(shift < 0) {
                    for(unsigned int i=0; i<N; ++i) quot[i] = 0;
                    return;
                }
                _shift_left<N>(b_ref,b_ref,static_cast<unsigned int>(shift));

                unsigned int n = N;
                while(n > 0) {
                    --n;

                    quot[n] = 0;
                    int end = static_cast<int>(n * sizeof(full_uint) * 8);
                    for(; shift >= end; --shift) {
                        quot[n] <<= 1;
                        if(cmp(a_ref,b_ref) >= 0) {
                            subborrow<N>(a_ref,b_ref);
                            quot[n] |= 1;
                        }
                        _shift_right<false,N>(b_ref,b_ref,1);
                    }
                }
            }
        } else {
            *quot = *a / *b;
            *a = *a % *b;
        }
    }
}

template<comp_or_single_i T,comp_or_single_i U,modulo_t::value_t Mod>
auto divmod(const T &a,const U &b,modulo_t::type<Mod>) noexcept {
    using namespace detail;

    constexpr bool signed_ = signed_type<T> || signed_type<U>;
    constexpr unsigned int N = max_int_count<signed_,T,U>;
    quot_rem<compound_xint<N,signed_>,compound_xint<int_count<signed_,U>,signed_>> r;

#if _POLY_OPS_IMPL_HAVE_DIVX || _POLY_OPS_IMPL_GCC_X86_ASM
    constexpr unsigned int Nb = int_count<signed_,U> == 1 ? 1 : N;
#else
    constexpr unsigned int Nb = N;
#endif

    if constexpr(N == 1) {
        auto a_tmp = compi_hi<1,signed_>(a);
        auto b_tmp = compi_hi<1,signed_>(b);
        r.quot = a_tmp / b_tmp;
        r.rem = a_tmp % b_tmp;

        if constexpr(signed_ && Mod == modulo_t::euclid_v) {
            if(compi_hi<1,signed_>(r.rem)) {
                r.rem = abs(r.rem);
                if((sign(a_tmp) ^ sign(b_tmp))) {
                    r.rem = abs(b_tmp) - r.rem;
                    r.quot -= 1;
                }
            }
        }
    } else if constexpr(signed_) {
        compound_int<N> a_tmp(a);
        compound_int<Nb> b_tmp(b);

        full_int sa = sign(static_cast<full_int>(a_tmp.hi()));
        full_int sb = sign(static_cast<full_int>(b_tmp.hi()));

        if(sa) a_tmp = -a_tmp;
        if(sb) b_tmp = -b_tmp;

#if _POLY_OPS_IMPL_HAVE_DIVX || _POLY_OPS_IMPL_GCC_X86_ASM
        if constexpr(Nb == 1) {
            udivmod_long<N>(a_tmp.data(),b_tmp.hi(),r.quot.data());
        } else
#endif
        {
            udivmod_shift_sub<N>(a_tmp.data(),b_tmp.data(),r.quot.data());
        }

        r.rem = compound_int<int_count<false,U>>(a_tmp);

        if constexpr(Mod == modulo_t::truncate_v) {
            if(sa ^ sb) r.quot = -r.quot;
            if(sa) r.rem = -r.rem;
        } else if(Mod == modulo_t::euclid_v) {
            if(sa ^ sb) {
                r.rem = abs(b) - r.rem;
                r.quot = -1 - r.quot;
            }
        }
    } else {
        compound_uint<N> a_tmp(a);
        compound_uint<Nb> b_tmp(b);

#if _POLY_OPS_IMPL_HAVE_DIVX || _POLY_OPS_IMPL_GCC_X86_ASM
        if constexpr(Nb == 1) {
            udivmod_long<N>(a_tmp.data(),b_tmp.hi(),r.quot.data());
        } else
#endif
        {
            udivmod_shift_sub<N>(a_tmp.data(),b_tmp.data(),r.quot.data());
        }

        r.rem = compound_uint<int_count<false,U>>(a_tmp);
    }
    return r;
}

template<unsigned int N,bool Signed,comp_or_single_i T> auto operator*(const compound_xint<N,Signed> &a,const T &b) noexcept {
    return mul(a,b);
}
template<detail::integral T,unsigned int N,bool Signed> auto operator*(const T &a,const compound_xint<N,Signed> &b) noexcept {
    return mul(a,b);
}

template<std::size_t Size> using sized_int = std::conditional_t<
    (Size <= sizeof(full_int)),
    detail::bltin_sized_int<Size>,
    compound_int<(Size + sizeof(full_int) - 1)/sizeof(full_int)>>;

template<detail::small_sints T,detail::small_sints U> auto mul(const T &a,const U &b) noexcept {
    return sized_int<std::max(sizeof(T),sizeof(U))>(a) * b;
}

template<unsigned int N,bool Signed,comp_or_single_i T> auto operator/(const compound_xint<N,Signed> &a,const T &b) noexcept {
    return divmod(a,b).quot;
}
template<detail::integral T,unsigned int N,bool Signed> auto operator/(const T &a,const compound_xint<N,Signed> &b) noexcept {
    return divmod(a,b).quot;
}

namespace detail {
    template<char C> struct digit_value {
        static_assert((C >= '0' && C <= '9') || (C >= 'a' && C <= 'f') || (C >= 'A' && C <= 'F'));
        static constexpr full_uint value = static_cast<full_uint>((C >= '0' && C <= '9') ? (C-'0') : ((C >= 'a' && C <= 'f') ? (C-'a'+10) : (C-'A'+10)));
    };

    template<char... C> struct char_pack {};
    template<char C1,char... C> struct char_pack<C1,C...> {};

    template<full_uint... V> struct full_uint_pack {};
    template<std::size_t X> using size_t_const = std::integral_constant<std::size_t,X>;

    template<full_uint Val1> consteval auto make_compound_int() {
        return compound_uint<1>{Val1};
    }

    template<full_uint Val1,full_uint Val2,full_uint... Vals>
    consteval auto make_compound_int() {
        return compound_uint<sizeof...(Vals)+2>{Val1,make_compound_int<Val2,Vals...>()};
    }

    template<std::size_t Nybble,full_uint CurVal,full_uint... Vals>
    consteval auto hex_value(char_pack<>,size_t_const<Nybble>,full_uint_pack<CurVal,Vals...>) {
        return compound_int<sizeof...(Vals)+1>{make_compound_int<CurVal,Vals...>()};
    }

    template<char C1,char... C,std::size_t Nybble,full_uint CurVal,full_uint... Vals>
    consteval auto hex_value(char_pack<C1,C...>,size_t_const<Nybble>,full_uint_pack<CurVal,Vals...>) {
        if constexpr(Nybble >= sizeof(full_uint)*2) {
            return hex_value(char_pack<C...>{},size_t_const<1>{},full_uint_pack<digit_value<C1>::value,CurVal,Vals...>{});
        } else if constexpr(sizeof...(C) == 0 && Nybble+1 == sizeof(full_uint)*2 && digit_value<C1>::value >= 8) {
            /* the literals are always positive but the type is signed, so if the
            highest bit of the highest component integer is one, another component
            is added */
            return hex_value(char_pack<>{},size_t_const<Nybble+1>{},full_uint_pack<0,CurVal | (digit_value<C1>::value << (Nybble*4)),Vals...>{});
        } else {
            return hex_value(char_pack<C...>{},size_t_const<Nybble+1>{},full_uint_pack<CurVal | (digit_value<C1>::value << (Nybble*4)),Vals...>{});
        }
    }

    template<char Ca1,char... Ca,char... Cb> consteval auto rev_char_pack(char_pack<Ca1,Ca...>,char_pack<Cb...>) {
        return rev_char_pack(char_pack<Ca...>{},char_pack<Ca1,Cb...>{});
    }
    template<char... Cb> consteval auto rev_char_pack(char_pack<>,char_pack<Cb...> b) {
        return b;
    }

    /*template<char C1,char... C> consteval auto skip_leading_0() {
        if constexpr(sizeof...(C) > 0 && C1 == '0') {
            return skip_leading_0<C...>();
        } else {
            return hex_value(rev_char_pack(char_pack<C1,C...>{},char_pack<>{}),size_t_const<0>{},full_uint_pack<0>{});
        }
    }*/

    template<char C1,char C2,char C3,char... C> consteval auto skip_0x() {
        static_assert(C1 == '0' && (C2 == 'x' || C2 == 'X'),"compound_int must start with '0x' or '0X'");
        return hex_value(rev_char_pack(char_pack<C3,C...>{},char_pack<>{}),size_t_const<0>{},full_uint_pack<0>{});
    }
} // namespace detail

/* Parse a hexidecimal value and return a compound_int instance with the
smallest size that can fit all the digits, including Leading zeros (after the
initial "0x"). */
template<char... C> constexpr auto operator""_compi() {
    static_assert(sizeof...(C) >= 3,"compound_int literal cannot have fewer than 3 characters");
    constexpr auto r = detail::skip_0x<C...>();
    return r;
}

}

#endif
