poly_ops/large_ints.hpp
=============================

.. cpp:namespace:: poly_ops::large_ints


Concepts
------------------

.. cpp:concept:: template<typename T> comp_or_single_i

    Matches an instance of :cpp:class:`compound_xint` or a built-in integer.

.. cpp:concept:: template<typename T,unsigned int N,bool Signed> safe_comp_or_single_i

    Matches an instance of :cpp:class:`compound_xint` or a built-in integer than
    can cast to :cpp:class:`compound_xint\<N,Signed>` without loss of
    information.

Types
------------------

.. cpp:class:: template<unsigned int N,bool Signed> compound_xint

    .. cpp:type:: value_type = full_uint

    .. cpp:function:: compound_xint() noexcept = default

    .. cpp:function:: constexpr compound_xint(full_uint b) noexcept requires (N == 1)

    .. cpp:function:: template<comp_or_single_i T> requires (N > 1)\
        constexpr compound_xint(full_uint _hi,T _lo) noexcept
        :nocontentsentry:

    .. cpp:function:: template<comp_or_single_i T>\
        explicit(detail::int_count<Signed,std::decay_t<T>> > N || (detail::signed_type<T> && !Signed))\
        constexpr compound_xint(const T &b)
        :nocontentsentry:

    .. cpp:function:: constexpr compound_xint(const compound_xint&) noexcept = default
        :nocontentsentry:

    .. cpp:function:: constexpr compound_xint &operator=(const compound_xint&) noexcept = default

    .. cpp:function:: template<safe_comp_or_single_i<N,Signed> T>\
        compound_xint &operator=(const T &b) noexcept
        :nocontentsentry:

    .. cpp:function:: compound_xint operator-() const noexcept requires Signed

    .. cpp:function:: template<detail::integral T> explicit operator T() const noexcept

    .. cpp:function:: template<std::floating_point T> explicit operator T() const noexcept
        :nocontentsentry:

    .. cpp:function:: constexpr bool negative() const noexcept

    .. cpp:function:: explicit constexpr operator bool() const noexcept

    .. cpp:function:: constexpr operator compound_xint_ref<N>() noexcept

    .. cpp:function:: constexpr operator const_compound_xint_ref<N>() const noexcept

    .. cpp:function:: constexpr full_uint hi() const noexcept

    .. cpp:function:: constexpr full_uint &hi() noexcept
        :nocontentsentry:

    .. cpp:function:: constexpr const_compound_xint_ref<N-1> lo() const noexcept requires (N > 1)

    .. cpp:function:: constexpr compound_xint_ref<N-1> lo() noexcept requires (N > 1)

    .. cpp:function:: constexpr full_uint operator[](unsigned int i) const noexcept

    .. cpp:function:: constexpr full_uint &operator[](unsigned int i) noexcept
        :nocontentsentry:

    .. cpp:function:: std::size_t size() const

    .. cpp:function:: const full_uint *data() const noexcept

    .. cpp:function:: full_uint *data() noexcept
        :nocontentsentry:

    .. cpp:function:: const full_uint *begin() const noexcept

    .. cpp:function:: full_uint *begin() noexcept
        :nocontentsentry:

    .. cpp:function:: const full_uint *end() const noexcept

    .. cpp:function:: full_uint *end() noexcept
        :nocontentsentry:


.. cpp:type:: template<unsigned int N> compound_int = compound_xint<N,true>

.. cpp:type:: template<unsigned int N> compound_uint = compound_xint<N,false>


.. cpp:class:: template<unsigned int N,bool Const> _compound_xint_ref

    .. cpp:type:: value_type = std::conditional_t<Const,const full_uint,full_uint>;


    .. cpp:function:: explicit constexpr _compound_xint_ref(value_type *_data) noexcept

    .. cpp:function:: constexpr _compound_xint_ref(const _compound_xint_ref&) noexcept = default

    .. cpp:function:: constexpr _compound_xint_ref(const _compound_xint_ref<N,false> &b) noexcept requires Const

    .. cpp:function:: template<bool ConstB> requires (!Const)\
        constexpr const _compound_xint_ref &operator=(const _compound_xint_ref<N,ConstB> &b) const noexcept

    .. cpp:function:: template<typename T> requires (!Const)\
        constexpr void set(const T &b) const noexcept

    .. cpp:function:: constexpr value_type &hi() const noexcept

    .. cpp:function:: constexpr auto lo() const noexcept requires (N > 1)

    .. cpp:function:: constexpr value_type &operator[](unsigned int i) const noexcept

    .. cpp:function:: value_type *data() const noexcept

    .. cpp:function:: value_type *begin() const noexcept

    .. cpp:function:: value_type *end() const noexcept


.. cpp:type:: template<unsigned int N> compound_xint_ref = _compound_xint_ref<N,false>


.. cpp:type:: template<unsigned int N> const_compound_xint_ref = _compound_xint_ref<N,true>


.. cpp:type:: template<std::size_t Size> sized_int = std::conditional_t<\
    (Size <= sizeof(full_int)),\
    detail::bltin_sized_int<Size>,\
    compound_int<(Size + sizeof(full_int) - 1)/sizeof(full_int)>>


.. cpp:struct:: template<typename Q,typename R=Q> quot_rem

    .. cpp:member:: Q quot

    .. cpp:member:: R rem


.. cpp:namespace-push:: modulo_t

.. cpp:enum:: value_t
    
    .. cpp:enumerator:: truncate_v
    
    .. cpp:enumerator:: euclid_v

.. cpp:type:: template<value_t Mod> type = std::integral_constant<value_t,Mod>

.. cpp:namespace-pop::


Constants
----------------

.. cpp:namespace-push:: modulo_t

.. cpp:var:: inline constexpr type<truncate_v> truncate = {}

.. cpp:var:: inline constexpr type<euclid_v> euclid = {}

.. cpp:namespace-pop::


Functions
----------------

.. cpp:function:: template<typename T,typename U> auto add(const T &a,const U &b) noexcept

.. cpp:function:: template<typename T,typename U> auto sub(const T &a,const U &b) noexcept

.. cpp:function:: template<typename T,typename U> auto mul(const T &a,const U &b) noexcept

    Multiply `a` and `b` and return a type big enough to not overflow

.. cpp:function:: template<comp_or_single_i T,comp_or_single_i U,modulo_t::value_t Mod=modulo_t::truncate_v>\
    auto divmod(const T &a,const U &b,modulo_t::type<Mod> = {}) noexcept

.. cpp:function:: template<unsigned int Nr,typename T,typename U,modulo_t::value_t Mod=modulo_t::truncate_v,bool Signed = detail::signed_type<T> || detail::signed_type<U>>\
    auto unmul(const T &a,const U &b,modulo_t::type<Mod> = {}) noexcept

    "unmul" is short for un-multiply. It is like a normal division function
    except it assumes a/b fits inside ``compound_xint<Nr,Signed>``.

    Most of the time, this is just normal division followed by a cast, but on
    x86 platforms, when Nr is 1, T is equivalent to :cpp:class:`compound_xint`
    with a size of 2 and U is equivalent to :cpp:class:`compound_xint` with a
    size of 1, this operation only needs a single CPU instruction.

.. cpp:function:: template<comp_or_single_i T,comp_or_single_i U,modulo_t::value_t Mod>\
    auto divmod(const T &a,const U &b,modulo_t::type<Mod>) noexcept

.. cpp:function:: template<typename T> auto abs(const T &x) noexcept

.. cpp:function:: template<typename T> int countl_zero(const T &x) noexcept

.. cpp:function:: template<typename T> auto shift_right(const T &a,unsigned int amount) noexcept

.. cpp:function:: template<typename T> auto shift_left(const T &a,unsigned char amount) noexcept

.. cpp:function:: template<typename T,typename U,unsigned int N = detail::max_int_count<detail::signed_type<T> || detail::signed_type<U>,T,U>>\
    bool eq(const T &a,const U &b) noexcept

.. cpp:function:: template<typename T,typename U> std::strong_ordering cmp(const T &a,const U &b) noexcept

.. cpp:function:: template<unsigned int N,bool Signed> bool negative(const compound_xint<N,Signed> &x) noexcept

.. cpp:function:: template<detail::integral T> bool negative(T x) noexcept
    :nocontentsentry:

.. cpp:function:: template<unsigned int N,bool Signed,comp_or_single_i T> bool operator==(const compound_xint<N,Signed> &a,const T &b) noexcept

.. cpp:function:: template<detail::integral T,unsigned int N,bool Signed> bool operator==(const T &a,const compound_xint<N,Signed> &b) noexcept
    :nocontentsentry:

.. cpp:function:: template<unsigned int N,bool Signed,comp_or_single_i T> auto operator+(const compound_xint<N,Signed> &a,const T &b) noexcept

.. cpp:function:: template<detail::integral T,unsigned int N,bool Signed> auto operator+(const T &a,const compound_xint<N,Signed> &b) noexcept
    :nocontentsentry:

.. cpp:function:: template<unsigned int N,bool Signed,comp_or_single_i T> auto operator-(const compound_xint<N,Signed> &a,const T &b) noexcept

.. cpp:function:: template<detail::integral T,unsigned int N,bool Signed> auto operator-(const T &a,const compound_xint<N,Signed> &b) noexcept
    :nocontentsentry:

.. cpp:function:: template<unsigned int N,bool Signed,comp_or_single_i T> auto operator*(const compound_xint<N,Signed> &a,const T &b) noexcept

.. cpp:function:: template<detail::integral T,unsigned int N,bool Signed> auto operator*(const T &a,const compound_xint<N,Signed> &b) noexcept
    :nocontentsentry:

.. cpp:function:: template<unsigned int N,bool Signed,comp_or_single_i T> auto operator/(const compound_xint<N,Signed> &a,const T &b) noexcept

.. cpp:function:: template<detail::integral T,unsigned int N,bool Signed> auto operator/(const T &a,const compound_xint<N,Signed> &b) noexcept
    :nocontentsentry:

.. cpp:function:: template<unsigned int N,bool Signed,safe_comp_or_single_i<N,Signed> T> compound_xint<N,Signed> &operator+=(compound_xint<N,Signed> &a,const T &b) noexcept

.. cpp:function:: template<unsigned int N,bool Signed,safe_comp_or_single_i<N,Signed> T> compound_xint<N,Signed> &operator-=(compound_xint<N,Signed> &a,const T &b) noexcept

.. cpp:function:: template<unsigned int N,bool Signed> compound_xint<N,Signed> operator>>(const compound_xint<N,Signed> &a,unsigned char amount) noexcept

.. cpp:function:: template<unsigned int N,bool Signed> compound_xint<N,Signed> &operator>>=(compound_xint<N,Signed> &a,unsigned char amount) noexcept

.. cpp:function:: template<unsigned int N,bool Signed> compound_xint<N,Signed> operator<<(const compound_xint<N,Signed> &a,unsigned char amount) noexcept

.. cpp:function:: template<unsigned int N,bool Signed> compound_xint<N,Signed> &operator<<=(compound_xint<N,Signed> &a,unsigned char amount) noexcept

.. cpp:function:: template<unsigned int N,bool Signed,comp_or_single_i T> auto operator<=>(const compound_xint<N,Signed> &a,const T &b) noexcept

.. cpp:function:: template<char... C> constexpr auto operator""_compi()

    Parse a hexadecimal value and return a compound_int instance with the
    smallest size that can fit all the digits, including Leading zeros (after
    the initial "0x")
