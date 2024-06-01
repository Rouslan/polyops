poly_ops/large_ints.hpp
=============================


.. cpp:namespace:: poly_ops

Concepts
------------------

.. doxygenconcept:: large_ints::comp_or_single_i

-----------------------------

.. doxygenconcept:: large_ints::safe_comp_or_single_i


Types
------------------

.. doxygenclass:: large_ints::compound_xint
    :members:
    :undoc-members:

-----------------------------

.. doxygentypedef:: large_ints::compound_int

-----------------------------

.. doxygentypedef:: large_ints::compound_uint

-----------------------------

.. doxygenclass:: large_ints::_compound_xint_ref
    :members:
    :undoc-members:

-----------------------------

.. doxygentypedef:: large_ints::compound_xint_ref

-----------------------------

.. doxygentypedef:: large_ints::const_compound_xint_ref

-----------------------------

.. doxygentypedef:: large_ints::sized_int

-----------------------------

.. doxygenstruct:: large_ints::quot_rem
    :members:
    :undoc-members:

-----------------------------

.. doxygenenum:: large_ints::modulo_t::value_t

-----------------------------

.. doxygentypedef:: large_ints::modulo_t::type


Constants
----------------

.. doxygenvariable:: large_ints::modulo_t::truncate

-----------------------------

.. doxygenvariable:: large_ints::modulo_t::euclid


Functions
----------------

.. doxygenfunction:: large_ints::add

-----------------------------

.. doxygenfunction:: large_ints::sub

-----------------------------

.. doxygenfunction:: large_ints::mul

-----------------------------

.. doxygenfunction:: large_ints::divmod

-----------------------------

.. doxygenfunction:: large_ints::unmul

-----------------------------

.. doxygenfunction:: large_ints::abs

-----------------------------

.. doxygenfunction:: large_ints::countl_zero

-----------------------------

.. doxygenfunction:: large_ints::shift_right

-----------------------------

.. doxygenfunction:: large_ints::shift_left

-----------------------------

.. doxygenfunction:: large_ints::eq

-----------------------------

.. doxygenfunction:: large_ints::cmp

-----------------------------

.. doxygenfunction:: large_ints::negative(const compound_xint<N,Signed>&)

-----------------------------

.. doxygenfunction:: large_ints::negative(T x)

-----------------------------

.. doxygenfunction:: large_ints::operator==(const compound_xint<N,Signed>&,const T&)

-----------------------------

.. doxygenfunction:: large_ints::operator==(const T&,const compound_xint<N,Signed>&)

-----------------------------

.. doxygenfunction:: large_ints::operator+(const compound_xint<N,Signed>&,const T&)

-----------------------------

.. doxygenfunction:: large_ints::operator+(const T&,const compound_xint<N,Signed>&)

-----------------------------

.. doxygenfunction:: large_ints::operator-(const compound_xint<N,Signed>&,const T&)

-----------------------------

.. doxygenfunction:: large_ints::operator-(const T&,const compound_xint<N,Signed>&)

-----------------------------

.. doxygenfunction:: large_ints::operator*(const compound_xint<N,Signed>&,const T&)

-----------------------------

.. doxygenfunction:: large_ints::operator*(const T&,const compound_xint<N,Signed>&)

-----------------------------

.. doxygenfunction:: large_ints::operator/(const compound_xint<N,Signed>&,const T&)

-----------------------------

.. doxygenfunction:: large_ints::operator/(const T&,const compound_xint<N,Signed>&)

-----------------------------

.. doxygenfunction:: large_ints::operator+=

-----------------------------

.. doxygenfunction:: large_ints::operator-=

-----------------------------

.. doxygenfunction:: large_ints::operator>>

-----------------------------

.. doxygenfunction:: large_ints::operator>>=

-----------------------------

.. doxygenfunction:: large_ints::operator<<

-----------------------------

.. doxygenfunction:: large_ints::operator<<=

-----------------------------

.. doxygenfunction:: large_ints::operator<=>

-----------------------------

.. doxygenfunction:: large_ints::operator""_compi
