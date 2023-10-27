poly_ops/large_ints.hpp
=============================

Concepts
------------------

.. doxygenconcept:: poly_ops::large_ints::comp_or_single_i

-----------------------------

.. doxygenconcept:: poly_ops::large_ints::safe_comp_or_single_i


Types
------------------

.. doxygenclass:: poly_ops::large_ints::compound_xint
    :members:
    :undoc-members:

-----------------------------

.. doxygentypedef:: poly_ops::large_ints::compound_int

-----------------------------

.. doxygentypedef:: poly_ops::large_ints::compound_uint

-----------------------------

.. doxygenclass:: poly_ops::large_ints::_compound_xint_ref
    :members:
    :undoc-members:

-----------------------------

.. doxygentypedef:: poly_ops::large_ints::compound_xint_ref

-----------------------------

.. doxygentypedef:: poly_ops::large_ints::const_compound_xint_ref

-----------------------------

.. doxygentypedef:: poly_ops::large_ints::sized_int

-----------------------------

.. doxygenstruct:: poly_ops::large_ints::quot_rem
    :members:
    :undoc-members:

-----------------------------

.. doxygenenum:: poly_ops::large_ints::modulo_t::value_t

-----------------------------

.. doxygentypedef:: poly_ops::large_ints::modulo_t::type


Constants
----------------

.. doxygenvariable:: poly_ops::large_ints::modulo_t::truncate

-----------------------------

.. doxygenvariable:: poly_ops::large_ints::modulo_t::euclid


Functions
----------------

.. doxygenfunction:: poly_ops::large_ints::add

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::sub

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::mul

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::divmod

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::unmul

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::abs

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::countl_zero

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::shift_right

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::shift_left

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::eq

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::cmp

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::negative(const compound_xint<N,Signed>&)

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::negative(T x)

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::operator==(const compound_xint<N,Signed>&,const T&)

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::operator==(const T&,const compound_xint<N,Signed>&)

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::operator+(const compound_xint<N,Signed>&,const T&)

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::operator+(const T&,const compound_xint<N,Signed>&)

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::operator-(const compound_xint<N,Signed>&,const T&)

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::operator-(const T&,const compound_xint<N,Signed>&)

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::operator*(const compound_xint<N,Signed>&,const T&)

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::operator*(const T&,const compound_xint<N,Signed>&)

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::operator/(const compound_xint<N,Signed>&,const T&)

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::operator/(const T&,const compound_xint<N,Signed>&)

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::operator+=

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::operator-=

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::operator>>

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::operator>>=

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::operator<<

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::operator<<=

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::operator<=>

-----------------------------

.. doxygenfunction:: poly_ops::large_ints::operator""_compi
