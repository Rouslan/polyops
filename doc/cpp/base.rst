poly_ops/base.hpp
=================


Concepts
-----------

.. doxygenconcept:: poly_ops::coordinate

-----------------------------

.. doxygenconcept:: poly_ops::point

-----------------------------

.. doxygenconcept:: poly_ops::point_range

-----------------------------

.. doxygenconcept:: poly_ops::point_range_range

-----------------------------

.. doxygenconcept:: poly_ops::point_range_or_range_range


Types
------------------

.. doxygentypedef:: poly_ops::long_coord_t

-----------------------------

.. doxygentypedef:: poly_ops::real_coord_t

-----------------------------

.. doxygenstruct:: poly_ops::point_ops

-----------------------------

.. doxygenstruct:: poly_ops::coord_ops
    :members:
    :undoc-members:

-----------------------------

.. doxygenstruct:: poly_ops::point_t
    :members:
    :undoc-members:

-----------------------------

.. doxygenclass:: poly_ops::winding_dir_sink
    :members:
    :undoc-members:


Functions
----------------

.. doxygenfunction:: poly_ops::operator+(const point_t<T>&,const point_t<U>&)

-----------------------------

.. doxygenfunction:: poly_ops::operator-(const point_t<T>&,const point_t<U>&)

-----------------------------

.. doxygenfunction:: poly_ops::operator*(const point_t<T>&,const point_t<U>&)

-----------------------------

.. doxygenfunction:: poly_ops::operator*(const point_t<T>&,U)

-----------------------------

.. doxygenfunction:: poly_ops::operator*(T,const point_t<U>&)

-----------------------------

.. doxygenfunction:: poly_ops::operator/(const point_t<T>&,const point_t<U>&)

-----------------------------

.. doxygenfunction:: poly_ops::operator==(const point_t<T>&,const point_t<T>&)

-----------------------------

.. doxygenfunction:: poly_ops::operator!=(const point_t<T>&,const point_t<T>&)

-----------------------------

.. doxygenfunction:: poly_ops::vdot

-----------------------------

.. doxygenfunction:: poly_ops::square

-----------------------------

.. doxygenfunction:: poly_ops::vcast

-----------------------------

.. doxygenfunction:: poly_ops::vround

-----------------------------

.. doxygenfunction:: poly_ops::vmag

-----------------------------

.. doxygenfunction:: poly_ops::vangle

-----------------------------

.. doxygenfunction:: poly_ops::triangle_winding

-----------------------------

.. doxygenfunction:: poly_ops::winding_dir
