poly_ops/base.hpp
=================


.. cpp:namespace:: poly_ops

Concepts
-----------

.. doxygenconcept:: coordinate

-----------------------------

.. doxygenconcept:: point

-----------------------------

.. doxygenconcept:: point_range

-----------------------------

.. doxygenconcept:: point_range_range

-----------------------------

.. doxygenconcept:: point_range_or_range_range


Types
------------------

.. doxygentypedef:: long_coord_t

-----------------------------

.. doxygentypedef:: real_coord_t

-----------------------------

.. doxygenstruct:: point_ops

-----------------------------

.. doxygenstruct:: coord_ops
    :members:
    :undoc-members:

-----------------------------

.. doxygenstruct:: point_t
    :members:
    :undoc-members:

-----------------------------

.. doxygenclass:: winding_dir_sink
    :members:
    :undoc-members:


Functions
----------------

.. doxygenfunction:: operator+(const point_t<T>&,const point_t<U>&)

-----------------------------

.. doxygenfunction:: operator-(const point_t<T>&,const point_t<U>&)

-----------------------------

.. doxygenfunction:: operator*(const point_t<T>&,const point_t<U>&)

-----------------------------

.. doxygenfunction:: operator*(const point_t<T>&,U)

-----------------------------

.. doxygenfunction:: operator*(T,const point_t<U>&)

-----------------------------

.. doxygenfunction:: operator/(const point_t<T>&,const point_t<U>&)

-----------------------------

.. doxygenfunction:: operator==(const point_t<T>&,const point_t<T>&)

-----------------------------

.. doxygenfunction:: operator!=(const point_t<T>&,const point_t<T>&)

-----------------------------

.. doxygenfunction:: vdot

-----------------------------

.. doxygenfunction:: square

-----------------------------

.. doxygenfunction:: vcast

-----------------------------

.. doxygenfunction:: vround

-----------------------------

.. doxygenfunction:: vmag

-----------------------------

.. doxygenfunction:: vangle

-----------------------------

.. doxygenfunction:: triangle_winding

-----------------------------

.. doxygenfunction:: winding_dir
