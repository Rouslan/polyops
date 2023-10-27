poly_ops/clip.hpp
=====================


Types
------------------

.. doxygenclass:: poly_ops::i_point_tracker
    :members:
    :undoc-members:

-----------------------------

.. doxygenconcept:: poly_ops::point_tracker

-----------------------------

.. doxygenenum:: poly_ops::bool_op

-----------------------------

.. doxygenenum:: poly_ops::bool_set

-----------------------------

.. doxygenclass:: poly_ops::proto_loop_iterator

-----------------------------

.. doxygenclass:: poly_ops::temp_polygon_proxy
    :members:
    :undoc-members:

-----------------------------

.. doxygentypedef:: poly_ops::borrowed_temp_polygon_tree_range

-----------------------------

.. doxygentypedef:: poly_ops::borrowed_temp_polygon_range

-----------------------------

.. doxygentypedef:: poly_ops::temp_polygon_tree_range

-----------------------------

.. doxygentypedef:: poly_ops::temp_polygon_range

-----------------------------

.. doxygenclass:: poly_ops::clipper
    :members:
    :undoc-members:

-----------------------------

.. doxygenclass:: poly_ops::tclipper
    :members:
    :undoc-members:


Functions
----------------

.. doxygenfunction:: poly_ops::union_op(Input&&,std::pmr::memory_resource*)

-----------------------------

.. doxygenfunction:: poly_ops::union_op(Input&&,Tracker&&,std::pmr::memory_resource*)

-----------------------------

.. doxygenfunction:: poly_ops::normalize_op(Input&&,std::pmr::memory_resource*)

-----------------------------

.. doxygenfunction:: poly_ops::normalize_op(Input&&,Tracker&&,std::pmr::memory_resource*)

-----------------------------

.. doxygenfunction:: poly_ops::boolean_op(SInput&&,CInput&&,bool_op,std::pmr::memory_resource*)

-----------------------------

.. doxygenfunction:: poly_ops::boolean_op(SInput&&,CInput&&,bool_op,Tracker&&,std::pmr::memory_resource*)
