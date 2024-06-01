poly_ops/clip.hpp
=====================


.. cpp:namespace:: poly_ops

Types
------------------

.. doxygenclass:: i_point_tracker
    :members:
    :undoc-members:

-----------------------------

.. doxygenconcept:: point_tracker

-----------------------------

.. doxygenenum:: bool_op

-----------------------------

.. doxygenenum:: bool_set

-----------------------------

.. doxygenclass:: proto_loop_iterator

-----------------------------

.. doxygenclass:: temp_polygon_proxy
    :members:
    :undoc-members:

-----------------------------

.. doxygentypedef:: borrowed_temp_polygon_tree_range

-----------------------------

.. doxygentypedef:: borrowed_temp_polygon_range

-----------------------------

.. doxygentypedef:: temp_polygon_tree_range

-----------------------------

.. doxygentypedef:: temp_polygon_range

-----------------------------

.. doxygenclass:: clipper
    :members:
    :undoc-members:

-----------------------------

.. doxygenclass:: tclipper
    :members:
    :undoc-members:


Functions
----------------

.. doxygenfunction:: union_op(Input&&,std::pmr::memory_resource*)

-----------------------------

.. doxygenfunction:: union_op(Input&&,Tracker&&,std::pmr::memory_resource*)

-----------------------------

.. doxygenfunction:: normalize_op(Input&&,std::pmr::memory_resource*)

-----------------------------

.. doxygenfunction:: normalize_op(Input&&,Tracker&&,std::pmr::memory_resource*)

-----------------------------

.. doxygenfunction:: boolean_op(SInput&&,CInput&&,bool_op,std::pmr::memory_resource*)

-----------------------------

.. doxygenfunction:: boolean_op(SInput&&,CInput&&,bool_op,Tracker&&,std::pmr::memory_resource*)
