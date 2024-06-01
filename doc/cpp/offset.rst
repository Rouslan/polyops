poly_ops/offset.hpp
=====================


.. cpp:namespace:: poly_ops

Types
----------------

.. doxygenstruct:: point_and_origin
    :members:
    :undoc-members:

-----------------------------

.. doxygenclass:: origin_point_tracker


Functions
----------------

.. doxygenfunction:: add_offset_loops(clipper<Coord,Index>&,Input&&,bool_set,real_coord_t<Coord>,std::type_identity_t<Coord>,i_point_tracker<Index>*)

-----------------------------

.. doxygenfunction:: add_offset_loops(tclipper<Coord,Index,Tracker>&,Input&&,bool_set,real_coord_t<Coord>,std::type_identity_t<Coord>)

-----------------------------

.. doxygenfunction:: add_offset_loops_subject(clipper<Coord,Index>&,Input&&,real_coord_t<Coord>,std::type_identity_t<Coord>)

-----------------------------

.. doxygenfunction:: add_offset_loops_subject(tclipper<Coord,Index,Tracker>&,Input&&,real_coord_t<Coord>,std::type_identity_t<Coord>)

-----------------------------

.. doxygenfunction:: add_offset_loops_clip(clipper<Coord,Index>&,Input&&,real_coord_t<Coord>,std::type_identity_t<Coord>)

-----------------------------

.. doxygenfunction:: add_offset_loops_clip(tclipper<Coord,Index,Tracker>&,Input&&,real_coord_t<Coord>,std::type_identity_t<Coord>)

-----------------------------

.. doxygenfunction:: offset(Input&&,real_coord_t<Coord>,Coord,Tracker&&,std::pmr::memory_resource*)

-----------------------------

.. doxygenfunction:: offset(Input&&,real_coord_t<Coord>,Coord,std::pmr::memory_resource*)
