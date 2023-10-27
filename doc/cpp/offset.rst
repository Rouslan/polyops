poly_ops/offset.hpp
=====================


Types
----------------

.. doxygenstruct:: poly_ops::point_and_origin
    :members:
    :undoc-members:

-----------------------------

.. doxygenclass:: poly_ops::origin_point_tracker


Functions
----------------

.. doxygenfunction:: poly_ops::add_offset_loops(clipper<Coord,Index>&,Input&&,bool_set,real_coord_t<Coord>,std::type_identity_t<Coord>,i_point_tracker<Index>*)

-----------------------------

.. doxygenfunction:: poly_ops::add_offset_loops(tclipper<Coord,Index,Tracker>&,Input&&,bool_set,real_coord_t<Coord>,std::type_identity_t<Coord>)

-----------------------------

.. doxygenfunction:: poly_ops::add_offset_loops_subject(clipper<Coord,Index>&,Input&&,real_coord_t<Coord>,std::type_identity_t<Coord>)

-----------------------------

.. doxygenfunction:: poly_ops::add_offset_loops_subject(tclipper<Coord,Index,Tracker>&,Input&&,real_coord_t<Coord>,std::type_identity_t<Coord>)

-----------------------------

.. doxygenfunction:: poly_ops::add_offset_loops_clip(clipper<Coord,Index>&,Input&&,real_coord_t<Coord>,std::type_identity_t<Coord>)

-----------------------------

.. doxygenfunction:: poly_ops::add_offset_loops_clip(tclipper<Coord,Index,Tracker>&,Input&&,real_coord_t<Coord>,std::type_identity_t<Coord>)

-----------------------------

.. doxygenfunction:: poly_ops::offset(Input&&,real_coord_t<Coord>,Coord,Tracker&&,std::pmr::memory_resource*)

-----------------------------

.. doxygenfunction:: poly_ops::offset(Input&&,real_coord_t<Coord>,Coord,std::pmr::memory_resource*)
