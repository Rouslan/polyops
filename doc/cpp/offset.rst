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

.. doxygenfunction:: add_offset(clipper<Coord,Index>&,Input&&,bool_set,real_coord_t<Coord>,std::type_identity_t<Coord>,bool,i_point_tracker<Index>*)

-----------------------------

.. doxygenfunction:: add_offset(tclipper<Coord,Index,Tracker>&,Input&&,bool_set,real_coord_t<Coord>,bool,std::type_identity_t<Coord>)

-----------------------------

.. doxygenfunction:: add_offset_subject(clipper<Coord,Index>&,Input&&,real_coord_t<Coord>,std::type_identity_t<Coord>,bool)

-----------------------------

.. doxygenfunction:: add_offset_subject(tclipper<Coord,Index,Tracker>&,Input&&,real_coord_t<Coord>,std::type_identity_t<Coord>,bool)

-----------------------------

.. doxygenfunction:: add_offset_clip(clipper<Coord,Index>&,Input&&,real_coord_t<Coord>,std::type_identity_t<Coord>,bool)

-----------------------------

.. doxygenfunction:: add_offset_clip(tclipper<Coord,Index,Tracker>&,Input&&,real_coord_t<Coord>,std::type_identity_t<Coord>,bool)

-----------------------------

.. doxygenfunction:: offset(Input&&,real_coord_t<Coord>,Coord,bool,Tracker&&,std::pmr::memory_resource*)

-----------------------------

.. doxygenfunction:: offset(Input&&,real_coord_t<Coord>,Coord,bool,std::pmr::memory_resource*)
