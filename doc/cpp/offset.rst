poly_ops/offset.hpp
=====================

.. cpp:namespace:: poly_ops


Functions
----------------

.. cpp:function:: template<typename Coord,typename Index=std::size_t,typename Input>\
    void add_offset_loops(\
        clipper<Coord,Index> &n,\
        Input &&input,\
        bool_set set,\
        real_coord_t<Coord> magnitude,\
        std::type_identity_t<Coord> arc_step_size)

.. cpp:function:: template<typename Coord,typename Index=std::size_t,typename Input>\
    void add_offset_loops_subject(\
        clipper<Coord,Index> &n,\
        Input &&input,\
        real_coord_t<Coord> magnitude,\
        std::type_identity_t<Coord> arc_step_size)

.. cpp:function:: template<typename Coord,typename Index=std::size_t,typename Input>\
    void add_offset_loops_clip(\
        clipper<Coord,Index> &n,\
        Input &&input,\
        real_coord_t<Coord> magnitude,\
        std::type_identity_t<Coord> arc_step_size)

.. cpp:function:: template<bool TreeOut,typename Coord,typename Index=std::size_t,typename Input>\
    std::conditional_t<TreeOut,\
        temp_polygon_tree_range<Coord,Index>,\
        temp_polygon_range<Coord,Index>>\
    offset(\
        Input &&input,\
        real_coord_t<Coord> magnitude,\
        Coord arc_step_size,\
        point_tracker<Index> *pt=nullptr,\
        std::pmr::memory_resource *contig_mem=nullptr)