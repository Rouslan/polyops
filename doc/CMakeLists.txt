cmake_minimum_required(VERSION 3.19)
project(poly_ops_doc)

set(SPHINX_OUTPUT_TYPE "html" CACHE STRING "Sphinx builder type")

set(ENV{BUILDDIR} ${CMAKE_CURRENT_BINARY_DIR})

find_package(Python3 COMPONENTS Interpreter)
find_package(Doxygen)

if(Python3_FOUND AND DOXYGEN_FOUND)
    configure_file(Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile NO_SOURCE_PERMISSIONS)
    add_custom_target(
        documentation
        COMMAND Doxygen::doxygen ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        COMMAND Python3::Interpreter
            -m sphinx
            -b ${SPHINX_OUTPUT_TYPE}
            -D breathe_projects.polyops=${CMAKE_CURRENT_BINARY_DIR}/xml
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}/root
        DEPENDS
            conf.py
            index.rst
            cpp/base.rst
            cpp/clip.rst
            cpp/large_ints.rst
            cpp/index.rst
            cpp/offset.rst
            py/index.rst
            py/polyops.rst
            ${CMAKE_SOURCE_DIR}/include/poly_ops/base.hpp
            ${CMAKE_SOURCE_DIR}/include/poly_ops/clip.hpp
            ${CMAKE_SOURCE_DIR}/include/poly_ops/offset.hpp
            ${CMAKE_SOURCE_DIR}/include/poly_ops/large_ints.hpp
        VERBATIM)
endif()
