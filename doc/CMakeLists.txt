cmake_minimum_required(VERSION 3.19)
project(poly_ops_doc)

set(SPHINX_OUTPUT_TYPE "html" CACHE STRING "Sphinx builder type")

set(ENV{BUILDDIR} ${CMAKE_CURRENT_BINARY_DIR})

find_package(Python3 COMPONENTS Interpreter)

if(Python3_FOUND)
    add_custom_target(
        documentation
        COMMAND Python3::Interpreter -m sphinx -b ${SPHINX_OUTPUT_TYPE} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/root
        DEPENDS
            conf.py
            index.rst
            cpp/base.rst
            cpp/clip.rst
            cpp/large_ints.rst
            cpp/index.rst
            py/index.rst
            py/polyops.rst
        VERBATIM)
endif()
