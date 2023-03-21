# Note: this makefile is not meant for users of the poly_ops library. It is for
# the convenience of the developer(s). Read py/README.rst for instructions on
# building and installing the Python module as intended.

cmake_minimum_required(VERSION 3.19)
project(poly_ops)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(src_pyx ${CMAKE_CURRENT_SOURCE_DIR}/py/poly_ops.pyx)
set(src_cpp ${CMAKE_CURRENT_BINARY_DIR}/poly_ops.cpp)

find_package(Python3 REQUIRED COMPONENTS Interpreter NumPy Development.Module)

add_custom_command(
    OUTPUT ${src_cpp}
    COMMAND Python3::Interpreter -m cython --cplus -o ${src_cpp} ${src_pyx}
    DEPENDS ${src_pyx}
    VERBATIM)

if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -pedantic -fvisibility=hidden)
endif()

Python3_add_library(poly_ops MODULE WITH_SOABI ${src_cpp})
target_link_libraries(poly_ops PRIVATE Python3::Module Python3::NumPy)

target_include_directories(poly_ops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(tests)
add_subdirectory(example)
add_subdirectory(doc)
