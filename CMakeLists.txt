# Note: this makefile is not meant for users of the poly_ops library. It is for
# the convenience of the developer(s). Read py/README.rst for instructions on
# building and installing the Python module as intended.

cmake_minimum_required(VERSION 3.19)
project(poly_ops)

add_subdirectory(tests)
add_subdirectory(example)
add_subdirectory(py)
add_subdirectory(doc)
