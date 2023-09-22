cmake_minimum_required(VERSION 3.19)
project(poly_ops LANGUAGES CXX)

if(NOT DEFINED SKBUILD)
    include(CTest)
    enable_testing()

    add_subdirectory(tests)
    add_subdirectory(example)
    add_subdirectory(doc)
endif()

add_subdirectory(py)

