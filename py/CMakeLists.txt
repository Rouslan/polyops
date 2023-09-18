# Note: this makefile is not the preferred way of building the poly_ops module.
# Read py/README.md for instructions on building and installing the module as
# intended.

cmake_minimum_required(VERSION 3.19)
project(poly_ops_py)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Python3 COMPONENTS Interpreter NumPy Development.Module)

function(run_cython cythonfile cppfile)
    add_custom_command(
        OUTPUT ${cppfile}
        COMMAND Python3::Interpreter -m cython --cplus -I${CMAKE_CURRENT_SOURCE_DIR} -o ${cppfile} ${cythonfile}
        DEPENDS ${cythonfile}
        VERBATIM)
endfunction()

function(cython_module modname cythonfile cppfile)
    run_cython(${cythonfile} ${cppfile})
    Python3_add_library(${modname} MODULE WITH_SOABI ${cppfile})
    target_link_libraries(${modname} PRIVATE Python3::Module Python3::NumPy)
    target_compile_definitions(${modname} PRIVATE NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION)
    target_include_directories(${modname} PRIVATE ${CMAKE_SOURCE_DIR}/include)
endfunction()

if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -pedantic -fvisibility=hidden)
endif()

if(Python3_FOUND)
    cython_module(poly_ops ${CMAKE_CURRENT_SOURCE_DIR}/poly_ops.pyx ${CMAKE_CURRENT_BINARY_DIR}/poly_ops.cpp)
    cython_module(polydraw ${CMAKE_CURRENT_SOURCE_DIR}/tests/polydraw.pyx ${CMAKE_CURRENT_BINARY_DIR}/polydraw.cpp)

    target_include_directories(polydraw PRIVATE ${CMAKE_SOURCE_DIR}/tests)
endif()
