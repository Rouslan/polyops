cmake_minimum_required(VERSION 3.19)
project(poly_ops_py LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(SKBUILD)
    set(python_required_str REQUIRED)
endif()

find_package(Python3 ${python_required_str} COMPONENTS Interpreter NumPy Development.Module)

function(run_cython cythonfile cppfile modname hfile)
    add_custom_command(
        OUTPUT ${cppfile} ${hfile}
        COMMAND Python3::Interpreter -m cython --cplus -I${CMAKE_CURRENT_SOURCE_DIR} -o ${cppfile} --module-name ${modname} ${cythonfile}
        DEPENDS ${cythonfile} ${CMAKE_SOURCE_DIR}/py/common.pxd
        VERBATIM)
endfunction()

function(cython_module modname fullname cythonfile cppfile hfile)
    run_cython(${cythonfile} ${cppfile} ${fullname} "${hfile}")

    Python3_add_library(${modname} MODULE WITH_SOABI ${cppfile})
    target_link_libraries(${modname} PRIVATE Python3::Module Python3::NumPy)
    target_compile_definitions(${modname} PRIVATE
        NPY_NO_DEPRECATED_API
        NPY_1_7_API_VERSION
        CYTHON_USE_PYLONG_INTERNALS=0 # don't bloat the binaries with this unneeded optimization
        CYTHON_EXTERN_C= # don't actually mark anything "extern"
    )
    target_include_directories(${modname} PRIVATE ${CMAKE_SOURCE_DIR}/include)
endfunction()

if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -pedantic -fvisibility=hidden)
endif()

if(Python3_FOUND)
    cython_module(
        _polyops
        polyops._polyops
        ${CMAKE_CURRENT_SOURCE_DIR}/_polyops.pyx
        ${CMAKE_CURRENT_BINARY_DIR}/_polyops.cpp

        # A header file is generated but not actually used. A few functions are
        # declared "public" so that their names are not mangled by Cython, and
        # can be used directly in C++ code.
        ${CMAKE_CURRENT_BINARY_DIR}/_polyops.h)
    cython_module(
        polydraw polyops.tests.polydraw
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/polydraw.pyx
        ${CMAKE_CURRENT_BINARY_DIR}/polydraw.cpp
        "")

    if(SKBUILD)
        install(TARGETS _polyops LIBRARY DESTINATION polyops)
        install(FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pyi
            ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
            ${CMAKE_CURRENT_SOURCE_DIR}/py.typed
            DESTINATION polyops)

        install(TARGETS polydraw LIBRARY DESTINATION polyops/tests)
        install(FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bitmap.py
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_misc.py
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/__main__.py
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/__init__.py
            DESTINATION polyops/tests)

        install(DIRECTORY ${CMAKE_SOURCE_DIR}/tests/test_data DESTINATION polyops/tests)
    endif()
endif()
