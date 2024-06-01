cmake_minimum_required(VERSION 3.19)
project(poly_ops_tests)

enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(THREADS_PREFER_PTHREAD_FLAG True)


find_package(Boost 1.69.0)
find_package(Threads)


if (MSVC)
    add_compile_options(/W4)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -Wundef -Wconversion -Wsign-conversion -pedantic -fvisibility=hidden)
endif()

if(Boost_FOUND)
    add_library(websocket_server SHARED server.cpp)
    target_link_libraries(websocket_server PRIVATE Boost::boost PUBLIC Threads::Threads)

    add_executable(random_shapes random_shapes.cpp)
    target_link_libraries(random_shapes websocket_server)
endif()

find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(GMPXX IMPORTED_TARGET gmpxx>=6.2.0)
endif()

function(def_test_large_ints exe_name)
    add_executable(${exe_name} test_large_ints.cpp)
    if(GMPXX_FOUND)
        target_compile_definitions(${exe_name} PRIVATE HAVE_GMP_LIBRARY)
        target_link_libraries(${exe_name} PkgConfig::GMPXX)
    endif()
endfunction()

def_test_large_ints(test_large_ints_ext)
def_test_large_ints(test_large_ints_noext)
target_compile_definitions(test_large_ints_noext PRIVATE POLY_OPS_NO_COMPILER_EXTENSIONS)
add_test(NAME TestLargeInts COMMAND test_large_ints_ext)
add_test(NAME TestLargeIntsNoExt COMMAND test_large_ints_noext)

add_executable(btest btest.cpp bitmap.cpp)
set(test_input_file ${CMAKE_CURRENT_SOURCE_DIR}/test_data/input.txt)
add_test(NAME UnionBitmapTest COMMAND btest ${test_input_file} union)
add_test(NAME IntersectionBitmapTest COMMAND btest ${test_input_file} intersection)
add_test(NAME XorBitmapTest COMMAND btest ${test_input_file} xor)
add_test(NAME DifferenceBitmapTest COMMAND btest ${test_input_file} difference)
add_test(NAME NormalizeBitmapTest COMMAND btest ${test_input_file} normalize)
add_test(NAME OffsetBitmapTest COMMAND btest ${test_input_file} offset)

add_executable(whitebox white_box.cpp)
add_test(NAME WhiteBox COMMAND whitebox -t2000 -n500 -p10)

#add_executable(misc_tests misc_tests.cpp)

add_executable(gen_discontinuous gen_discontinuous.cpp bitmap.cpp)

add_custom_target(test_bin)
add_dependencies(test_bin btest whitebox test_large_ints_ext test_large_ints_noext)

add_executable(polydraw_debug_log EXCLUDE_FROM_ALL polydraw_debug_log.cpp)
