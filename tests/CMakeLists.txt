cmake_minimum_required(VERSION 3.19)
project(poly_ops_tests)

enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(THREADS_PREFER_PTHREAD_FLAG True)


#function(def_test_large_ints exe_name)
#    add_executable(${exe_name} test_large_ints.cpp)
#    target_link_libraries(${exe_name} PRIVATE Catch2::Catch2WithMain)
#    target_compile_options(${exe_name} PRIVATE ${GMP_CFLAGS})
#    target_link_options(${exe_name} PRIVATE ${GMP_LDFLAGS})
#endfunction()


find_package(Boost 1.69.0 COMPONENTS unit_test_framework)
find_package(Threads)


if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wundef -Wconversion -Wsign-conversion -pedantic -fvisibility=hidden)
endif()

if(Boost_FOUND)
    add_library(websocket_server SHARED server.cpp)
    target_link_libraries(websocket_server PRIVATE Boost::boost PUBLIC Threads::Threads)

    add_executable(random_shapes random_shapes.cpp)
    target_link_libraries(random_shapes websocket_server)


    add_executable(unit_tests unit_tests.cpp)
    target_link_libraries(unit_tests Boost::unit_test_framework)
    target_compile_definitions(unit_tests PRIVATE BOOST_TEST_DYN_LINK)
endif()

#find_package(PkgConfig)

#if(PKG_CONFIG_FOUND)
#    pkg_check_modules(GMP gmpxx>=6.2.0)
#    if(GMP_FOUND)
#        def_test_large_ints(test_large_ints_ext)
#        def_test_large_ints(test_large_ints_noext)
#        target_compile_definitions(test_large_ints_noext PRIVATE POLY_OPS_NO_COMPILER_EXTENSIONS)
#    endif()
#endif()

add_executable(simple simple.cpp)

add_executable(btest btest.cpp bitmap.cpp)
set(test_input_file ${CMAKE_CURRENT_SOURCE_DIR}/test_data/input.txt)
add_test(NAME UnionBitmapTest COMMAND btest ${test_input_file} union)
add_test(NAME IntersectionBitmapTest COMMAND btest ${test_input_file} intersection)
add_test(NAME XorBitmapTest COMMAND btest ${test_input_file} xor)
add_test(NAME DifferenceBitmapTest COMMAND btest ${test_input_file} difference)
add_test(NAME NormalizeBitmapTest COMMAND btest ${test_input_file} normalize)

add_executable(whitebox white_box.cpp)
add_test(NAME WhiteBox COMMAND whitebox -t2000 -n50000 -p50)

add_executable(gen_discontinuous gen_discontinuous.cpp bitmap.cpp)
