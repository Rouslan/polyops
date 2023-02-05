cmake_minimum_required(VERSION 3.19)
project(poly_ops_tests)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(THREADS_PREFER_PTHREAD_FLAG True)

find_package(Boost 1.69.0 REQUIRED COMPONENTS unit_test_framework)
find_package(Threads)

if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -pedantic -fvisibility=hidden)
endif()

add_library(websocket_server SHARED server.cpp)
target_link_libraries(websocket_server PRIVATE Boost::boost PUBLIC Threads::Threads)

add_executable(random_shapes random_shapes.cpp)
target_link_libraries(random_shapes websocket_server)


add_executable(unit_tests unit_tests.cpp)
target_link_libraries(unit_tests Boost::unit_test_framework)
target_compile_definitions(unit_tests PRIVATE BOOST_TEST_DYN_LINK)
