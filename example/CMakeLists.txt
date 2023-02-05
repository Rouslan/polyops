cmake_minimum_required(VERSION 3.19)
project(poly_ops_examples)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(SDL2 REQUIRED)

if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()


add_executable(alphabet_orbit alphabet_orbit.cpp)
target_link_libraries(alphabet_orbit SDL2::SDL2)
