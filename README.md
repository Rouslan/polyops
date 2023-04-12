polyops
================================

A header-only C++ library for boolean and inset/outset operations on polygons
with integer coordinates.

**Notice:** This is a work in progress. Some features might not work and the
interface may change drastically between versions. If you are looking for a
mature library, I recommend
[Clipper2](https://github.com/AngusJohnson/Clipper2).

Basic usage:
```c++
using namespace poly_ops;

std::vector<point_t<int>> loop_a{
    {58,42},
    {36,52},
    {20,34},
    {32,13},
    {55,18}};
std::vector<point_t<int>> loop_b{
    {76,58},
    {43,56},
    {45,23},
    {78,26}};

auto output = boolean_op<false,int>(loop_a,loop_b,bool_op::difference);
for(auto loop : output) {
    for(auto p : loop) {
        std::cout << p[0] << ',' << p[1] << '\n';
    }
    std::cout << '\n';
}
```