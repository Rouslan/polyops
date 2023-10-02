polyops
================================

A header-only C++ library for boolean and inset/outset operations on polygons
with integer coordinates.

**Notice:** This is a work in progress. Some features might not work and the
interface may change drastically between versions. If you are looking for a
mature library, I recommend
[Clipper2](https://github.com/AngusJohnson/Clipper2).

[Documentation](https://rouslan.github.io/polyops/cpp/index.html) |
[Python Binding](https://github.com/Rouslan/polyops/tree/master/py)

When splitting lines in integer coordinates, the new points need to be rounded
to integer coordinates. This rounding causes the lines to be offset slightly,
which can cause lines to intersect where they previously did not intersect.
Unlike other polygon clipping libraries, polyops will catch 100% of these new
intersections, thus the output is guaranteed not to have any lines that cross
(lines may still touch at endpoints or overlap completely).

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