polyops
==========================================================================

Boolean and inset/outset operations on polygons with integer coordinates.

**Notice:** This is a work in progress. Some features might not work and the
interface may change drastically between versions.

[Documentation](https://rouslan.github.io/polyops/py/index.html) |
[C++ Version](https://github.com/Rouslan/polyops)

When splitting lines in integer coordinates, the new points need to be rounded
to integer coordinates. This rounding causes the lines to be offset slightly,
which can cause lines to intersect where they previously did not intersect.
Unlike other polygon clipping libraries, polyops will catch 100% of these new
intersections, thus the output is guaranteed not to have any lines that cross
(lines may still touch at endpoints or overlap completely).

Basic usage:
```python
loop_a = [
    [58,42],
    [36,52],
    [20,34],
    [32,13],
    [55,18]]
loop_b = [
    [76,58],
    [43,56],
    [45,23],
    [78,26]]

output = poly_ops.boolean_op_flat([loop_a],[loop_b],poly_ops.difference)
print(output)
```