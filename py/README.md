polyops
==========================================================================

Boolean and inset/outset operations on polygons with integer coordinates.

**Notice:** This is a work in progress. Some features might not work and the
interface may change drastically between versions.

[Documentation](https://rouslan.github.io/polyops/py/index.html) |
[C++ Version](https://github.com/Rouslan/polyops)

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