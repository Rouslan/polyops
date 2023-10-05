Python API
===========


Basic usage:

.. code-block:: python

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

Using the :py:class:`poly_ops.Clipper` class:

.. code-block:: python

    clip = poly_ops.Clipper()
    clip.add_loop_subject(loop_a)
    clip.add_loop_clip(loop_b)

    output2 = clip.execute_flat(poly_ops.xor)

Getting the results as a hierarchy instead of a flat list:

.. code-block:: python

    def print_polygon(poly,level=0):
        indent = '  '*level
        for p in poly[0]:
            print(f"{indent}{p[0]},{p[1]}")

        print()

        if poly[1]:
            print(f"{indent}nested polygons:")
            for child in poly[1]: print_polygon(child,level+1);

    ...

    output = poly_ops.boolean_op_tree([loop_a],[loop_b],poly_ops.normalize)
    for loop in output: print_polygon(loop)


Installing:

`pip install --user polyops @ git+https://github.com/Rouslan/polyops`

Reference:

.. toctree::
    :maxdepth: 4

    polyops
