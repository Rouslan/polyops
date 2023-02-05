#cython: language_level=3, boundscheck=False, wraparound=False
#distutils: language = c++

import numpy as np
cimport numpy as np
cimport cython
cimport cpython.tuple
from libc.stdint cimport int32_t
from libcpp.vector cimport vector


np.import_array()

cdef extern from *:
    """
    #include <cstddef>
    #include <cstdint>
    #include <cassert>
    #include <type_traits>

    #include <poly_ops/poly_ops.hpp>

    using temp_poly_tree_range = poly_ops::temp_polygon_tree_range<std::uint32_t,std::int32_t>;
    using temp_poly_range = poly_ops::temp_polygon_range<std::uint32_t,std::int32_t>;
    using temp_poly_proxy = poly_ops::temp_polygon_proxy<std::uint32_t,std::int32_t>;
    using temp_poly_proxy_child_range = decltype(std::declval<temp_poly_proxy>().inner_loops());

    struct loop_entry {
        const char *data;
        std::size_t major_size;
        std::ptrdiff_t minor_stride;
        std::ptrdiff_t major_stride;
    };

    constexpr auto to_range = [](const loop_entry &e) {
        return poly_ops::blob_to_point_range<std::int32_t>(
            e.data,e.minor_stride,e.major_stride,e.major_size);
    };

    auto loop_vector_to_ranges(const std::vector<loop_entry> &loops) {
        return loops | std::views::transform(to_range);
    }

    PyObject *poly_proxy_to_py(const temp_poly_proxy &x) noexcept {
        npy_intp dims[2] = {static_cast<npy_intp>(x.size()),2};
        PyObject *r = PyArray_SimpleNew(2,dims,NPY_INT32);
        if(!r) return r;

        assert(PyArray_IS_C_CONTIGUOUS(r));

        auto *array = reinterpret_cast<std::int32_t*>(PyArray_DATA(r));
        for(const auto &p : x) {
            *array++ = static_cast<std::int32_t>(p[0]);
            *array++ = static_cast<std::int32_t>(p[1]);
        }

        return r;
    }

    template<typename R,typename ChildF>
    PyObject *proxy_range_to_py(const R &x,ChildF childf) noexcept;

    PyObject *poly_proxy_to_py_tree(const temp_poly_proxy &x) noexcept {
        PyObject *r = PyTuple_New(2);
        if(!r) return r;
        PyObject *tmp = poly_proxy_to_py(x);
        if(!tmp) {
            Py_DECREF(r);
            return tmp;
        }
        PyTuple_SET_ITEM(r,0,tmp);

        tmp = proxy_range_to_py(x.inner_loops(),&poly_proxy_to_py_tree);
        if(!tmp) {
            Py_DECREF(r);
            return tmp;
        }
        PyTuple_SET_ITEM(r,1,tmp);

        return r;
    }

    template<typename R,typename ChildF>
    PyObject *proxy_range_to_py(const R &x,ChildF childf) noexcept {
        PyObject *r = PyTuple_New(static_cast<Py_ssize_t>(x.size()));
        if(!r) return r;

        Py_ssize_t ti = 0;
        for(auto &&p : x) {
            PyObject *item = childf(p);
            if(!item) {
                Py_DECREF(r);
                return item;
            }
            PyTuple_SET_ITEM(r,ti,item);
            ++ti;
        }
        return r;
    }

    PyObject *unexpected_exc() noexcept {
        PyErr_SetString(PyExc_RuntimeError,"unexpected exception thrown by poly_ops");
        return nullptr;
    }

    class gil_unlocker {
        PyThreadState *state;
    public:
        gil_unlocker() : state(PyEval_SaveThread()) {}
        ~gil_unlocker() { PyEval_RestoreThread(state); }
    };

    template<bool Tree,typename R> auto normalize_without_gil(R &&loops) {
        gil_unlocker unlocker;
        return poly_ops::normalize<Tree,std::uint32_t,std::int32_t>(
            std::forward<R>(loops));
    }

    PyObject *_normalize_tree(const std::vector<loop_entry> &loops) noexcept {
        try {
            return proxy_range_to_py(
                normalize_without_gil<true>(loop_vector_to_ranges(loops)),
                &poly_proxy_to_py_tree);
        } catch(const std::bad_alloc&) {
            return PyErr_NoMemory();
        } catch(...) {
            return unexpected_exc();
        }
    }

    PyObject *_normalize_flat(const std::vector<loop_entry> &loops) {
        try {
            return proxy_range_to_py(
                normalize_without_gil<false>(loop_vector_to_ranges(loops)),
                &poly_proxy_to_py);
        } catch(const std::bad_alloc&) {
            return PyErr_NoMemory();
        } catch(...) {
            return unexpected_exc();
        }
    }

    long _winding_dir(const loop_entry &e) {
        gil_unlocker unlocker;
        return poly_ops::winding_dir<std::int32_t>(to_range(e));
    }
    """
    struct loop_entry:
        const char *data
        size_t major_size
        ptrdiff_t minor_stride
        ptrdiff_t major_stride

    object _normalize_tree(const vector[loop_entry]&)
    object _normalize_flat(const vector[loop_entry]&)
    long _winding_dir(const loop_entry&)


cdef array_to_loop_entry(loop_entry &e,object a):
    cdef np.ndarray[np.int32_t,ndim=2] loop
    loop = <np.ndarray[np.int32_t,ndim=2]?>a
    if loop.shape[1] != 2:
        raise TypeError()

    e.data = np.PyArray_BYTES(loop)
    e.major_size = loop.shape[0]
    e.minor_stride = loop.strides[1]
    e.major_stride = loop.strides[0]

cdef py_to_cpp_loops(object loops,vector[loop_entry] &out):
    cdef loop_entry e
    # we need to keep references to the arrays
    cdef tuple loops_tup = tuple(loops)

    cdef np.ndarray[np.int32_t,ndim=2] loop
    for loop in loops_tup:
        array_to_loop_entry(e,loop)
        out.push_back(e)
    return loops_tup

def normalize_tree(loops):
    cdef vector[loop_entry] c_loops
    tmp = py_to_cpp_loops(loops,c_loops)
    return _normalize_tree(c_loops)

def normalize_flat(loops):
    cdef vector[loop_entry] c_loops
    tmp = py_to_cpp_loops(loops,c_loops)
    return _normalize_flat(c_loops)

def winding_dir(loop):
    cdef loop_entry e
    array_to_loop_entry(e,loop)
    return _winding_dir(e)
