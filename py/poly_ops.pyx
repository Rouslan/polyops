#cython: language_level=3, boundscheck=False, wraparound=False
#distutils: language = c++

import numpy as np
cimport numpy as np


np.import_array()

cdef extern from *:
    """
    #include <cstddef>
    #include <cstdint>
    #include <cassert>
    #include <type_traits>

    #include <poly_ops/poly_ops.hpp>

    using temp_poly_tree_range = poly_ops::temp_polygon_tree_range<std::size_t,std::int32_t>;
    using temp_poly_range = poly_ops::temp_polygon_range<std::size_t,std::int32_t>;
    using temp_poly_proxy = poly_ops::temp_polygon_proxy<std::size_t,std::int32_t>;
    using temp_poly_proxy_child_range = decltype(std::declval<temp_poly_proxy>().inner_loops());
    using clipper = poly_ops::clipper<std::size_t,std::int32_t>;

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

    /* Cython 0.29 doesn't work with enum class */
    enum _bool_op {
        BOOL_OP_UNION = static_cast<int>(poly_ops::bool_op::union_),
        BOOL_OP_INTERSECTION = static_cast<int>(poly_ops::bool_op::intersection),
        BOOL_OP_XOR = static_cast<int>(poly_ops::bool_op::xor_),
        BOOL_OP_DIFFERENCE = static_cast<int>(poly_ops::bool_op::difference)};
    enum _bool_cat {
        BOOL_CAT_SUBJECT = static_cast<int>(poly_ops::bool_cat::subject),
        BOOL_CAT_CLIP = static_cast<int>(poly_ops::bool_cat::clip)};

    template<bool Tree> auto clipper_execute_without_gil(clipper &c,_bool_op op) {
        gil_unlocker unlocker;
        c.execute(static_cast<poly_ops::bool_op>(op));
        return c.get_output<Tree>();
    }

    PyObject *_clipper_add_loop(
        clipper &c,
        const loop_entry &loop,
        _bool_cat cat)
    {
        try {
            c.add_loop(to_range(loop),static_cast<poly_ops::bool_cat>(cat));
            Py_RETURN_NONE;
        } catch(const std::bad_alloc&) {
            return PyErr_NoMemory();
        } catch(...) {
            return unexpected_exc();
        }
    }

    PyObject *_clipper_execute_tree(clipper &c,_bool_op op) noexcept
    {
        try {
            return proxy_range_to_py(
                clipper_execute_without_gil<true>(c,op),
                &poly_proxy_to_py_tree);
        } catch(const std::bad_alloc&) {
            return PyErr_NoMemory();
        } catch(...) {
            return unexpected_exc();
        }
    }

    PyObject *_clipper_execute_flat(clipper &c,_bool_op op) noexcept
    {
        try {
            return proxy_range_to_py(
                clipper_execute_without_gil<false>(c,op),
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
    
    cpdef enum BoolOp "_bool_op":
        union "BOOL_OP_UNION",
        intersection "BOOL_OP_INTERSECTION",
        xor "BOOL_OP_XOR",
        difference "BOOL_OP_DIFFERENCE"
    
    cpdef enum BoolCat "_bool_cat":
        subject "BOOL_CAT_SUBJECT",
        clip "BOOL_CAT_CLIP"
    
    cppclass clipper:
        void reset()

    object _clipper_add_loop(clipper&,const loop_entry&,BoolCat)
    object _clipper_execute_tree(clipper&,BoolOp)
    object _clipper_execute_flat(clipper&,BoolOp)
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

cdef load_loops(clipper &c,loops,BoolCat cat):
    cdef loop_entry e
    for loop in loops:
        array_to_loop_entry(e,loop)
        _clipper_add_loop(c,e,cat)

def union_tree(loops):
    cdef clipper c
    load_loops(c,loops,BoolCat.subject)
    return _clipper_execute_tree(c,BoolOp.union)

def union_flat(loops):
    cdef clipper c
    load_loops(c,loops,BoolCat.subject)
    return _clipper_execute_flat(c,BoolOp.union)

def boolean_op_tree(subject,clip,BoolOp op):
    cdef clipper c
    load_loops(c,subject,BoolCat.subject)
    load_loops(c,clip,BoolCat.clip)
    return _clipper_execute_tree(c,op)

def boolean_op_flat(subject,clip,BoolOp op):
    cdef clipper c
    load_loops(c,subject,BoolCat.subject)
    load_loops(c,clip,BoolCat.clip)
    return _clipper_execute_flat(c,op)

cdef class Clipper:
    cdef clipper _clip

    cpdef add_loop(self,loop,BoolCat cat):
        cdef loop_entry e
        array_to_loop_entry(e,loop)
        _clipper_add_loop(self._clip,e,cat)
    
    def add_loop_subject(self,loop):
        self.add_loop(loop,BoolCat.subject)
    
    def add_loop_clip(self,loop):
        self.add_loop(loop,BoolCat.clip)
    
    cpdef add_loops(self,loops,BoolCat cat):
        for loop in loops: self.add_loop(loop,cat)
    
    def add_loops_subject(self,loops):
        self.add_loops(loops,BoolCat.subject)
    
    def add_loops_clip(self,loops):
        self.add_loops(loops,BoolCat.clip)
    
    def execute_tree(self,BoolOp op):
        r = _clipper_execute_tree(self._clip,op)
        self._clip.reset()
        return r
    
    def execute_flat(self,BoolOp op):
        r = _clipper_execute_flat(self._clip,op)
        self._clip.reset()
        return r
    
    def reset(self):
        self._clip.reset()

def winding_dir(loop):
    cdef loop_entry e
    array_to_loop_entry(e,loop)
    return _winding_dir(e)
