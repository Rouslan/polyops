#cython: language_level=3, boundscheck=False, wraparound=False
#distutils: language = c++

cimport cython
import numpy as np
cimport numpy as np
cimport common
import enum


np.import_array()

cdef extern from *:
    """
    #include <cstddef>
    #include <cassert>
    #include <type_traits>
    #include <limits>

    #include <poly_ops/poly_ops.hpp>


    using temp_poly_tree_range = poly_ops::temp_polygon_tree_range<coord_t>;
    using temp_poly_range = poly_ops::temp_polygon_range<coord_t>;
    using temp_poly_proxy = poly_ops::temp_polygon_proxy<coord_t>;
    using temp_poly_proxy_child_range = decltype(std::declval<temp_poly_proxy>().inner_loops());
    using clipper = poly_ops::clipper<coord_t>;
    using long_coord_t = poly_ops::long_coord_t<coord_t>;

    PyObject *poly_proxy_to_py(const temp_poly_proxy &x,PyArray_Descr *dtype) noexcept {
        npy_intp dims[2] = {static_cast<npy_intp>(x.size()),2};

        Py_INCREF(dtype);
        PyObject *r = PyArray_NewFromDescr(&PyArray_Type,dtype,2,dims,nullptr,nullptr,0,nullptr);
        if(NPY_UNLIKELY(!r)) return nullptr;

        if(x.size()) {
            npy_range itr(
                reinterpret_cast<PyArrayObject*>(r),
                NPY_ITER_WRITEONLY | NPY_ITER_UPDATEIFCOPY | NPY_ITER_REFS_OK,
                NPY_UNSAFE_CASTING);
            if(NPY_UNLIKELY(!itr)) {
                Py_DECREF(r);
                return nullptr;
            }

            cond_gil_unlocker unlocker{!itr.needs_api()};
            std::ranges::copy(x,itr.begin());
        }

        return r;
    }

    template<typename R,typename ChildF>
    PyObject *proxy_range_to_py(const R &x,ChildF childf,PyArray_Descr *dtype) noexcept;

    PyObject *poly_proxy_to_py_tree(const temp_poly_proxy &x,PyArray_Descr *dtype) noexcept {
        PyObject *r = PyTuple_New(2);
        if(NPY_UNLIKELY(!r)) return nullptr;
        PyObject *tmp = poly_proxy_to_py(x,dtype);
        if(NPY_UNLIKELY(!tmp)) {
            Py_DECREF(r);
            return tmp;
        }
        PyTuple_SET_ITEM(r,0,tmp);

        tmp = proxy_range_to_py(x.inner_loops(),&poly_proxy_to_py_tree,dtype);
        if(NPY_UNLIKELY(!tmp)) {
            Py_DECREF(r);
            return nullptr;
        }
        PyTuple_SET_ITEM(r,1,tmp);

        return r;
    }

    template<typename R,typename ChildF>
    PyObject *proxy_range_to_py(const R &x,ChildF childf,PyArray_Descr *dtype) noexcept {
        PyObject *r = PyTuple_New(static_cast<Py_ssize_t>(x.size()));
        if(NPY_UNLIKELY(!r)) return r;

        Py_ssize_t ti = 0;
        for(auto &&p : x) {
            PyObject *item = childf(p,dtype);
            if(NPY_UNLIKELY(!item)) {
                Py_DECREF(r);
                return nullptr;
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

    /* Cython 0.29 doesn't work with enum class */
    enum _bool_op {
        BOOL_OP_UNION = static_cast<int>(poly_ops::bool_op::union_),
        BOOL_OP_INTERSECTION = static_cast<int>(poly_ops::bool_op::intersection),
        BOOL_OP_XOR = static_cast<int>(poly_ops::bool_op::xor_),
        BOOL_OP_DIFFERENCE = static_cast<int>(poly_ops::bool_op::difference),
        BOOL_OP_NORMALIZE = static_cast<int>(poly_ops::bool_op::normalize),
        BOOL_OP_INVALID};
    enum _bool_set {
        BOOL_SET_SUBJECT = static_cast<int>(poly_ops::bool_set::subject),
        BOOL_SET_CLIP = static_cast<int>(poly_ops::bool_set::clip),
        BOOL_SET_INVALID};

    template<bool Tree> auto clipper_execute_without_gil(clipper &c,_bool_op op) {
        gil_unlocker unlocker;
        return c.execute<Tree>(static_cast<poly_ops::bool_op>(op));
    }

    PyObject *_clipper_add_loop(
        clipper &c,
        PyArrayObject *ar,
        _bool_set cat,
        NPY_CASTING casting) noexcept
    {
        if(PyArray_SIZE(ar) == 0) Py_RETURN_NONE;

        try {
            npy_range itr(ar,NPY_ITER_READONLY | NPY_ITER_REFS_OK,casting);
            if(NPY_UNLIKELY(!itr)) return nullptr;

            {
                cond_gil_unlocker unlocker{!itr.needs_api()};
                c.add_loops(itr,static_cast<poly_ops::bool_set>(cat));
            }

            Py_RETURN_NONE;
        } catch(const std::bad_alloc&) {
            return PyErr_NoMemory();
        } catch(...) {
            return unexpected_exc();
        }
    }

    PyObject *_clipper_add_loop_offset(
        clipper &c,
        PyArrayObject *ar,
        _bool_set cat,
        double magnitude,
        int step_size,
        NPY_CASTING casting) noexcept
    {
        if(PyArray_SIZE(ar) == 0) Py_RETURN_NONE;

        try {
            npy_range itr(ar,NPY_ITER_READONLY | NPY_ITER_REFS_OK,casting);
            if(NPY_UNLIKELY(!itr)) return nullptr;

            {
                cond_gil_unlocker unlocker{!itr.needs_api()};
                poly_ops::add_offset_loops(c,itr,static_cast<poly_ops::bool_set>(cat),magnitude,step_size);
            }

            Py_RETURN_NONE;
        } catch(const std::bad_alloc&) {
            return PyErr_NoMemory();
        } catch(...) {
            return unexpected_exc();
        }
    }

    PyObject *_clipper_execute_(int tree,clipper &c,_bool_op op,PyArray_Descr *dtype) noexcept {
        try {
            if(tree) {
                return proxy_range_to_py(
                    clipper_execute_without_gil<true>(c,op),
                    &poly_proxy_to_py_tree,
                    dtype);
            }

            return proxy_range_to_py(
                clipper_execute_without_gil<false>(c,op),
                &poly_proxy_to_py,
                dtype);
        } catch(const std::bad_alloc&) {
            return PyErr_NoMemory();
        } catch(...) {
            return unexpected_exc();
        }
    }

    PyObject *pylong_from_(long x) { return PyLong_FromLong(x); }
    PyObject *pylong_from_(unsigned long x) { return PyLong_FromUnsignedLong(x); }
    PyObject *pylong_from_(long long x) { return PyLong_FromLongLong(x); }
    PyObject *pylong_from_(unsigned long long x) { return PyLong_FromUnsignedLongLong(x); }
    PyObject *pylong_from_(const poly_ops::large_ints::compound_int<2> &x) {
    #if NPY_BYTE_ORDER == NPY_LITTLE_ENDIAN
        std::uint64_t bits[2] = {x[0],x[1]};
        const int little_endian = 1;
    #else
        std::uint64_t bits[2] = {x[1],x[0]};
        const int little_endian = 0;
    #endif
        return _PyLong_FromByteArray(reinterpret_cast<const unsigned char*>(bits),16,little_endian,1);
    }

    PyObject *_winding_dir(PyArrayObject *ar,NPY_CASTING casting) noexcept {
        if(PyArray_SIZE(ar) == 0) return pylong_from_(0l);

        npy_range itr(ar,NPY_ITER_READONLY | NPY_ITER_REFS_OK,casting);
        if(NPY_UNLIKELY(!itr)) return nullptr;

        poly_ops::long_coord_t<coord_t> r;

        {
            cond_gil_unlocker unlocker{!itr.needs_api()};
            r = poly_ops::winding_dir<coord_t>(itr);
        }

        return pylong_from_(r);
    }

    PyObject *obj_to_dtype_opt(PyObject *obj) noexcept {
        PyArray_Descr *r;
        if(PyArray_DescrConverter2(obj,&r) == NPY_FAIL) return nullptr;
        if(r == nullptr) Py_RETURN_NONE;
        return reinterpret_cast<PyObject*>(r);
    }
    """
    cdef enum _bool_op:
        BOOL_OP_UNION
        BOOL_OP_INTERSECTION
        BOOL_OP_XOR
        BOOL_OP_DIFFERENCE
        BOOL_OP_NORMALIZE
        BOOL_OP_INVALID

    cdef enum _bool_set:
        BOOL_SET_SUBJECT
        BOOL_SET_CLIP
        BOOL_SET_INVALID

    cppclass clipper:
        void reset()

    object _clipper_add_loop(clipper&,np.ndarray,_bool_set,np.NPY_CASTING)
    object _clipper_add_loop_offset(clipper&,np.ndarray,_bool_set,double,int,np.NPY_CASTING)
    object _clipper_execute_(bint tree,clipper&,_bool_op,np.dtype)
    object _winding_dir(np.ndarray,np.NPY_CASTING)

    object obj_to_dtype_opt(object)

    np.dtype PyArray_PromoteTypes(np.dtype,np.dtype)
    int PyArray_OrderConverter(object,np.NPY_ORDER*) except common.NPY_FAIL
    bint PyArray_CanCastTypeTo(np.dtype,np.dtype,np.NPY_CASTING)


cdef load_loop(clipper &c,loop,_bool_set bset,common_dtype,np.NPY_CASTING casting):
    cdef np.ndarray ar
    cdef np.dtype r

    ar = common.to_array(loop)
    if common_dtype is None:
        r = ar.descr
    else:
        r = PyArray_PromoteTypes(ar.descr,<np.dtype>common_dtype)
    _clipper_add_loop(c,ar,bset,casting)
    return r

cdef load_loops(clipper &c,loops,_bool_set bset,common_dtype,np.NPY_CASTING casting):
    for loop in loops:
        common_dtype = load_loop(c,loop,bset,common_dtype,casting)
    return common_dtype

cdef load_loop_offset(clipper &c,loop,_bool_set bset,double magnitude,int step_size,common_dtype,np.NPY_CASTING casting):
    cdef np.ndarray ar
    cdef np.dtype r

    if step_size < 1:
        raise ValueError("'arc_step_size' must be greater than zero")

    ar = common.to_array(loop)
    if common_dtype is None:
        r = ar.descr
    else:
        r = PyArray_PromoteTypes(ar.descr,<np.dtype>common_dtype)
    _clipper_add_loop_offset(c,ar,bset,magnitude,step_size,casting)
    return r

cdef load_loops_offset(clipper &c,loops,_bool_set bset,double magnitude,int step_size,common_dtype,np.NPY_CASTING casting):
    for loop in loops:
        common_dtype = load_loop_offset(c,loop,bset,magnitude,step_size,common_dtype,casting)
    return common_dtype

cdef np.dtype decide_dtype(np.dtype calculated,explicit):
    return calculated if explicit is None else <np.dtype>explicit

cdef check_dtype(obj):
    obj = obj_to_dtype_opt(obj)
    if obj is not None:
        if not PyArray_CanCastTypeTo(
                np.PyArray_DescrFromType(common.COORD_T_NPY),
                <np.dtype>obj,
                np.NPY_UNSAFE_CASTING):
            raise TypeError('Cannot convert to the specified "dtype"')
    return obj

cdef unary_op_(bint tree,loops,_bool_op op,casting_obj,dtype_obj):
    cdef np.NPY_CASTING casting
    cdef clipper c

    common._casting_converter(casting_obj,&casting)
    dtype_obj = check_dtype(dtype_obj)

    calc_dtype = load_loops(c,loops,_bool_set.BOOL_SET_SUBJECT,None,casting)
    if calc_dtype is None: return ()

    return _clipper_execute_(tree,c,op,decide_dtype(<np.dtype>calc_dtype,dtype_obj))

cdef offset_op_(bint tree,loops,double magnitude,int step_size,casting_obj,dtype_obj):
    cdef np.NPY_CASTING casting
    cdef clipper c

    common._casting_converter(casting_obj,&casting)
    dtype_obj = check_dtype(dtype_obj)

    calc_dtype = load_loops_offset(c,loops,_bool_set.BOOL_SET_SUBJECT,magnitude,step_size,None,casting)
    if calc_dtype is None: return ()

    return _clipper_execute_(tree,c,_bool_op.BOOL_OP_UNION,decide_dtype(<np.dtype>calc_dtype,dtype_obj))

# using pure-Python enums results in a lot less code generated
class BoolOp(enum.IntEnum):
    union = _bool_op.BOOL_OP_UNION
    intersection = _bool_op.BOOL_OP_INTERSECTION
    xor = _bool_op.BOOL_OP_XOR
    difference = _bool_op.BOOL_OP_DIFFERENCE
    normalize = _bool_op.BOOL_OP_NORMALIZE

class BoolSet(enum.IntEnum):
    subject = _bool_set.BOOL_SET_SUBJECT
    clip = _bool_set.BOOL_SET_CLIP

cdef _bool_op to_bool_op(x) except _bool_op.BOOL_OP_INVALID:
    cdef int val = x
    if (val != _bool_op.BOOL_OP_UNION
        and val != _bool_op.BOOL_OP_INTERSECTION
        and val != _bool_op.BOOL_OP_XOR
        and val != _bool_op.BOOL_OP_DIFFERENCE
        and val != _bool_op.BOOL_OP_NORMALIZE):
        raise TypeError('"op" is not one of the valid operation types')
    return <_bool_op>val

cdef _bool_set to_bool_set(x) except _bool_set.BOOL_SET_INVALID:
    cdef int val = x
    if (val != _bool_set.BOOL_SET_SUBJECT
        and val != _bool_set.BOOL_SET_CLIP):
        raise TypeError('"bset" is not one of the valid operation types')
    return <_bool_set>val

def union_tree(loops,*,casting='same_kind',dtype=None):
    return unary_op_(1,loops,_bool_op.BOOL_OP_UNION,casting,dtype)

def union_flat(loops,*,casting='same_kind',dtype=None):
    return unary_op_(0,loops,_bool_op.BOOL_OP_UNION,casting,dtype)

def normalize_tree(loops,*,casting='same_kind',dtype=None):
    return unary_op_(1,loops,_bool_op.BOOL_OP_NORMALIZE,casting,dtype)

def normalize_flat(loops,*,casting='same_kind',dtype=None):
    return unary_op_(0,loops,_bool_op.BOOL_OP_NORMALIZE,casting,dtype)

def offset_tree(loops,double magnitude,int arc_step_size,*,casting='same_kind',dtype=None):
    return offset_op_(1,loops,magnitude,arc_step_size,casting,dtype)

def offset_flat(loops,double magnitude,int arc_step_size,*,casting='same_kind',dtype=None):
    return offset_op_(0,loops,magnitude,arc_step_size,casting,dtype)

cdef boolean_op_(bint tree,subject,clip,op,casting_obj,dtype_obj):
    cdef np.NPY_CASTING casting
    cdef clipper c

    common._casting_converter(casting_obj,&casting)
    dtype_obj = check_dtype(dtype_obj)

    calc_dtype = load_loops(c,subject,_bool_set.BOOL_SET_SUBJECT,None,casting)
    calc_dtype = load_loops(c,clip,_bool_set.BOOL_SET_CLIP,calc_dtype,casting)
    if calc_dtype is None: return ()

    return _clipper_execute_(tree,c,to_bool_op(op),decide_dtype(<np.dtype>calc_dtype,dtype_obj))

def boolean_op_tree(subject,clip,op,*,casting='same_kind',dtype=None):
    return boolean_op_(1,subject,clip,op,casting,dtype)

def boolean_op_flat(subject,clip,op,*,casting='same_kind',dtype=None):
    return boolean_op_(0,subject,clip,op,casting,dtype)

@cython.auto_pickle(False)
cdef class Clipper:
    cdef object __weakref__
    cdef clipper *_clip # clipper is not standard-layout
    cdef object calc_dtype

    def __cinit__(self):
        self._clip = new clipper()
    
    def __dealloc__(self):
        del self._clip

    cdef _add_loop(self,loop,bset,casting_obj):
        cdef np.NPY_CASTING casting
        common._casting_converter(casting_obj,&casting)
        self.calc_dtype = load_loop(self._clip[0],loop,to_bool_set(bset),self.calc_dtype,casting)
    
    cdef _add_loop_offset(self,loop,bset,double magnitude,int step_size,casting_obj):
        cdef np.NPY_CASTING casting
        common._casting_converter(casting_obj,&casting)
        self.calc_dtype = load_loop_offset(self._clip[0],loop,to_bool_set(bset),magnitude,step_size,self.calc_dtype,casting)

    def add_loop(self,loop,bset,*,casting='same_kind'):
        self._add_loop(loop,bset,casting)

    def add_loop_subject(self,loop,*,casting='same_kind'):
        self._add_loop(loop,_bool_set.BOOL_SET_SUBJECT,casting)

    def add_loop_clip(self,loop,*,casting='same_kind'):
        self._add_loop(loop,_bool_set.BOOL_SET_CLIP,casting)
    
    def add_loop_offset(self,loop,bset,double magnitude,int arc_step_size,*,casting='same_kind'):
        self._add_loop_offset(loop,bset,magnitude,arc_step_size,casting)

    def add_loop_offset_subject(self,loop,double magnitude,int arc_step_size,*,casting='same_kind'):
        self._add_loop_offset(loop,_bool_set.BOOL_SET_SUBJECT,magnitude,arc_step_size,casting)

    def add_loop_offset_clip(self,loop,double magnitude,int arc_step_size,*,casting='same_kind'):
        self._add_loop_offset(loop,_bool_set.BOOL_SET_CLIP,magnitude,arc_step_size,casting)

    cdef _add_loops(self,loops,bset,casting_obj):
        cdef np.NPY_CASTING casting
        common._casting_converter(casting_obj,&casting)
        self.calc_dtype = load_loops(self._clip[0],loops,to_bool_set(bset),self.calc_dtype,casting)

    cdef _add_loops_offset(self,loops,bset,double magnitude,int step_size,casting_obj):
        cdef np.NPY_CASTING casting
        common._casting_converter(casting_obj,&casting)
        self.calc_dtype = load_loops_offset(self._clip[0],loops,bset,magnitude,step_size,self.calc_dtype,casting)

    def add_loops(self,loops,bset,*,casting='same_kind'):
        self._add_loops(loops,bset,casting)

    def add_loops_subject(self,loops,*,casting='same_kind'):
        self._add_loops(loops,_bool_set.BOOL_SET_SUBJECT,casting)

    def add_loops_clip(self,loops,*,casting='same_kind'):
        self._add_loops(loops,_bool_set.BOOL_SET_CLIP,casting)
    
    def add_loops_offset(self,loops,bset,double magnitude,int arc_step_size,*,casting='same_kind'):
        self._add_loops_offset(loops,bset,magnitude,arc_step_size,casting)

    def add_loops_offset_subject(self,loops,double magnitude,int arc_step_size,*,casting='same_kind'):
        self._add_loops_offset(loops,_bool_set.BOOL_SET_SUBJECT,magnitude,arc_step_size,casting)

    def add_loops_offset_clip(self,loops,double magnitude,int arc_step_size,*,casting='same_kind'):
        self._add_loops_offset(loops,_bool_set.BOOL_SET_CLIP,magnitude,arc_step_size,casting)

    cdef execute_(self,bint tree,op,dtype_obj):
        dtype_obj = check_dtype(dtype_obj)

        if self.calc_dtype is None: return ()

        r = _clipper_execute_(tree,self._clip[0],to_bool_op(op),decide_dtype(<np.dtype>self.calc_dtype,dtype_obj))
        self._clip[0].reset()
        self.calc_dtype = None
        return r

    def execute_tree(self,op,*,dtype=None):
        return self.execute_(1,op,dtype)

    def execute_flat(self,op,*,dtype=None):
        return self.execute_(0,op,dtype)

    def reset(self):
        self._clip[0].reset()
        self.calc_dtype = None

def winding_dir(loop,*,casting='same_kind'):
    cdef np.NPY_CASTING _casting
    common._casting_converter(casting,&_casting)
    return _winding_dir(common.to_array(loop),_casting)
