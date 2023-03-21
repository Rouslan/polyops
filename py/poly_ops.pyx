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
    #include <limits>

    #include <poly_ops/poly_ops.hpp>


    #if POLY_OPS_HAVE_128BIT_INT
    using coord_t = std::int64_t;
    constexpr inline int COORD_T_NPY = NPY_INT64;
    #else
    using coord_t = std::int32_t;
    constexpr inline int COORD_T_NPY = NPY_INT32;
    #endif

    using temp_poly_tree_range = poly_ops::temp_polygon_tree_range<coord_t>;
    using temp_poly_range = poly_ops::temp_polygon_range<coord_t>;
    using temp_poly_proxy = poly_ops::temp_polygon_proxy<coord_t>;
    using temp_poly_proxy_child_range = decltype(std::declval<temp_poly_proxy>().inner_loops());
    using clipper = poly_ops::clipper<coord_t>;
    using long_coord_t = poly_ops::long_coord_t<coord_t>;


    struct npy_iterator {
        NpyIter *itr;
        NpyIter_IterNextFunc *itr_next;
        char **data_ptr;
        npy_intp *stride_ptr;
        npy_intp *inner_size_ptr;

        char *data;
        npy_intp stride;
        const char *run_end;

        npy_iterator(PyArrayObject *ar,int flag,NPY_CASTING casting) : itr{nullptr}, itr_next{nullptr}
        {
            auto dtype = PyArray_DescrFromType(COORD_T_NPY);
            if(NPY_UNLIKELY(!dtype)) return;
            itr = NpyIter_New(
                ar,
                flag | NPY_ITER_REFS_OK | NPY_ITER_COPY | NPY_ITER_EXTERNAL_LOOP,
                NPY_CORDER,
                casting,
                dtype);
            Py_DECREF(dtype);
            if(NPY_UNLIKELY(!itr)) return;

            itr_next = NpyIter_GetIterNext(itr,nullptr);
            if(NPY_UNLIKELY(!itr_next)) return;

            data_ptr = NpyIter_GetDataPtrArray(itr);
            stride_ptr = NpyIter_GetInnerStrideArray(itr);
            inner_size_ptr = NpyIter_GetInnerLoopSizePtr(itr);

            read_pointers();
        }
        npy_iterator(const npy_iterator&) = delete;
        ~npy_iterator() { if(itr) NpyIter_Deallocate(itr); }

        npy_iterator &operator=(const npy_iterator&) = delete;

        explicit operator bool() const {
            return itr_next != nullptr;
        }

        void read_pointers() {
            data = *data_ptr;
            stride = *stride_ptr;
            run_end = data + *inner_size_ptr * stride;
        }

        bool next_check() {
            data += stride;
            if(data == run_end) {
                if(!(*itr_next)(itr)) return false;
                read_pointers();
            }
            return true;
        }

        void next() {
        #ifdef NDEBUG
            next_check();
        #else
            bool more = next_check();
            assert(more);
        #endif
        }

        coord_t &item() { return *reinterpret_cast<coord_t*>(data); }
    };

    PyObject *poly_proxy_to_py(const temp_poly_proxy &x,PyArray_Descr *dtype) noexcept {
        npy_intp dims[2] = {static_cast<npy_intp>(x.size()),2};

        Py_INCREF(dtype);
        PyObject *r = PyArray_NewFromDescr(&PyArray_Type,dtype,2,dims,nullptr,nullptr,0,nullptr);
        if(NPY_UNLIKELY(!r)) return nullptr;

        if(x.size()) {
            npy_iterator itr(
                reinterpret_cast<PyArrayObject*>(r),
                NPY_ITER_WRITEONLY | NPY_ITER_UPDATEIFCOPY,
                NPY_UNSAFE_CASTING);
            if(NPY_UNLIKELY(!itr)) {
                Py_DECREF(r);
                return nullptr;
            }

            auto p_itr = x.begin();
            do {
                assert(p_itr != x.end());
                auto &&p = *p_itr++;
                itr.item() = p[0];
                itr.next();
                itr.item() = p[1];
            } while(itr.next_check());
            assert(p_itr == x.end());
        }

        return r;
    }

    template<typename R,typename ChildF>
    PyObject *proxy_range_to_py(const R &x,ChildF childf,PyArray_Descr *dtype) noexcept;

    PyObject *poly_proxy_to_py_tree(const temp_poly_proxy &x,PyArray_Descr *dtype) noexcept {
        PyObject *r = PyTuple_New(2);
        if(NPY_UNLIKELY(!r)) return r;
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
        if(!r) return r;

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
        BOOL_OP_DIFFERENCE = static_cast<int>(poly_ops::bool_op::difference),
        BOOL_OP_NORMALIZE = static_cast<int>(poly_ops::bool_op::normalize)};
    enum _bool_set {
        BOOL_SET_SUBJECT = static_cast<int>(poly_ops::bool_set::subject),
        BOOL_SET_CLIP = static_cast<int>(poly_ops::bool_set::clip)};

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
            auto sink = c.add_loop(static_cast<poly_ops::bool_set>(cat));

            npy_iterator itr(ar,NPY_ITER_READONLY,casting);
            if(NPY_UNLIKELY(!itr)) return nullptr;

            do {
                poly_ops::point_t<coord_t> p;
                p[0] = itr.item();
                itr.next();
                p[1] = itr.item();
                sink(p);
            } while(itr.next_check());

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

    PyObject *_winding_dir(PyArrayObject *ar,NPY_CASTING casting) noexcept {
        if(PyArray_SIZE(ar) == 0) return pylong_from_(0l);

        npy_iterator itr(ar,NPY_ITER_READONLY,casting);
        if(NPY_UNLIKELY(!itr)) return nullptr;

        poly_ops::point_t<coord_t> p;
        p[0] = itr.item();
        itr.next();
        p[1] = itr.item();

        poly_ops::winding_dir_sink<coord_t> sink{p};

        while(itr.next_check()) {
            p[0] = itr.item();
            itr.next();
            p[1] = itr.item();
            sink(p);
        }

        long_coord_t result = sink.close();

    #if POLY_OPS_HAVE_128BIT_INT
        
    #if NPY_BYTE_ORDER == NPY_LITTLE_ENDIAN
        std::uint64_t bits[2] = {result.lo(),result.hi()};
        const int little_endian = 1;
    #else
        std::uint64_t bits[2] = {result.hi(),result.lo()};
        const int little_endian = 0;
    #endif
        return _PyLong_FromByteArray(reinterpret_cast<const unsigned char*>(bits),16,little_endian,1);

    #else

        return pylong_from_(result);
        
    #endif
    }

    PyObject *obj_to_dtype_opt(PyObject *obj) noexcept {
        PyArray_Descr *r;
        if(PyArray_DescrConverter2(obj,&r) == NPY_FAIL) return nullptr;
        if(r == nullptr) Py_RETURN_NONE;
        return reinterpret_cast<PyObject*>(r);
    }

    int _casting_converter(PyObject *obj,NPY_CASTING *casting) noexcept {
        /* PyArray_CastingConverter will leave "casting" unchanged if "obj" is
        None */
        *casting = NPY_SAME_KIND_CASTING;
        return PyArray_CastingConverter(obj,casting);
    }
    """
    const int COORD_T_NPY

    cpdef enum BoolOp "_bool_op":
        union "BOOL_OP_UNION",
        intersection "BOOL_OP_INTERSECTION",
        xor "BOOL_OP_XOR",
        difference "BOOL_OP_DIFFERENCE"
        normalize "BOOL_OP_NORMALIZE"
    
    cpdef enum BoolSet "_bool_set":
        subject "BOOL_SET_SUBJECT",
        clip "BOOL_SET_CLIP"
    
    cppclass clipper:
        void reset()

    object _clipper_add_loop(clipper&,np.ndarray,BoolSet,np.NPY_CASTING)
    object _clipper_execute_(bint tree,clipper&,BoolOp,np.dtype)
    object _winding_dir(np.ndarray,np.NPY_CASTING)

    object obj_to_dtype_opt(object)

    const int NPY_FAIL

    np.dtype PyArray_PromoteTypes(np.dtype,np.dtype)
    int _casting_converter(object,np.NPY_CASTING*) except NPY_FAIL
    int PyArray_OrderConverter(object,np.NPY_ORDER*) except NPY_FAIL
    bint PyArray_CanCastTypeTo(np.dtype,np.dtype,np.NPY_CASTING)


cdef to_array(x):
    ar = <np.ndarray>np.PyArray_FROM_O(x)
    if ar.ndim != 2 or ar.shape[1] != 2:
        raise TypeError('input array must have two dimensions, with the second dimension having a size of two')
    return ar

cdef load_loop(clipper &c,loop,BoolSet bset,common_dtype,np.NPY_CASTING casting):
    cdef np.ndarray ar
    cdef np.dtype r

    ar = to_array(loop)
    if common_dtype is None:
        r = ar.descr
    else:
        r = PyArray_PromoteTypes(ar.descr,<np.dtype>common_dtype)
    _clipper_add_loop(c,ar,bset,casting)
    return r

cdef load_loops(clipper &c,loops,BoolSet bset,common_dtype,np.NPY_CASTING casting):
    for loop in loops:
        common_dtype = load_loop(c,loop,bset,common_dtype,casting)
    return common_dtype

cdef np.dtype decide_dtype(np.dtype calculated,explicit):
    return calculated if explicit is None else <np.dtype>explicit

cdef check_dtype(obj):
    obj = obj_to_dtype_opt(obj)
    if obj is not None:
        if not PyArray_CanCastTypeTo(
                np.PyArray_DescrFromType(COORD_T_NPY),
                <np.dtype>obj,
                np.NPY_UNSAFE_CASTING):
            raise TypeError('Cannot convert to the specified "dtype"')
    return obj

cdef unary_op_(bint tree,loops,BoolOp op,casting_obj,dtype_obj):
    cdef np.NPY_CASTING casting
    cdef clipper c

    _casting_converter(casting_obj,&casting)
    dtype_obj = check_dtype(dtype_obj)

    calc_dtype = load_loops(c,loops,BoolSet.subject,None,casting)
    if calc_dtype is None: return ()

    return _clipper_execute_(tree,c,op,decide_dtype(<np.dtype>calc_dtype,dtype_obj))

def union_tree(loops,*,casting='same_kind',dtype=None):
    return unary_op_(1,loops,BoolOp.union,casting,dtype)

def union_flat(loops,*,casting='same_kind',dtype=None):
    return unary_op_(0,loops,BoolOp.union,casting,dtype)

def normalize_tree(loops,*,casting='same_kind',dtype=None):
    return unary_op_(1,loops,BoolOp.normalize,casting,dtype)

def normalize_flat(loops,*,casting='same_kind',dtype=None):
    return unary_op_(0,loops,BoolOp.normalize,casting,dtype)

cdef boolean_op_(bint tree,subject,clip,BoolOp op,casting_obj,dtype_obj):
    cdef np.NPY_CASTING casting
    cdef clipper c

    _casting_converter(casting_obj,&casting)
    dtype_obj = check_dtype(dtype_obj)

    calc_dtype = load_loops(c,subject,BoolSet.subject,None,casting)
    calc_dtype = load_loops(c,clip,BoolSet.clip,calc_dtype,casting)
    if calc_dtype is None: return ()

    return _clipper_execute_(tree,c,op,decide_dtype(<np.dtype>calc_dtype,dtype_obj))

def boolean_op_tree(subject,clip,BoolOp op,*,casting='same_kind',dtype=None):
    return boolean_op_(1,subject,clip,op,casting,dtype)

def boolean_op_flat(subject,clip,BoolOp op,*,casting='same_kind',dtype=None):
    return boolean_op_(0,subject,clip,op,casting,dtype)

cdef class Clipper:
    cdef clipper _clip
    cdef object calc_dtype

    cdef _add_loop(self,loop,BoolSet bset,casting_obj):
        cdef np.NPY_CASTING casting
        _casting_converter(casting_obj,&casting)
        self.calc_dtype = load_loop(self._clip,loop,bset,self.calc_dtype,casting)

    def add_loop(self,loop,BoolSet bset,*,casting='same_kind'):
        self._add_loop(loop,bset,casting)
    
    def add_loop_subject(self,loop,*,casting='same_kind'):
        self._add_loop(loop,BoolSet.subject,casting)
    
    def add_loop_clip(self,loop,*,casting='same_kind'):
        self._add_loop(loop,BoolSet.clip,casting)
    
    cdef _add_loops(self,loops,BoolSet bset,casting_obj):
        cdef np.NPY_CASTING casting
        _casting_converter(casting_obj,&casting)
        self.calc_dtype = load_loops(self._clip,loops,bset,self.calc_dtype,casting)
    
    def add_loops(self,loops,BoolSet bset,*,casting='same_kind'):
        load_loops(self._clip,loops,bset,self.calc_dtype,casting)
    
    def add_loops_subject(self,loops,*,casting='same_kind'):
        self._add_loops(loops,BoolSet.subject,casting)
    
    def add_loops_clip(self,loops,*,casting='same_kind'):
        self._add_loops(loops,BoolSet.clip,casting)
    
    cdef execute_(self,bint tree,BoolOp op,dtype_obj):
        dtype_obj = check_dtype(dtype_obj)

        if self.calc_dtype is None: return ()

        r = _clipper_execute_(tree,self._clip,op,decide_dtype(<np.dtype>self.calc_dtype,dtype_obj))
        self._clip.reset()
        self.calc_dtype = None
        return r
    
    def execute_tree(self,BoolOp op,*,dtype=None):
        return self.execute_(1,op,dtype)
    
    def execute_flat(self,BoolOp op,*,dtype=None):
        return self.execute_(0,op,dtype)
    
    def reset(self):
        self._clip.reset()
        self.calc_dtype = None

def winding_dir(loop,*,casting='same_kind'):
    cdef np.NPY_CASTING _casting
    _casting_converter(casting,&_casting)
    return _winding_dir(to_array(loop),_casting)
