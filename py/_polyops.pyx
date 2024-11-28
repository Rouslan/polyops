#cython: language_level=3, boundscheck=False, wraparound=False
#distutils: language = c++

cimport cython
import numpy as np
cimport numpy as np
cimport common
import enum
import collections


np.import_array()

cdef extern from *:
    """
    #include <cstddef>
    #include <cassert>
    #include <type_traits>
    #include <limits>

    #include "poly_ops/poly_ops.hpp"


    /* these are generated by Cython */
    PyObject *make_tracked_loop(PyObject *, PyObject *, PyObject *);
    PyObject *make_recur_loop(PyObject *, PyObject *);
    PyObject *make_tracked_recur_loop(PyObject *, PyObject *, PyObject *, PyObject *);

    namespace {

    using point_tracker = poly_ops::origin_point_tracker<>;
    template<bool Tracked> using temp_poly_proxy = poly_ops::temp_polygon_proxy<coord_t,std::size_t,std::conditional_t<Tracked,point_tracker,poly_ops::null_tracker<coord_t>>>;
    using clipper = poly_ops::clipper<coord_t>;
    using ot_clipper = poly_ops::origin_tracked_clipper<coord_t>;
    using long_coord_t = poly_ops::long_coord_t<coord_t>;
    using i_point_tracker = poly_ops::i_point_tracker<>;

    struct py_ref_owner {
        PyObject *ref;
        explicit py_ref_owner(PyObject *ref) noexcept : ref{ref} { assert(ref); }
        py_ref_owner(const py_ref_owner &b) noexcept : ref{Py_NewRef(b.ref)} {}
        ~py_ref_owner() {
            Py_DECREF(ref);
        }

        PyObject *new_ref() const { return Py_NewRef(ref); }
    };
    inline void swap(py_ref_owner &a,py_ref_owner &b) noexcept {
        std::swap(a.ref,b.ref);
    }

    PyObject *poly_proxy_to_py(const temp_poly_proxy<false> &x,PyArray_Descr *dtype) noexcept {
        npy_intp dims[2] = {static_cast<npy_intp>(x.size()),2};

        Py_INCREF(dtype);
        PyObject *loop = PyArray_NewFromDescr(&PyArray_Type,dtype,2,dims,nullptr,nullptr,0,nullptr);
        if(NPY_UNLIKELY(!loop)) return nullptr;
        py_ref_owner loop_o{loop};

        if(x.size()) {
            polyops_npy_range itr(
                reinterpret_cast<PyArrayObject*>(loop),
                NPY_ITER_WRITEONLY | NPY_ITER_UPDATEIFCOPY | NPY_ITER_REFS_OK,
                NPY_UNSAFE_CASTING);
            if(NPY_UNLIKELY(!itr)) return nullptr;

            cond_gil_unlocker unlocker{!itr.needs_api()};
            std::ranges::copy(x,itr.begin());
        }

        return loop_o.new_ref();
    }

    struct tracked_loop_data {
        PyObject *loop;
        PyObject *offsets;
        PyObject *indices;
    };

    tracked_loop_data tracked_poly_proxy_to_py_common(const temp_poly_proxy<true> &x,PyArray_Descr *dtype) noexcept {
        npy_intp dims[2] = {static_cast<npy_intp>(x.size()),2};

        Py_INCREF(dtype);
        PyObject *loop = PyArray_NewFromDescr(&PyArray_Type,dtype,2,dims,nullptr,nullptr,0,nullptr);
        if(NPY_UNLIKELY(!loop)) return {};
        py_ref_owner loop_o{loop};

        std::size_t total_o_indices = 0;

        if(x.size()) {
            polyops_npy_range nr(
                reinterpret_cast<PyArrayObject*>(loop),
                NPY_ITER_WRITEONLY | NPY_ITER_UPDATEIFCOPY | NPY_ITER_REFS_OK,
                NPY_UNSAFE_CASTING);
            if(NPY_UNLIKELY(!nr)) return {};

            cond_gil_unlocker unlocker{!nr.needs_api()};
            auto itr = nr.begin();
            for(auto [p,original] : x) {
                *itr = p;
                ++itr;
                total_o_indices += original.size();
            }
        }

        npy_intp d(x.size() + 1);
        PyObject *offsets = PyArray_SimpleNew(1,&d,NPY_INTP);
        if(NPY_UNLIKELY(!offsets)) return {};
        py_ref_owner offsets_o{offsets};
        npy_intp *offsets_data = reinterpret_cast<npy_intp*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(offsets)));

        d = npy_intp(total_o_indices);
        PyObject *indices = PyArray_SimpleNew(1,&d,NPY_INTP);
        if(NPY_UNLIKELY(!indices)) return {};
        py_ref_owner indices_o{indices};
        npy_intp *indices_data = reinterpret_cast<npy_intp*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(indices)));

        npy_intp offset = 0;
        for(auto [p,original] : x) {
            *offsets_data++ = offset;
            for(std::size_t i : original) {
                *indices_data++ = npy_intp(i);
            }
            offset += npy_intp(original.size());
        }
        *offsets_data = offset;

        return {loop_o.new_ref(),offsets_o.new_ref(),indices_o.new_ref()};
    }

    PyObject *tracked_poly_proxy_to_py(const temp_poly_proxy<true> &x,PyArray_Descr *dtype) noexcept {
        auto [loop,offsets,indices] = tracked_poly_proxy_to_py_common(x,dtype);
        if(NPY_UNLIKELY(!loop)) return nullptr;
        PyObject *r = make_tracked_loop(loop,offsets,indices);
        Py_DECREF(loop);
        Py_DECREF(offsets);
        Py_DECREF(indices);
        return r;
    }

    template<typename R,typename ChildF>
    PyObject *proxy_range_to_py(const R &x,ChildF childf,PyArray_Descr *dtype) noexcept;

    PyObject *poly_proxy_to_py_tree(const temp_poly_proxy<false> &x,PyArray_Descr *dtype) noexcept {
        PyObject *loop = poly_proxy_to_py(x,dtype);
        if(NPY_UNLIKELY(!loop)) return nullptr;
        py_ref_owner loop_o{loop};

        PyObject *children = proxy_range_to_py(x.inner_loops(),&poly_proxy_to_py_tree,dtype);
        if(NPY_UNLIKELY(!children)) return nullptr;
        py_ref_owner children_o{children};

        return make_recur_loop(loop,children);
    }

    PyObject *tracked_poly_proxy_to_py_tree(const temp_poly_proxy<true> &x,PyArray_Descr *dtype) noexcept {
        auto [loop,offsets,indices] = tracked_poly_proxy_to_py_common(x,dtype);
        if(NPY_UNLIKELY(!loop)) return nullptr;
        PyObject *r = nullptr;
 
        PyObject *children = proxy_range_to_py(x.inner_loops(),&tracked_poly_proxy_to_py_tree,dtype);
        if(NPY_LIKELY(children)) {
            r = make_tracked_recur_loop(loop,children,offsets,indices);
            Py_DECREF(children);
        }
        Py_DECREF(loop);
        Py_DECREF(offsets);
        Py_DECREF(indices);

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
        PyErr_SetString(PyExc_RuntimeError,"unexpected exception type");
        return nullptr;
    }

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

    template<bool Tree> auto clipper_execute_without_gil(clipper &c,_bool_op op,point_tracker &pt) {
        gil_unlocker unlocker;
        return c.execute<Tree>(static_cast<poly_ops::bool_op>(op),pt);
    }

    PyObject *_clipper_add_loop(
        clipper &c,
        PyArrayObject *ar,
        _bool_set cat,
        NPY_CASTING casting,
        i_point_tracker *pt) noexcept
    {
        if(PyArray_SIZE(ar) == 0) Py_RETURN_NONE;

        try {
            polyops_npy_range itr(ar,NPY_ITER_READONLY | NPY_ITER_REFS_OK,casting);
            if(NPY_UNLIKELY(!itr)) return nullptr;

            {
                cond_gil_unlocker unlocker{!itr.needs_api()};
                c.add_loops(itr,static_cast<poly_ops::bool_set>(cat),pt);
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
        NPY_CASTING casting,
        i_point_tracker *pt) noexcept
    {
        if(PyArray_SIZE(ar) == 0) Py_RETURN_NONE;

        try {
            polyops_npy_range nr(ar,NPY_ITER_READONLY | NPY_ITER_REFS_OK,casting);
            if(NPY_UNLIKELY(!nr)) return nullptr;

            {
                cond_gil_unlocker unlocker{!nr.needs_api()};
                poly_ops::add_offset_loops(c,nr,static_cast<poly_ops::bool_set>(cat),magnitude,step_size,pt);
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

    PyObject *_tracked_clipper_execute_(int tree,clipper &c,_bool_op op,PyArray_Descr *dtype,point_tracker &pt) noexcept {
        try {
            if(tree) {
                return proxy_range_to_py(
                    clipper_execute_without_gil<true>(c,op,pt),
                    &tracked_poly_proxy_to_py_tree,
                    dtype);
            }

            return proxy_range_to_py(
                clipper_execute_without_gil<false>(c,op,pt),
                &tracked_poly_proxy_to_py,
                dtype);
        } catch(const std::bad_alloc&) {
            return PyErr_NoMemory();
        } catch(...) {
            return unexpected_exc();
        }
    }

    [[maybe_unused]] PyObject *pylong_from_(long x) { return PyLong_FromLong(x); }
    [[maybe_unused]] PyObject *pylong_from_(unsigned long x) { return PyLong_FromUnsignedLong(x); }
    [[maybe_unused]] PyObject *pylong_from_(long long x) { return PyLong_FromLongLong(x); }
    [[maybe_unused]] PyObject *pylong_from_(unsigned long long x) { return PyLong_FromUnsignedLongLong(x); }
    [[maybe_unused]] PyObject *pylong_from_(const poly_ops::large_ints::compound_int<2> &x) {
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

        polyops_npy_range itr(ar,NPY_ITER_READONLY | NPY_ITER_REFS_OK,casting);
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

    } // namespace
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
    
    cppclass i_point_tracker:
        pass
    
    cppclass point_tracker:
        i_point_tracker *callbacks()

    cppclass ot_clipper:
        point_tracker tracker
        clipper base
        void reset()

    object _clipper_add_loop(clipper&,np.ndarray,_bool_set,np.NPY_CASTING,i_point_tracker*)
    object _clipper_add_loop_offset(clipper&,np.ndarray,_bool_set,double,int,np.NPY_CASTING,i_point_tracker*)
    object _tracked_clipper_execute_(bint tree,clipper&,_bool_op,np.dtype,point_tracker&)
    object _clipper_execute_(bint tree,clipper&,_bool_op,np.dtype)
    object _winding_dir(np.ndarray,np.NPY_CASTING)

    object obj_to_dtype_opt(object)

    np.dtype PyArray_PromoteTypes(np.dtype,np.dtype)
    int PyArray_OrderConverter(object,np.NPY_ORDER*) except common.NPY_FAIL
    bint PyArray_CanCastTypeTo(np.dtype,np.dtype,np.NPY_CASTING)


cdef np.npy_intp intp_get(np.ndarray a,Py_ssize_t i):
    return (<np.npy_intp*>np.PyArray_DATA(a))[i]

cdef class PointMap:
    cdef readonly np.ndarray offsets
    cdef readonly np.ndarray indices

    def __cinit__(self,np.ndarray offsets,np.ndarray indices):
        if np.PyArray_TYPE(offsets) != np.NPY_INTP or (np.PyArray_FLAGS(offsets) & np.NPY_ARRAY_C_CONTIGUOUS) != np.NPY_ARRAY_C_CONTIGUOUS:
            raise TypeError('"offsets" must be a contiguous array of type numpy.intp')
        self.offsets = offsets
        self.indices = indices

    def __len__(self):
        return self.offsets.shape[0] - 1

    def __getitem__(self,Py_ssize_t i):
        if not (0 <= i < np.PyArray_DIMS(self.offsets)[0] - 1):
            raise IndexError("index out of range")
        return self.indices[intp_get(self.offsets,i) : intp_get(self.offsets,i+1)]
    
    def index_map(self,out=None):
        cdef np.npy_intp[:] out_raw
        cdef Py_ssize_t out_i = 0
        cdef np.npy_intp next_i
        cdef np.npy_intp orig_i = 0
        cdef np.npy_intp[1] dims = [self.indices.shape[0]]
        cdef np.npy_intp[::1] raw_offsets = self.offsets
        if out is None:
            out = np.PyArray_SimpleNew(1,dims,np.NPY_INTP)
            out_raw = out
        else:
            out_raw = out
            if out_raw.shape[0] != self.indices.shape[0]:
                raise ValueError('"out" must have the same length as "self.indices"')
        for next_i in raw_offsets[1:]:
            while out_i < next_i:
                out_raw[out_i] = orig_i
                out_i += 1
            orig_i += 1

        return out

TrackedLoop = collections.namedtuple('TrackedLoop','loop originals')
RecursiveLoop = collections.namedtuple('RecursiveLoop','loop children')
TrackedRecursiveLoop = collections.namedtuple('RecursiveLoop','loop children originals')

cdef public object make_tracked_loop(loop,offsets,indices):
    return TrackedLoop(loop,PointMap.__new__(PointMap,offsets,indices))

cdef public object make_recur_loop(loop,children):
    return RecursiveLoop(loop,children)

cdef public object make_tracked_recur_loop(loop,children,offsets,indices):
    return RecursiveLoop(loop,children,PointMap.__new__(PointMap,offsets,indices))

cdef load_loop(clipper &c,loop,_bool_set bset,common_dtype,np.NPY_CASTING casting,i_point_tracker *pt):
    cdef np.ndarray ar
    cdef np.dtype r

    ar = common.to_array(loop)
    if common_dtype is None:
        r = ar.descr
    else:
        r = PyArray_PromoteTypes(ar.descr,<np.dtype>common_dtype)
    _clipper_add_loop(c,ar,bset,casting,pt)
    return r

cdef load_loops(clipper &c,loops,_bool_set bset,common_dtype,np.NPY_CASTING casting,i_point_tracker *pt):
    for loop in loops:
        common_dtype = load_loop(c,loop,bset,common_dtype,casting,pt)
    return common_dtype

cdef load_loop_offset(clipper &c,loop,_bool_set bset,double magnitude,int step_size,common_dtype,np.NPY_CASTING casting,i_point_tracker *pt):
    cdef np.ndarray ar
    cdef np.dtype r

    if step_size < 1:
        raise ValueError("'arc_step_size' must be greater than zero")

    ar = common.to_array(loop)
    if common_dtype is None:
        r = ar.descr
    else:
        r = PyArray_PromoteTypes(ar.descr,<np.dtype>common_dtype)
    _clipper_add_loop_offset(c,ar,bset,magnitude,step_size,casting,pt)
    return r

cdef load_loops_offset(clipper &c,loops,_bool_set bset,double magnitude,int step_size,common_dtype,np.NPY_CASTING casting,i_point_tracker *pt):
    for loop in loops:
        common_dtype = load_loop_offset(c,loop,bset,magnitude,step_size,common_dtype,casting,pt)
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

cdef unary_op_(bint tree,loops,_bool_op op,casting_obj,dtype_obj,bint track_points):
    cdef np.NPY_CASTING casting
    cdef clipper c
    cdef point_tracker tracker

    common._casting_converter(casting_obj,&casting)
    dtype_obj = check_dtype(dtype_obj)

    if track_points:
        calc_dtype = load_loops(c,loops,_bool_set.BOOL_SET_SUBJECT,None,casting,tracker.callbacks())
        if calc_dtype is None: return ()

        return _tracked_clipper_execute_(tree,c,op,decide_dtype(<np.dtype>calc_dtype,dtype_obj),tracker)
    
    calc_dtype = load_loops(c,loops,_bool_set.BOOL_SET_SUBJECT,None,casting,NULL)
    if calc_dtype is None: return ()

    return _clipper_execute_(tree,c,op,decide_dtype(<np.dtype>calc_dtype,dtype_obj))

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

def union(loops,*,casting='same_kind',dtype=None,bint tree_out=False,track_points=False):
    return unary_op_(tree_out,loops,_bool_op.BOOL_OP_UNION,casting,dtype,track_points)

def normalize(loops,*,casting='same_kind',dtype=None,bint tree_out=False,track_points=False):
    return unary_op_(tree_out,loops,_bool_op.BOOL_OP_NORMALIZE,casting,dtype,track_points)

def offset(loops,double magnitude,int arc_step_size,*,casting='same_kind',dtype=None,bint tree_out=False,track_points=False):
    cdef np.NPY_CASTING casting_raw
    cdef clipper c
    cdef point_tracker tracker

    common._casting_converter(casting,&casting_raw)
    dtype = check_dtype(dtype)

    if track_points:
        calc_dtype = load_loops_offset(c,loops,_bool_set.BOOL_SET_SUBJECT,magnitude,arc_step_size,None,casting_raw,tracker.callbacks())
        if calc_dtype is None: return ()

        return _tracked_clipper_execute_(tree_out,c,_bool_op.BOOL_OP_UNION,decide_dtype(<np.dtype>calc_dtype,dtype),tracker)

    calc_dtype = load_loops_offset(c,loops,_bool_set.BOOL_SET_SUBJECT,magnitude,arc_step_size,None,casting_raw,NULL)
    if calc_dtype is None: return ()

    return _clipper_execute_(tree_out,c,_bool_op.BOOL_OP_UNION,decide_dtype(<np.dtype>calc_dtype,dtype))

def boolean_op(subject,clip,op,*,casting='same_kind',dtype=None,bint tree_out=False,track_points=False):
    cdef np.NPY_CASTING casting_raw
    cdef clipper c
    cdef point_tracker tracker

    common._casting_converter(casting,&casting_raw)
    dtype = check_dtype(dtype)

    if track_points:
        calc_dtype = load_loops(c,subject,_bool_set.BOOL_SET_SUBJECT,None,casting_raw,tracker.callbacks())
        calc_dtype = load_loops(c,clip,_bool_set.BOOL_SET_CLIP,calc_dtype,casting_raw,tracker.callbacks())
        if calc_dtype is None: return ()

        return _tracked_clipper_execute_(tree_out,c,to_bool_op(op),decide_dtype(<np.dtype>calc_dtype,dtype),tracker)

    calc_dtype = load_loops(c,subject,_bool_set.BOOL_SET_SUBJECT,None,casting_raw,NULL)
    calc_dtype = load_loops(c,clip,_bool_set.BOOL_SET_CLIP,calc_dtype,casting_raw,NULL)
    if calc_dtype is None: return ()

    return _clipper_execute_(tree_out,c,to_bool_op(op),decide_dtype(<np.dtype>calc_dtype,dtype))

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
        self.calc_dtype = load_loop(self._clip[0],loop,to_bool_set(bset),self.calc_dtype,casting,NULL)
    
    cdef _add_loop_offset(self,loop,bset,double magnitude,int step_size,casting_obj):
        cdef np.NPY_CASTING casting
        common._casting_converter(casting_obj,&casting)
        self.calc_dtype = load_loop_offset(self._clip[0],loop,to_bool_set(bset),magnitude,step_size,self.calc_dtype,casting,NULL)

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
        self.calc_dtype = load_loops(self._clip[0],loops,to_bool_set(bset),self.calc_dtype,casting,NULL)

    cdef _add_loops_offset(self,loops,bset,double magnitude,int step_size,casting_obj):
        cdef np.NPY_CASTING casting
        common._casting_converter(casting_obj,&casting)
        self.calc_dtype = load_loops_offset(self._clip[0],loops,bset,magnitude,step_size,self.calc_dtype,casting,NULL)

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

    def execute(self,op,*,dtype=None,bint tree_out=False):
        dtype = check_dtype(dtype)

        if self.calc_dtype is None: return ()

        r = _clipper_execute_(tree_out,self._clip[0],to_bool_op(op),decide_dtype(<np.dtype>self.calc_dtype,dtype))
        self._clip.reset()
        self.calc_dtype = None
        return r

    def reset(self):
        self._clip.reset()
        self.calc_dtype = None

@cython.auto_pickle(False)
cdef class TrackedClipper:
    cdef object __weakref__
    cdef ot_clipper *_clip # ot_clipper is not standard-layout
    cdef object calc_dtype

    def __cinit__(self):
        self._clip = new ot_clipper()

    def __dealloc__(self):
        del self._clip

    cdef _add_loop(self,loop,bset,casting_obj):
        cdef np.NPY_CASTING casting
        common._casting_converter(casting_obj,&casting)
        self.calc_dtype = load_loop(self._clip.base,loop,to_bool_set(bset),self.calc_dtype,casting,self._clip.tracker.callbacks())
    
    cdef _add_loop_offset(self,loop,bset,double magnitude,int step_size,casting_obj):
        cdef np.NPY_CASTING casting
        common._casting_converter(casting_obj,&casting)
        self.calc_dtype = load_loop_offset(self._clip.base,loop,to_bool_set(bset),magnitude,step_size,self.calc_dtype,casting,self._clip.tracker.callbacks())

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
        self.calc_dtype = load_loops(self._clip.base,loops,to_bool_set(bset),self.calc_dtype,casting,self._clip.tracker.callbacks())

    cdef _add_loops_offset(self,loops,bset,double magnitude,int step_size,casting_obj):
        cdef np.NPY_CASTING casting
        common._casting_converter(casting_obj,&casting)
        self.calc_dtype = load_loops_offset(self._clip.base,loops,bset,magnitude,step_size,self.calc_dtype,casting,self._clip.tracker.callbacks())

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

    def execute(self,op,*,dtype=None,bint tree_out=False):
        dtype = check_dtype(dtype)

        if self.calc_dtype is None: return ()

        r = _tracked_clipper_execute_(tree_out,self._clip.base,to_bool_op(op),decide_dtype(<np.dtype>self.calc_dtype,dtype),self._clip.tracker)
        self._clip.reset()
        self.calc_dtype = None
        return r

    def reset(self):
        self._clip.reset()
        self.calc_dtype = None

def winding_dir(loop,*,casting='same_kind'):
    cdef np.NPY_CASTING _casting
    common._casting_converter(casting,&_casting)
    return _winding_dir(common.to_array(loop),_casting)
