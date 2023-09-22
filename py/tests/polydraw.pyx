#cython: language_level=3, boundscheck=False, wraparound=False
#distutils: language = c++

cimport cython
from cython.operator import dereference
from libcpp cimport bool
import numpy as np
cimport numpy as np
cimport common


np.import_array()

cdef extern from "<optional>" namespace "std" nogil:
    cdef cppclass optional[T]:
        optional()
        void reset()
        T& operator*()
        optional& operator=(optional&)
        optional& operator=[U](U&)
        bool operator bool()
        bool operator!()
        bool operator==[U](optional&, U&)
        bool operator!=[U](optional&, U&)


cdef extern from *:
    """
    #include <cstddef>
    #include <cstdint>
    #include <cassert>
    #include <type_traits>
    #include <limits>
    #include <optional>

    #include <poly_ops/polydraw.hpp>


    using rasterizer = poly_ops::draw::rasterizer<coord_t>;
    using scan_line_sweep_state = poly_ops::draw::scan_line_sweep_state<coord_t>;
    using poly_ops::draw::scan_line;

    PyObject *unexpected_exc() noexcept {
        PyErr_SetString(PyExc_RuntimeError,"unexpected exception thrown by poly_ops");
        return nullptr;
    }

    PyObject *_rasterizer_add_loop(
        rasterizer &r,
        PyArrayObject *ar,
        NPY_CASTING casting) noexcept
    {
        if(PyArray_SIZE(ar) == 0) Py_RETURN_NONE;

        try {
            auto sink = r.add_loop();

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

    bool _rasterizer_execute(rasterizer &r,unsigned int width,unsigned int height,std::optional<scan_line_sweep_state> &state) noexcept {
        try {
            state = r.scan_lines(width,height);
            return 1;
        } catch(const std::bad_alloc&) {
            PyErr_NoMemory();
            return 0;
        } catch(...) {
            unexpected_exc();
            return 0;
        }
    }
    PyObject *_rasterizer_next(scan_line_sweep_state &state,rasterizer &rast,scan_line &sc) noexcept {
        try {
            PyObject *r = state(rast,sc) ? Py_True : Py_False;
            Py_INCREF(r);
            return r;
        } catch(const std::bad_alloc&) {
            return PyErr_NoMemory();
        } catch(...) {
            return unexpected_exc();
        }
    }
    """
    const int COORD_T_NPY

    ctypedef struct scan_line:
        unsigned int x1
        unsigned int x2
        unsigned int y
        long winding

    ctypedef struct scan_line_sweep_state:
        pass

    cppclass rasterizer:
        void clear()

    object _rasterizer_add_loop(rasterizer&,np.ndarray,np.NPY_CASTING)
    bint _rasterizer_execute(rasterizer&,unsigned int,unsigned int,optional[scan_line_sweep_state]&) except 0
    object _rasterizer_next(scan_line_sweep_state&,rasterizer&,scan_line&)


@cython.auto_pickle(False)
cdef class ScanLineIter:
    cdef Rasterizer _rast
    cdef optional[scan_line_sweep_state] _state

    def __iter__(self):
        return self

    def __next__(self):
        cdef scan_line sc
        if (not self._state) or _rasterizer_next(dereference(self._state),self._rast._rast,sc) is False:
            if self._state:
                self._rast._itr = None
                self._state.reset()
            raise StopIteration
        return (sc.x1,sc.x2,sc.y,sc.winding)

@cython.auto_pickle(False)
cdef class Rasterizer:
    cdef object __weakref__
    cdef rasterizer _rast
    cdef ScanLineIter _itr

    cdef void _clear_itr(self):
        if self._itr is not None:
            self._itr._state.reset()
            self._itr = None

    def add_loop(self,loop,*,casting='same_kind'):
        cdef np.NPY_CASTING _casting
        self._clear_itr()
        common._casting_converter(casting,&_casting)
        _rasterizer_add_loop(self._rast,common.to_array(loop),_casting)

    def add_loops(self,loops,*,casting='same_kind'):
        cdef np.NPY_CASTING _casting
        self._clear_itr()
        common._casting_converter(casting,&_casting)
        for loop in loops:
            _rasterizer_add_loop(self._rast,common.to_array(loop),_casting)

    def scan_lines(self,unsigned int width,unsigned int height):
        cdef ScanLineIter r = ScanLineIter()
        r._rast = self
        _rasterizer_execute(self._rast,width,height,r._state)
        self._itr = r
        return r

    def reset(self):
        self._rast.clear()

