cimport numpy as np

cdef extern from *:
    """
    #include <cstdint>
    #include <type_traits>
    #include <poly_ops/large_ints.hpp>

    constexpr inline bool IS_64BIT_PLATFORM = sizeof(poly_ops::large_ints::full_uint) == 8;

    using coord_t = std::conditional_t<IS_64BIT_PLATFORM,std::int64_t,std::int32_t>;
    constexpr inline int COORD_T_NPY = IS_64BIT_PLATFORM ? NPY_INT64 : NPY_INT32;

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

    class gil_unlocker {
        PyThreadState *state;
    public:
        gil_unlocker() : state(PyEval_SaveThread()) {}
        ~gil_unlocker() { PyEval_RestoreThread(state); }
    };

    inline int _casting_converter(PyObject *obj,NPY_CASTING *casting) noexcept {
        /* PyArray_CastingConverter will leave "casting" unchanged if "obj" is
        None */
        *casting = NPY_SAME_KIND_CASTING;
        return PyArray_CastingConverter(obj,casting);
    }
    """
    const int NPY_FAIL

    int _casting_converter(object,np.NPY_CASTING*) except NPY_FAIL

cdef inline to_array(x):
    cdef np.ndarray ar = <np.ndarray>np.PyArray_FROM_O(x)
    if np.PyArray_NDIM(ar) != 2 or np.PyArray_DIMS(ar)[1] != 2:
        raise TypeError('input array must have two dimensions, with the second dimension having a size of two')
    return ar
