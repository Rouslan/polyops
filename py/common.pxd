cimport numpy as np

cdef extern from *:
    """
    #include <cstdint>
    #include <type_traits>
    #include <poly_ops/base.hpp>
    #include <poly_ops/large_ints.hpp>

    constexpr inline bool IS_64BIT_PLATFORM = sizeof(poly_ops::large_ints::full_uint) == 8;

    using coord_t = std::conditional_t<IS_64BIT_PLATFORM,std::int64_t,std::int32_t>;
    constexpr inline int COORD_T_NPY = IS_64BIT_PLATFORM ? NPY_INT64 : NPY_INT32;

    struct npy_point_iterator;

    struct npy_range {
        NpyIter *itr;
        NpyIter_IterNextFunc *itr_next;
        char **data_ptr;
        npy_intp *stride_ptr;
        npy_intp *inner_size_ptr;

        char *data;
        npy_intp stride;
        const char *run_end;

        npy_range(PyArrayObject *ar,int flag,NPY_CASTING casting) noexcept : itr{nullptr}, itr_next{nullptr}
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
        npy_range(const npy_range&) = delete;
        ~npy_range() { if(itr) NpyIter_Deallocate(itr); }

        npy_range &operator=(const npy_range&) = delete;

        explicit operator bool() const {
            return itr_next != nullptr;
        }

        void read_pointers() {
            data = *data_ptr;
            stride = *stride_ptr;
            run_end = data + *inner_size_ptr * stride;
        }

        npy_point_iterator begin() noexcept;
        std::default_sentinel_t end() noexcept { return {}; }

        bool needs_api() const noexcept {
            return NpyIter_IterationNeedsAPI(itr) != NPY_FALSE;
        }
    };

    struct npy_point_proxy {
        npy_range *range;

        coord_t &operator[](npy_intp i) const noexcept {
            assert((range->data + i*range->stride) != range->run_end);
            return *reinterpret_cast<coord_t*>(range->data + i*range->stride);
        }

        const npy_point_proxy &operator=(const poly_ops::point_t<coord_t> &b) const noexcept {
            (*this)[0] = b.x();
            (*this)[1] = b.y();
            return *this;
        }
    };

    namespace poly_ops {
        template<> struct point_ops<npy_point_proxy> {
            static const coord_t &get_x(npy_point_proxy p) noexcept { return p[0]; }
            static const coord_t &get_y(npy_point_proxy p) noexcept { return p[1]; }
        };
    }

    struct npy_point_iterator : private npy_point_proxy {
        using value_type = const npy_point_proxy;
        using difference_type = std::ptrdiff_t;

        npy_point_iterator() noexcept = default;
        explicit npy_point_iterator(npy_range *range) : npy_point_proxy{range} {}
        npy_point_iterator(const npy_point_iterator&) noexcept = default;
    
        npy_point_iterator &operator++() noexcept {
            assert(range && range->data);

            range->data += range->stride*2;
            if(range->data == range->run_end) {
                if(!(*range->itr_next)(range->itr)) range->data = nullptr;
                else range->read_pointers();
            }
            return *this;
        }

        void operator++(int) noexcept {
            ++(*this);
        }

        value_type &operator*() const noexcept {
            assert(range);
            return *this;
        }

        friend bool operator==(const npy_point_iterator &a,std::default_sentinel_t) noexcept {
            assert(a.range);
            return a.range->data == nullptr;
        }
    };

    npy_point_iterator npy_range::begin() noexcept {
        return npy_point_iterator{this};
    }

    class gil_unlocker {
        PyThreadState *state;
    public:
        gil_unlocker() noexcept : state(PyEval_SaveThread()) {}
        ~gil_unlocker() { PyEval_RestoreThread(state); }
    };

    class cond_gil_unlocker {
        PyThreadState *state;
    public:
        cond_gil_unlocker(bool unlock) noexcept : state(unlock ? PyEval_SaveThread() : nullptr) {}
        ~cond_gil_unlocker() { if(state) PyEval_RestoreThread(state); }
    };

    inline int _casting_converter(PyObject *obj,NPY_CASTING *casting) noexcept {
        /* PyArray_CastingConverter will leave "casting" unchanged if "obj" is
        None */
        *casting = NPY_SAME_KIND_CASTING;
        return PyArray_CastingConverter(obj,casting);
    }
    """
    const int NPY_FAIL
    const int COORD_T_NPY

    int _casting_converter(object,np.NPY_CASTING*) except NPY_FAIL

cdef inline to_array(x):
    cdef np.ndarray ar = <np.ndarray>np.PyArray_FROM_O(x)
    if np.PyArray_NDIM(ar) != 2 or np.PyArray_DIMS(ar)[1] != 2:
        raise TypeError('input array must have two dimensions, with the second dimension having a size of two')
    return ar
