#ifndef mini_flat_set_hpp
#define mini_flat_set_hpp

#include <memory>
#include <algorithm>
#include <type_traits>
#include <iterator>
#include <vector>


#ifndef POLY_OPS_ASSERT
#include <cassert>
#define POLY_OPS_ASSERT assert
#endif

namespace poly_ops::detail {

/* We need many sets that will most likely have, at most, one item. This set
only allocates memory when its size is greater than 1. T must be a POD type.
Additionally, the allocator is not stored here, but instead needs to be passed
to most functions. */
template<typename T,typename Allocator=std::allocator<T>>
requires(std::is_trivial_v<T>)
class mini_flat_set {
    using alloc_traits = std::allocator_traits<Allocator>;

    struct data_t {
        size_t capacity;
        T items[1];
    };

    size_t _size;
    union {
        T item;
        data_t *data;
    } u;

    static size_t alloc_size(size_t n) {
        return n + (offsetof(data_t,items) + sizeof(T) - 1) / sizeof(T);
    }

    static data_t *alloc_data(Allocator &alloc,size_t n) {
        data_t *r = reinterpret_cast<data_t*>(alloc_traits::allocate(alloc,alloc_size(n)));
        r->capacity = n;
        return r;
    }
    static void dealloc_data(Allocator &alloc,data_t *d) {
        alloc_traits::deallocate(alloc,reinterpret_cast<T*>(d),alloc_size(d->capacity));
    }

    template<std::input_iterator InputIt>
    T *create_items(Allocator &alloc,InputIt first,InputIt last) {
        POLY_OPS_ASSERT(_size == 1);

        T tmp = u.item;
        u.data = alloc_data(alloc,2);
        _size = 2;
        u.data->items[0] = tmp;
        u.data->items[1] = *first;
        ++first;
        return append_items(alloc,first,last);
    }

    template<std::forward_iterator InputIt>
    T *create_items(Allocator &alloc,InputIt first,InputIt last) {
        POLY_OPS_ASSERT(_size == 1);

        auto isize = std::distance(first,last);

        T tmp = u.item;
        u.data = alloc_data(alloc,isize+1);
        u.data->items[0] = tmp;
        return std::copy(first,last,u.data->items+1);
    }

    template<std::input_iterator InputIt>
    T *append_items(Allocator &alloc,InputIt first,InputIt last) {
        POLY_OPS_ASSERT(_size > 1);

        size_t total = _size;
        for(; first != last; ++first) {
            if(u.data->capacity == total) {
                data_t *tmp = u.data;
                u.data = alloc_data(alloc,total * 2);
                std::copy(tmp->items,tmp->items+total,u.data->items);
                dealloc_data(alloc,tmp);
            }
            u.data->items[total++] = *first;
        }
        return u.data->items + total;
    }

    template<std::forward_iterator InputIt>
    T *append_items(Allocator &alloc,InputIt first,InputIt last) {
        POLY_OPS_ASSERT(_size > 1);

        auto isize = std::distance(first,last);

        size_t total = _size + isize;
        if(total > u.data->capacity) {
            data_t *tmp = u.data;
            u.data = alloc_data(alloc,total);
            std::copy(tmp->items,tmp->items+_size,u.data->items);
            dealloc_data(alloc,tmp);
        }
        return std::copy(first,last,u.data->items+_size);
    }

public:
    mini_flat_set() noexcept : _size(0) {}

    mini_flat_set(const Allocator&) noexcept : _size(0) {}

    mini_flat_set(Allocator alloc,const mini_flat_set &b) : _size(b._size) {
        if(_size == 1) u.item = b.u.item;
        else if(_size > 1) {
            u.data = alloc_data(alloc,_size);
            std::copy(b.u.data->items,b.u.data->items+_size,u.data->items);
        }
    }

    mini_flat_set(mini_flat_set &&b) noexcept : _size(b._size), u(b.u) {
        b._size = 0;
    }

    mini_flat_set(const Allocator&,mini_flat_set &&b) noexcept : _size(b._size), u(b.u) {
        b._size = 0;
    }

    mini_flat_set &operator=(const mini_flat_set &b) = delete;
    mini_flat_set &operator=(mini_flat_set &&b) = delete;

    mini_flat_set &assign(Allocator alloc,const mini_flat_set &b) {
        if(b._size <= 1) {
            if(_size > 1) dealloc_data(alloc,u.data);
            u.item = b.u.item;
        } else {
            if(_size <= 1) {
                u.data = alloc_data(alloc,b._size);
            } else if(u.data->capacity < b._size) {
                dealloc_data(alloc,u.data);
                u.data = alloc_data(alloc,b._size);
            }
            std::copy(b.u.data->items,b.u.data->items+b._size,u.data->items);
        }
        _size = b._size;
        return *this;
    }

    mini_flat_set &assign(Allocator alloc,mini_flat_set &&b) {
        destroy(alloc);
        u = b.u;
        _size = b._size;
        b._size = 0;
        return *this;
    }

    void destroy(Allocator alloc) {
        if(_size > 1) dealloc_data(alloc,u.data);
    }

    size_t size() const noexcept { return _size; }
    bool empty() const noexcept { return _size != 0; }
    T *begin() noexcept { return _size > 1 ? u.data->items : &u.item; }
    const T *begin() const noexcept { return _size > 1 ? u.data->items : &u.item; }
    T *end() noexcept { return begin() + _size; }
    const T *end() const noexcept { return begin() + _size; }

    std::pair<T*,bool> insert(Allocator alloc,T value) {
        if(_size == 0) {
            u.item = value;
            ++_size;
            return {&u.item,true};
        } else if(_size == 1) {
            if(value == u.item) return {&u.item,false};

            T tmp = u.item;
            u.data = alloc_data(alloc,2);
            size_t i1 = 0;
            size_t i2 = 1;
            if(value < tmp) std::swap(i1,i2);

            u.data->items[i1] = tmp;
            u.data->items[i2] = value;
            return {u.data->items + i2,true};
        } else {
            T* pos = std::upper_bound(u.data->items,u.data->items+_size,value);
            if(pos != u.data->items && *(pos-1) == value) return {pos-1,false};

            if(_size == u.data->capacity) {
                data_t *tmp = u.data;
                u.data = alloc_data(alloc,_size * 2);
                T *new_pos = std::copy(tmp->items,pos,u.data->items);
                *new_pos = value;
                std::copy(pos,tmp->items+_size,new_pos+1);
                ++_size;
                dealloc_data(alloc,tmp);
                return {new_pos,true};
            } else {
                std::copy_backward(pos,u.data->items+_size,u.data->items+_size+1);
                *pos = value;
                ++_size;
                return {pos,true};
            }
        }
    }

    template<std::input_iterator InputIt> void insert(Allocator alloc,InputIt first,InputIt last) {
        if(_size == 0) {
            if(first == last) return;
            u.item = *first;
            _size = 1;
            ++first;
        }
        if(_size == 1) {
            for(;;) {
                if(first == last) return;
                if(*first != u.item) break;
                ++first;
            }

            T *new_end = create_items(alloc,first,last);
            std::sort(u.data->items,new_end);
            _size = std::unique(u.data->items,new_end) - u.data->items;
        } else {
            if(first == last) return;

            T *new_end = append_items(alloc,first,last);
            std::sort(u.data->items+_size,new_end);
            std::inplace_merge(u.data->items,u.data->items+_size,new_end);
            _size = std::unique(u.data->items,new_end) - u.data->items;
        }
    }

    void merge(Allocator alloc,mini_flat_set &b) {
        if(b._size > 1 && b.u.data->capacity >= _size + b._size) swap(*this,b);
        insert(alloc,b.begin(),b.end());
    }

    friend void swap(mini_flat_set &a,mini_flat_set &b) noexcept {
        std::swap(a._size,b._size);
        std::swap(a.u,b.u);
    }
};

template<typename T,typename Allocator>
class mini_flat_set_alloc_proxy {
    mini_flat_set<T,Allocator> &data;
    Allocator alloc;

public:
    mini_flat_set_alloc_proxy(mini_flat_set<T,Allocator> &data,Allocator alloc)
        : data(data), alloc(alloc) {}

    mini_flat_set_alloc_proxy &operator=(const mini_flat_set<T,Allocator> &b) const {
        data.assign(alloc,b);
        return *this;
    }
    mini_flat_set_alloc_proxy &operator=(mini_flat_set<T,Allocator> &&b) const {
        data.assign(alloc,std::move(b));
        return *this;
    }

    mini_flat_set_alloc_proxy &operator=(const mini_flat_set_alloc_proxy &b) const {
        POLY_OPS_ASSERT(alloc == b.alloc);
        data.assign(alloc,b.data);
        return *this;
    }
    mini_flat_set_alloc_proxy &operator=(mini_flat_set_alloc_proxy &&b) {
        POLY_OPS_ASSERT(alloc == b.alloc);
        data.assign(alloc,std::move(b.data));
        return *this;
    }

    size_t size() const noexcept { return data.size(); }
    bool empty() const noexcept { return data.empty(); }
    T *begin() const noexcept { return data.begin(); }
    T *end() const noexcept { return data.end(); }

    //T *begin() noexcept { return data.begin(); }
    //T *end() noexcept { return data.end(); }

    std::pair<T*,bool> insert(T value) const {
        return data.insert(alloc,value);
    }

    template<std::input_iterator InputIt> void insert(InputIt first,InputIt last) const {
        data.insert(alloc,first,last);
    }

    void merge(const mini_flat_set_alloc_proxy &b) const {
        data.merge(alloc,b.data);
    }
};

/* An allocator for mini_flat_set that also passes an allocator to mini_flat_set
as needed instead of requiring it to store a stateful allocator. */
template<typename Alloc1,typename T2> struct two_tier_allocator_adapter : public Alloc1 {
    using alloc1_traits = std::allocator_traits<Alloc1>;
    using alloc2_t = typename alloc1_traits::template rebind_alloc<T2>;
    using alloc2_traits = std::allocator_traits<alloc2_t>;

    template<typename U> struct rebind {
        using other = two_tier_allocator_adapter<typename alloc1_traits::template rebind_alloc<U>,T2>;
    };

    [[no_unique_address]] alloc2_t alloc2;

    template<typename Alloc> two_tier_allocator_adapter(const Alloc &alloc)
        : Alloc1(alloc), alloc2(alloc) {}

    two_tier_allocator_adapter(const two_tier_allocator_adapter &b)
        : Alloc1(b), alloc2(b.alloc2) {}

    two_tier_allocator_adapter(two_tier_allocator_adapter &&b)
        : Alloc1(std::move(b)), alloc2(std::move(b.alloc2)) {}

    two_tier_allocator_adapter &operator=(const two_tier_allocator_adapter &b) {
        Alloc1::operator=(b);
        alloc2 = b.alloc2;
        return *this;
    }

    two_tier_allocator_adapter &operator=(two_tier_allocator_adapter &&b) {
        Alloc1::operator=(std::move(b));
        alloc2 = std::move(b.alloc2);
        return *this;
    }

    template<typename... Args> void construct(typename Alloc1::value_type *x,Args&&... args) {
        alloc1_traits::construct(*this,x,alloc2,std::forward<Args>(args)...);
    }

    void destroy(typename Alloc1::value_type *x) {
        x->destroy(alloc2);
        alloc1_traits::destroy(*this,x);
    }
};

template<typename T,typename Allocator> struct mini_set_proxy_vector {
    using real_value_type = mini_flat_set<T,Allocator>;
    using alloc_adapter = two_tier_allocator_adapter<
        typename std::allocator_traits<Allocator>::template rebind_alloc<real_value_type>,T>;
    using base_type = std::vector<real_value_type,alloc_adapter>;
    using proxy = mini_flat_set_alloc_proxy<T,Allocator>;

    base_type data;

    explicit mini_set_proxy_vector(const Allocator &alloc)
        : data(alloc_adapter(alloc)) {}

    /* since only the non-const functions need an allocator, we can return the
    real type in const accessors */
    const real_value_type &operator[](typename base_type::size_type i) const { return data[i]; }
    proxy operator[](typename base_type::size_type i) { return {data[i],data.get_allocator().alloc2}; }

    const real_value_type &font() const { return data.front(); }
    proxy front() { return {data.front(),data.get_allocator().alloc2}; }
    const real_value_type &back() const { return data.back(); }
    proxy back() { return {data.back(),data.get_allocator().alloc2}; }

    auto begin() const { return data.begin(); }
    auto end() const { return data.end(); }

    auto empty() const { return data.empty(); }
    auto size() const { return data.size(); }

    void reserve(typename base_type::size_type new_cap) { data.reserve(new_cap); }
    void clear() { data.clear(); }
    void emplace_back() { data.emplace_back(); } // mini_flat_set's constructor doesn't take any parameters
};

} // namespace poly_ops::detail

#endif
