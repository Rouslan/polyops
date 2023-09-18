#ifndef POLY_OPS_SWEEP_SET_HPP
#define POLY_OPS_SWEEP_SET_HPP

#include <iterator>
#include <vector>
#include <limits>
#include <cassert>
#include <concepts>

#include "rbtree_algorithms.hpp"


namespace poly_ops::detail {

enum class color_t {black,red};

template<typename T,typename Index> struct set_node {
    Index parent
#ifndef NDEBUG
        = std::numeric_limits<Index>::max();
#endif
    Index left;
    Index right;
    color_t color;

    T value;

    set_node() = default;
    set_node(T &&value) : value(std::forward<T>(value)) {}
};

template<typename T,typename Element>
struct offset_ptr_base {
    Element &operator*() const {
        return *static_cast<const T*>(this)->get();
    }
    Element *operator->() const {
        return static_cast<const T*>(this)->get();
    }
};

template<typename T>
struct offset_ptr_base<T,void> {};

template<typename T,typename Index,typename Vec> struct offset_ptr : offset_ptr_base<offset_ptr<T,Index,Vec>,T> {
    using element_type = T;

    static constexpr Index NIL = std::numeric_limits<Index>::max();

    Vec *vec;
    Index i;

    offset_ptr() noexcept : vec(nullptr), i(NIL) {}
    offset_ptr(Vec *vec,Index i) noexcept : vec(vec), i(i) {}
    template<typename U> requires std::is_convertible_v<U*,T*>
    offset_ptr(offset_ptr<U,Index,Vec> b) noexcept : vec(b.vec), i(b.i)  {}

    template<typename U> requires std::is_convertible_v<U*,T*>
    offset_ptr &operator=(offset_ptr<U,Index,Vec> b) noexcept {
        vec = b.vec;
        i = b.i;
        return *this;
    }

    explicit operator bool() const noexcept { return i != NIL; }

    T *get() const noexcept {
        assert(i != NIL);
        return &(*vec)[i];
    }

    friend bool operator==(offset_ptr a,offset_ptr b) noexcept {
        return a.i == b.i;
    }
    friend auto operator<=>(offset_ptr a,offset_ptr b) noexcept {
        return a.i <=> b.i;
    }

    template<typename U> using rebind = offset_ptr<U,Index,Vec>;

    template<typename UPtr>
    requires requires(UPtr::element_type *x) { const_cast<T*>(x); }
    static offset_ptr const_cast_from(const UPtr &x) noexcept {
        return {x.vec,x.i};
    }

    auto unconst() const {
        return offset_ptr<std::remove_const_t<T>,Index,Vec>(vec,i);
    }
};

template<typename Index,typename Base,typename Vec>
using set_node_ptr = offset_ptr<set_node<Base,Index>,Index,Vec>;

template<typename Index,typename Base,typename Vec>
using const_set_node_ptr = offset_ptr<const set_node<Base,Index>,Index,Vec>;

template<typename Index,typename Base,typename Vec> struct _node_traits {
    using node = set_node<Base,Index>;
    using node_ptr = offset_ptr<node,Index,Vec>;
    using const_node_ptr = offset_ptr<const node,Index,Vec>;
    using color = color_t;
    static node_ptr get_parent(const_node_ptr n) noexcept { return {n.vec,n->parent}; }
    static void set_parent(node_ptr n, node_ptr parent) noexcept { n->parent = parent.i; }
    static node_ptr get_left(const_node_ptr n) noexcept { return {n.vec,n->left}; }
    static void set_left(node_ptr n, node_ptr left) noexcept { n->left = left.i; }
    static node_ptr get_right(const_node_ptr n) noexcept { return {n.vec,n->right}; }
    static void set_right(node_ptr n, node_ptr right) noexcept { n->right = right.i; }
    static color get_color(const_node_ptr n) noexcept { return n->color; }
    static void set_color(node_ptr n, color c) noexcept { n->color = c; }
    static constexpr color black() noexcept { return color_t::black; }
    static constexpr color red() noexcept { return color_t::red; }
};

template<typename Index,typename Base,typename Vec> struct _value_traits {
    Vec *vec;

    using node_traits = _node_traits<Index,Base,Vec>;
    using node = node_traits::node;
    using node_ptr = node_traits::node_ptr;
    using const_node_ptr = node_traits::const_node_ptr;
    using value_type = node;
    using pointer = node*;
    using const_pointer = const node*;

    node_ptr to_node_ptr(value_type &value) const noexcept {
        return {vec,static_cast<Index>(&value - vec->data())};
    }
    const_node_ptr to_node_ptr(const value_type &value) const noexcept {
        return {vec,static_cast<Index>(&value - vec->data())};
    }
    node_ptr to_node_ptr(Index i) const noexcept {
        return {vec,i};
    }
    pointer to_value_ptr(node_ptr n) const noexcept { return n.operator->(); }
    const_pointer to_value_ptr(const_node_ptr n) const noexcept { return n.operator->(); }
};


template<typename T,typename Index,typename Base,typename Vec=std::pmr::vector<set_node<Base,Index>>>
class tree_iterator {
public:
    // this needs to be non-const
    using node_ptr = _node_traits<Index,Base,Vec>::node_ptr;

    using node_algo = rbtree_algorithms<_node_traits<Index,Base,Vec>>;

    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = T*;
    using reference = T&;

private:
    node_ptr ptr;

public:
    tree_iterator() = default;

    explicit tree_iterator(node_ptr nodeptr) : ptr(nodeptr) {}

    tree_iterator(const tree_iterator &other) = default;

    template<typename U> requires std::is_convertible_v<U*,T*>
    tree_iterator(const tree_iterator<U,Index,Base,Vec> &b) : ptr(b.pointed_node()) {}

    tree_iterator &operator=(const tree_iterator&) = default;

    tree_iterator &operator=(node_ptr nodeptr) {
        ptr = nodeptr;
        return *this;
    }

    node_ptr pointed_node() const noexcept { return ptr; }

    Index index() const noexcept { return ptr.i; }

    tree_iterator &operator++() {
        ptr = node_algo::next_node(ptr);
        return *this;
    }

    tree_iterator operator++(int) {
        tree_iterator result(*this);
        ptr = node_algo::next_node(ptr);
        return result;
    }

    tree_iterator& operator--() {
        ptr = node_algo::prev_node(ptr);
        return *this;
    }

    tree_iterator operator--(int) {
        tree_iterator r(*this);
        ptr = node_algo::prev_node(ptr);
        return r;
    }

    bool operator!() const { return !ptr; }

    friend bool operator==(const tree_iterator &a,const tree_iterator &b) {
        return a.ptr == b.ptr;
    }

    T &operator*() const { return *ptr; }

    T *operator->() const { return ptr.get(); }

    auto unconst() const {
        return tree_iterator<std::remove_const_t<T>,Index,Base,Vec>(ptr.unconst());
    }
};

/* A custom "set" class that doesn't store the header node of the red-black tree
in itself. Instead, it is assumed to be the first element of "store". */
template<typename T,typename Index,typename Compare,typename Vec=std::pmr::vector<set_node<T,Index>>> class sweep_set {
public:
    using element_type = set_node<T,Index>;
    using iterator = tree_iterator<element_type,Index,T,Vec>;
    using const_iterator = tree_iterator<const element_type,Index,T,Vec>;
    using value_traits = _value_traits<Index,T,Vec>;
    using node_traits = value_traits::node_traits;
    using node_ptr = value_traits::node_ptr;
    using const_node_ptr = value_traits::const_node_ptr;
    using node_algo = rbtree_algorithms<node_traits>;

    struct cmp_wrapper {
        Compare base_cmp;

        template<typename U> static const U &to_value(const U &x) {
            return x;
        }
        static const T &to_value(const element_type &x) {
            return x.value;
        }
        static const T &to_value(node_ptr x) {
            return x->value;
        }
        static const T &to_value(const_node_ptr x) {
            return x->value;
        }

        bool operator()(const auto &a,const auto &b) const {
            return base_cmp(to_value(a),to_value(b));
        }
    };

    node_ptr to_node_ptr(node_ptr x) const noexcept { return x; }
    static node_ptr to_node_ptr(const_iterator itr) noexcept { return itr.pointed_node(); }
    node_ptr to_node_ptr(element_type &value) const noexcept { return traits_val.to_node_ptr(value); }
    node_ptr to_node_ptr(Index i) const noexcept { return traits_val.to_node_ptr(i); }

    sweep_set(Vec &store,Compare &&cmp={})
        : traits_val{&store}, compare{std::forward<Compare>(cmp)}
    {
        assert(store.size());
        clear();
    }

    void clear() {
        node_algo::init_header(header_ptr());
    }

    template<typename U> iterator erase(U &&pos) noexcept {
        return _erase(to_node_ptr(pos));
    }

    template<typename U> std::pair<iterator,bool> insert(U &&pos) {
        return _insert(to_node_ptr(pos));
    }

    template<typename Pos,typename Value> iterator insert_before(Pos pos,Value value) noexcept {
        return _insert_before(to_node_ptr(pos),to_node_ptr(value));
    }

    bool empty() const noexcept {
        return node_algo::unique(header_ptr());
    }

    template<typename Key> std::size_t count(Key &&key) const {
        return node_algo::count(header_ptr(),key,compare);
    }

    iterator iterator_to(element_type &value) noexcept {
        return iterator(traits_val.to_node_ptr(value));
    }
    iterator iterator_to(Index i) noexcept {
        return iterator(traits_val.to_node_ptr(i));
    }
    const_iterator iterator_to(const element_type &value) const noexcept {
        return const_iterator(traits_val.to_node_ptr(value));
    }
    const_iterator iterator_to(Index i) const noexcept {
        return const_iterator(traits_val.to_node_ptr(i));
    }

    iterator begin() noexcept {
        return iterator(node_algo::begin_node(header_ptr()));
    }

    const_iterator begin() const noexcept { return cbegin(); }

    const_iterator cbegin() const noexcept {
        return const_iterator(node_algo::begin_node(header_ptr()));
    }

    iterator end() noexcept {
        return iterator(node_algo::end_node(header_ptr()));
    }
    const_iterator end() const noexcept { return cend(); }

    const_iterator cend() const noexcept {
        return const_iterator(node_algo::end_node(header_ptr()));
    }

    void init_node(element_type &value) const {
        node_algo::init(traits_val.to_node_ptr(value));
    }
    void init_node(Index i) const {
        node_algo::init(traits_val.to_node_ptr(i));
    }

    bool unique(const element_type &value) const {
        return node_algo::unique(traits_val.to_node_ptr(value));
    }
    bool unique(Index i) const {
        return node_algo::unique(traits_val.to_node_ptr(i));
    }

    const Compare &key_comp() const noexcept { return compare.base_cmp; }
    Compare &key_comp() noexcept { return compare.base_cmp; }
    const cmp_wrapper &node_comp() const noexcept { return compare; }

private:
    node_ptr header_ptr() noexcept { return {traits_val.vec,0}; }
    const_node_ptr header_ptr() const noexcept { return {traits_val.vec,0}; }

    iterator _erase(node_ptr ptr) noexcept {
        iterator r(ptr);
        ++r;
        node_algo::erase(header_ptr(),ptr);
#ifndef NDEBUG
        ptr->parent = node_ptr::NIL;
#endif
        return r;
    }

    std::pair<iterator,bool> _insert(node_ptr ptr) {
        assert(node_algo::unique(ptr));

        typename node_algo::insert_commit_data commit_data;
        std::pair<node_ptr,bool> ret = node_algo::insert_unique_check(
            header_ptr(),
            ptr,
            compare,
            commit_data);
        return std::pair<iterator,bool>(
            ret.second ? insert_unique_commit(ptr,commit_data) : iterator(ret.first),
            ret.second);
    }

    iterator _insert_before(node_ptr pos,node_ptr value) noexcept {
        assert(node_algo::unique(value));
        return iterator(node_algo::insert_before(header_ptr(),pos,value));
    }

    iterator insert_unique_commit(node_ptr ptr,const typename node_algo::insert_commit_data &commit_data) noexcept {
        node_algo::insert_unique_commit(header_ptr(),ptr,commit_data);
        return iterator(ptr);
    }

    _value_traits<Index,T,Vec> traits_val;
    [[no_unique_address]] cmp_wrapper compare;
};

}

#endif
