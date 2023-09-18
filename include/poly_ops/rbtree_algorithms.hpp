/* This code was copied and pasted from the Boost intrusive library, then
modified to remove some of unneeded stuff. The original copyright notice is as
follows: */

/////////////////////////////////////////////////////////////////////////////
//
// (C) Copyright Olaf Krzikalla 2004-2006.
// (C) Copyright Ion Gaztanaga  2006-2014.
//
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org/libs/intrusive for documentation.
//
/////////////////////////////////////////////////////////////////////////////
//
// The tree destruction algorithm is based on Julienne Walker and The EC Team code:
//
// This code is in the public domain. Anyone may use it or change it in any way that
// they see fit. The author assumes no responsibility for damages incurred through
// use of the original code or any variations thereof.
//
// It is requested, but not required, that due credit is given to the original author
// and anyone who has modified the code through a header comment, such as this one.


#ifndef POLY_OPS_RB_TREE_ALGORITHMS_HPP
#define POLY_OPS_RB_TREE_ALGORITHMS_HPP

#include <cassert>
#include <bit>

namespace poly_ops::detail {

inline std::size_t floor_log2(std::size_t n) {
   return sizeof(std::size_t)*8 - std::size_t(1) - std::size_t(std::countl_zero(n));
}

template <class NodePtr>
struct insert_commit_data_t
{
   insert_commit_data_t()
      : link_left(false), node()
   {}
   bool     link_left;
   NodePtr  node;
};

template <class NodePtr>
struct data_for_rebalance_t
{
   NodePtr  x;
   NodePtr  x_parent;
   NodePtr  y;
};


template<class NodeTraits>
class bstree_algorithms_base
{
public:
   typedef typename NodeTraits::node            node;
   typedef NodeTraits                           node_traits;
   typedef typename NodeTraits::node_ptr        node_ptr;
   typedef typename NodeTraits::const_node_ptr  const_node_ptr;

   static node_ptr next_node(node_ptr n) noexcept
   {
      node_ptr const n_right(NodeTraits::get_right(n));
      if(n_right){
         return minimum(n_right);
      }
      else {
         node_ptr p(NodeTraits::get_parent(n));
         while(n == NodeTraits::get_right(p)){
            n = p;
            p = NodeTraits::get_parent(p);
         }
         return NodeTraits::get_right(n) != p ? p : n;
      }
   }

   static node_ptr prev_node(node_ptr n) noexcept
   {
      if(is_header(n)){
         return NodeTraits::get_right(n);
      }
      else if(NodeTraits::get_left(n)){
         return maximum(NodeTraits::get_left(n));
      }
      else {
         node_ptr p(n);
         node_ptr x = NodeTraits::get_parent(p);
         while(p == NodeTraits::get_left(x)){
            p = x;
            x = NodeTraits::get_parent(x);
         }
         return x;
      }
   }

   static node_ptr minimum(node_ptr n)
   {
      for(node_ptr p_left = NodeTraits::get_left(n)
         ;p_left
         ;p_left = NodeTraits::get_left(n)){
         n = p_left;
      }
      return n;
   }

   static node_ptr maximum(node_ptr n)
   {
      for(node_ptr p_right = NodeTraits::get_right(n)
         ;p_right
         ;p_right = NodeTraits::get_right(n)){
         n = p_right;
      }
      return n;
   }

   static bool is_header(const_node_ptr p) noexcept
   {
      node_ptr p_left (NodeTraits::get_left(p));
      node_ptr p_right(NodeTraits::get_right(p));
      if(!NodeTraits::get_parent(p) || //Header condition when empty tree
         (p_left && p_right &&         //Header always has leftmost and rightmost
            (p_left == p_right ||      //Header condition when only node
               (NodeTraits::get_parent(p_left)  != p ||
                NodeTraits::get_parent(p_right) != p ))
               //When tree size > 1 headers can't be leftmost's
               //and rightmost's parent
          )){
         return true;
      }
      return false;
   }

   static node_ptr get_header(const_node_ptr n)
   {
      node_ptr nn(n.unconst());
      node_ptr p(NodeTraits::get_parent(n));
      //If p is null, then nn is the header of an empty tree
      if(p){
         //Non-empty tree, check if nn is neither root nor header
         node_ptr pp(NodeTraits::get_parent(p));
         //If granparent is not equal to nn, then nn is neither root nor header,
         //the try the fast path
         if(nn != pp){
            do{
               nn = p;
               p = pp;
               pp = NodeTraits::get_parent(pp);
            }while(nn != pp);
            nn = p;
         }
         //Check if nn is root or header when size() > 0
         else if(!bstree_algorithms_base::is_header(nn)){
            nn = p;
         }
      }
      return nn;
   }
};

template<class NodeTraits>
class bstree_algorithms : public bstree_algorithms_base<NodeTraits>
{
public:
   typedef typename NodeTraits::node            node;
   typedef NodeTraits                           node_traits;
   typedef typename NodeTraits::node_ptr        node_ptr;
   typedef typename NodeTraits::const_node_ptr  const_node_ptr;
   typedef insert_commit_data_t<node_ptr>       insert_commit_data;
   typedef data_for_rebalance_t<node_ptr>       data_for_rebalance;

   typedef bstree_algorithms<NodeTraits>        this_type;
   typedef bstree_algorithms_base<NodeTraits>   base_type;

public:
   static node_ptr begin_node(const_node_ptr header) noexcept
   {  return node_traits::get_left(header);   }

   static node_ptr end_node(const_node_ptr header) noexcept
   {  return header.unconst();   }

   static node_ptr root_node(const_node_ptr header) noexcept
   {
      node_ptr p = node_traits::get_parent(header);
      return p ? p : header.unconst();
   }

   static bool unique(const_node_ptr n) noexcept
   { return !NodeTraits::get_parent(n); }

   static void init(node_ptr n) noexcept
   {
      NodeTraits::set_parent(n, node_ptr());
      NodeTraits::set_left(n, node_ptr());
      NodeTraits::set_right(n, node_ptr());
   }

   static bool inited(const_node_ptr n)
   {
      return !NodeTraits::get_parent(n) &&
             !NodeTraits::get_left(n)   &&
             !NodeTraits::get_right(n)  ;
   }

   static void init_header(node_ptr header) noexcept
   {
      NodeTraits::set_parent(header, node_ptr());
      NodeTraits::set_left(header, header);
      NodeTraits::set_right(header, header);
   }

   static node_ptr unlink_leftmost_without_rebalance(node_ptr header) noexcept
   {
      node_ptr leftmost = NodeTraits::get_left(header);
      if (leftmost == header)
         return node_ptr();
      node_ptr leftmost_parent(NodeTraits::get_parent(leftmost));
      node_ptr leftmost_right (NodeTraits::get_right(leftmost));
      bool is_root = leftmost_parent == header;

      if (leftmost_right){
         NodeTraits::set_parent(leftmost_right, leftmost_parent);
         NodeTraits::set_left(header, base_type::minimum(leftmost_right));

         if (is_root)
            NodeTraits::set_parent(header, leftmost_right);
         else
            NodeTraits::set_left(NodeTraits::get_parent(header), leftmost_right);
      }
      else if (is_root){
         NodeTraits::set_parent(header, node_ptr());
         NodeTraits::set_left(header,  header);
         NodeTraits::set_right(header, header);
      }
      else{
         NodeTraits::set_left(leftmost_parent, node_ptr());
         NodeTraits::set_left(header, leftmost_parent);
      }
      return leftmost;
   }

   static std::size_t size(const_node_ptr header) noexcept
   {
      node_ptr beg(begin_node(header));
      node_ptr end(end_node(header));
      std::size_t i = 0;
      for(;beg != end; beg = base_type::next_node(beg)) ++i;
      return i;
   }

   template<class KeyType, class KeyNodePtrCompare>
   static node_ptr find
      (const_node_ptr header, const KeyType &key, KeyNodePtrCompare comp)
   {
      node_ptr end = header.unconst();
      node_ptr y = lower_bound(header, key, comp);
      return (y == end || comp(key, y)) ? end : y;
   }

   template< class KeyType, class KeyNodePtrCompare>
   static std::pair<node_ptr, node_ptr> bounded_range
      ( const_node_ptr header
      , const KeyType &lower_key
      , const KeyType &upper_key
      , KeyNodePtrCompare comp
      , bool left_closed
      , bool right_closed)
   {
      node_ptr y = header.unconst();
      node_ptr x = NodeTraits::get_parent(header);

      while(x){
         //If x is less than lower_key the target
         //range is on the right part
         if(comp(x, lower_key)){
            //Check for invalid input range
            assert(comp(x, upper_key));
            x = NodeTraits::get_right(x);
         }
         //If the upper_key is less than x, the target
         //range is on the left part
         else if(comp(upper_key, x)){
            y = x;
            x = NodeTraits::get_left(x);
         }
         else{
            //x is inside the bounded range(lower_key <= x <= upper_key),
            //so we must split lower and upper searches
            //
            //Sanity check: if lower_key and upper_key are equal, then both left_closed and right_closed can't be false
            assert(left_closed || right_closed || comp(lower_key, x) || comp(x, upper_key));
            return std::pair<node_ptr,node_ptr>(
               left_closed
                  //If left_closed, then comp(x, lower_key) is already the lower_bound
                  //condition so we save one comparison and go to the next level
                  //following traditional lower_bound algo
                  ? lower_bound_loop(NodeTraits::get_left(x), x, lower_key, comp)
                  //If left-open, comp(x, lower_key) is not the upper_bound algo
                  //condition so we must recheck current 'x' node with upper_bound algo
                  : upper_bound_loop(x, y, lower_key, comp)
            ,
               right_closed
                  //If right_closed, then comp(upper_key, x) is already the upper_bound
                  //condition so we can save one comparison and go to the next level
                  //following lower_bound algo
                  ? upper_bound_loop(NodeTraits::get_right(x), y, upper_key, comp)
                  //If right-open, comp(upper_key, x) is not the lower_bound algo
                  //condition so we must recheck current 'x' node with lower_bound algo
                  : lower_bound_loop(x, y, upper_key, comp)
            );
         }
      }
      return std::pair<node_ptr,node_ptr> (y, y);
   }

   template<class KeyType, class KeyNodePtrCompare>
   static std::size_t count
      (const_node_ptr header, const KeyType &key, KeyNodePtrCompare comp)
   {
      std::pair<node_ptr, node_ptr> ret = equal_range(header, key, comp);
      std::size_t n = 0;
      while(ret.first != ret.second){
         ++n;
         ret.first = base_type::next_node(ret.first);
      }
      return n;
   }

   template<class KeyType, class KeyNodePtrCompare>
   static std::pair<node_ptr, node_ptr> equal_range
      (const_node_ptr header, const KeyType &key, KeyNodePtrCompare comp)
   {
      return bounded_range(header, key, key, comp, true, true);
   }

   template<class KeyType, class KeyNodePtrCompare>
   static std::pair<node_ptr, node_ptr> lower_bound_range
      (const_node_ptr header, const KeyType &key, KeyNodePtrCompare comp)
   {
      node_ptr const lb(lower_bound(header, key, comp));
      std::pair<node_ptr, node_ptr> ret_ii(lb, lb);
      if(lb != header && !comp(key, lb)){
         ret_ii.second = base_type::next_node(ret_ii.second);
      }
      return ret_ii;
   }

   template<class KeyType, class KeyNodePtrCompare>
   static node_ptr lower_bound
      (const_node_ptr header, const KeyType &key, KeyNodePtrCompare comp)
   {
      return lower_bound_loop(NodeTraits::get_parent(header), header.unconst(), key, comp);
   }

   template<class KeyType, class KeyNodePtrCompare>
   static node_ptr upper_bound
      (const_node_ptr header, const KeyType &key, KeyNodePtrCompare comp)
   {
      return upper_bound_loop(NodeTraits::get_parent(header), header.unconst(), key, comp);
   }

   static void insert_unique_commit
      (node_ptr header, node_ptr new_value, const insert_commit_data &commit_data) noexcept
   {  return insert_commit(header, new_value, commit_data); }

   template<class KeyType, class KeyNodePtrCompare>
   static std::pair<node_ptr, bool> insert_unique_check
      (const_node_ptr header, const KeyType &key
      ,KeyNodePtrCompare comp, insert_commit_data &commit_data
         , std::size_t *pdepth = 0
      )
   {
      std::size_t depth = 0;
      node_ptr h(header.unconst());
      node_ptr y(h);
      node_ptr x(NodeTraits::get_parent(y));
      node_ptr prev = node_ptr();

      //Find the upper bound, cache the previous value and if we should
      //store it in the left or right node
      bool left_child = true;
      while(x){
         ++depth;
         y = x;
         left_child = comp(key, x);
         x = left_child ?
               NodeTraits::get_left(x) : (prev = y, NodeTraits::get_right(x));
      }

      if(pdepth)  *pdepth = depth;

      //Since we've found the upper bound there is no other value with the same key if:
      //    - There is no previous node
      //    - The previous node is less than the key
      const bool not_present = !prev || comp(prev, key);
      if(not_present){
         commit_data.link_left = left_child;
         commit_data.node      = y;
      }
      return std::pair<node_ptr, bool>(prev, not_present);
   }

   template<class KeyType, class KeyNodePtrCompare>
   static std::pair<node_ptr, bool> insert_unique_check
      (const_node_ptr header, node_ptr hint, const KeyType &key
      ,KeyNodePtrCompare comp, insert_commit_data &commit_data
         , std::size_t *pdepth = 0
      )
   {
      //hint must be bigger than the key
      if(hint == header || comp(key, hint)){
         node_ptr prev(hint);
         //Previous value should be less than the key
         if(hint == begin_node(header) || comp((prev = base_type::prev_node(hint)), key)){
            commit_data.link_left = unique(header) || !NodeTraits::get_left(hint);
            commit_data.node      = commit_data.link_left ? hint : prev;
            if(pdepth){
               *pdepth = commit_data.node == header ? 0 : depth(commit_data.node) + 1;
            }
            return std::pair<node_ptr, bool>(node_ptr(), true);
         }
      }
      //Hint was wrong, use hintless insertion
      return insert_unique_check(header, key, comp, commit_data, pdepth);
   }

   template<class NodePtrCompare>
   static node_ptr insert_equal
      (node_ptr h, node_ptr hint, node_ptr new_node, NodePtrCompare comp
         , std::size_t *pdepth = 0
      )
   {
      insert_commit_data commit_data;
      insert_equal_check(h, hint, new_node, comp, commit_data, pdepth);
      insert_commit(h, new_node, commit_data);
      return new_node;
   }


   template<class NodePtrCompare>
   static node_ptr insert_equal_upper_bound
      (node_ptr h, node_ptr new_node, NodePtrCompare comp
         , std::size_t *pdepth = 0
      )
   {
      insert_commit_data commit_data;
      insert_equal_upper_bound_check(h, new_node, comp, commit_data, pdepth);
      insert_commit(h, new_node, commit_data);
      return new_node;
   }

   template<class NodePtrCompare>
   static node_ptr insert_equal_lower_bound
      (node_ptr h, node_ptr new_node, NodePtrCompare comp
         , std::size_t *pdepth = 0
      )
   {
      insert_commit_data commit_data;
      insert_equal_lower_bound_check(h, new_node, comp, commit_data, pdepth);
      insert_commit(h, new_node, commit_data);
      return new_node;
   }

   static node_ptr insert_before
      (node_ptr header, node_ptr pos, node_ptr new_node
         , std::size_t *pdepth = 0
      ) noexcept
   {
      insert_commit_data commit_data;
      insert_before_check(header, pos, commit_data, pdepth);
      insert_commit(header, new_node, commit_data);
      return new_node;
   }

   static void push_back
      (node_ptr header, node_ptr new_node
         , std::size_t *pdepth = 0
      ) noexcept
   {
      insert_commit_data commit_data;
      push_back_check(header, commit_data, pdepth);
      insert_commit(header, new_node, commit_data);
   }

   static void push_front
      (node_ptr header, node_ptr new_node
         , std::size_t *pdepth = 0
      ) noexcept
   {
      insert_commit_data commit_data;
      push_front_check(header, commit_data, pdepth);
      insert_commit(header, new_node, commit_data);
   }

   static std::size_t depth(const_node_ptr n) noexcept
   {
      std::size_t depth = 0;
      node_ptr p_parent;
      while(n != NodeTraits::get_parent(p_parent = NodeTraits::get_parent(n))){
         ++depth;
         n = p_parent;
      }
      return depth;
   }

   static void erase(node_ptr header, node_ptr z) noexcept
   {
      data_for_rebalance ignored;
      erase(header, z, ignored);
   }

   static void unlink(node_ptr n) noexcept
   {
      node_ptr x = NodeTraits::get_parent(n);
      if(x){
         while(!base_type::is_header(x))
            x = NodeTraits::get_parent(x);
         erase(x, n);
      }
   }

   static void rebalance(node_ptr header) noexcept
   {
      node_ptr root = NodeTraits::get_parent(header);
      if(root){
         rebalance_subtree(root);
      }
   }

   static node_ptr rebalance_subtree(node_ptr old_root) noexcept
   {
      //Taken from:
      //"Tree rebalancing in optimal time and space"
      //Quentin F. Stout and Bette L. Warren

      //To avoid irregularities in the algorithm (old_root can be a
      //left or right child or even the root of the tree) just put the
      //root as the right child of its parent. Before doing this backup
      //information to restore the original relationship after
      //the algorithm is applied.
      node_ptr super_root = NodeTraits::get_parent(old_root);
      assert(super_root);

      //Get root info
      node_ptr super_root_right_backup = NodeTraits::get_right(super_root);
      bool super_root_is_header = NodeTraits::get_parent(super_root) == old_root;
      bool old_root_is_right  = is_right_child(old_root);
      NodeTraits::set_right(super_root, old_root);

      std::size_t size;
      subtree_to_vine(super_root, size);
      vine_to_subtree(super_root, size);
      node_ptr new_root = NodeTraits::get_right(super_root);

      //Recover root
      if(super_root_is_header){
         NodeTraits::set_right(super_root, super_root_right_backup);
         NodeTraits::set_parent(super_root, new_root);
      }
      else if(old_root_is_right){
         NodeTraits::set_right(super_root, new_root);
      }
      else{
         NodeTraits::set_right(super_root, super_root_right_backup);
         NodeTraits::set_left(super_root, new_root);
      }
      return new_root;
   }

protected:

   static void erase(node_ptr header, node_ptr z, data_for_rebalance &info)
   {
      node_ptr y(z);
      node_ptr x;
      const node_ptr z_left(NodeTraits::get_left(z));
      const node_ptr z_right(NodeTraits::get_right(z));

      if(!z_left){
         x = z_right;    // x might be null.
      }
      else if(!z_right){ // z has exactly one non-null child. y == z.
         x = z_left;       // x is not null.
         assert(x);
      }
      else{ //make y != z
         // y = find z's successor
         y = base_type::minimum(z_right);
         x = NodeTraits::get_right(y);     // x might be null.
      }

      node_ptr x_parent;
      const node_ptr z_parent(NodeTraits::get_parent(z));
      const bool z_is_leftchild(NodeTraits::get_left(z_parent) == z);

      if(y != z){ //has two children and y is the minimum of z
         //y is z's successor and it has a null left child.
         //x is the right child of y (it can be null)
         //Relink y in place of z and link x with y's old parent
         NodeTraits::set_parent(z_left, y);
         NodeTraits::set_left(y, z_left);
         if(y != z_right){
            //Link y with the right tree of z
            NodeTraits::set_right(y, z_right);
            NodeTraits::set_parent(z_right, y);
            //Link x with y's old parent (y must be a left child)
            x_parent = NodeTraits::get_parent(y);
            assert(NodeTraits::get_left(x_parent) == y);
            if(x)
               NodeTraits::set_parent(x, x_parent);
            //Since y was the successor and not the right child of z, it must be a left child
            NodeTraits::set_left(x_parent, x);
         }
         else{ //y was the right child of y so no need to fix x's position
            x_parent = y;
         }
         NodeTraits::set_parent(y, z_parent);
         this_type::set_child(header, y, z_parent, z_is_leftchild);
      }
      else {  // z has zero or one child, x is one child (it can be null)
         //Just link x to z's parent
         x_parent = z_parent;
         if(x)
            NodeTraits::set_parent(x, z_parent);
         this_type::set_child(header, x, z_parent, z_is_leftchild);

         //Now update leftmost/rightmost in case z was one of them
         if(NodeTraits::get_left(header) == z){
            //z_left must be null because z is the leftmost
            assert(!z_left);
            NodeTraits::set_left(header, !z_right ?
               z_parent :  // makes leftmost == header if z == root
               base_type::minimum(z_right));
         }
         if(NodeTraits::get_right(header) == z){
            //z_right must be null because z is the rightmost
            assert(!z_right);
            NodeTraits::set_right(header, !z_left ?
               z_parent :  // makes rightmost == header if z == root
               base_type::maximum(z_left));
         }
      }

      //If z had 0/1 child, y == z and one of its children (and maybe null)
      //If z had 2 children, y is the successor of z and x is the right child of y
      info.x = x;
      info.y = y;
      //If z had 0/1 child, x_parent is the new parent of the old right child of y (z's successor)
      //If z had 2 children, x_parent is the new parent of y (z_parent)
      assert(!x || NodeTraits::get_parent(x) == x_parent);
      info.x_parent = x_parent;
   }

   static std::size_t subtree_size(const_node_ptr subtree) noexcept
   {
      std::size_t count = 0;
      if (subtree){
         node_ptr n = subtree.unconst();
         node_ptr m = NodeTraits::get_left(n);
         while(m){
            n = m;
            m = NodeTraits::get_left(n);
         }

         while(1){
            ++count;
            node_ptr n_right(NodeTraits::get_right(n));
            if(n_right){
               n = n_right;
               m = NodeTraits::get_left(n);
               while(m){
                  n = m;
                  m = NodeTraits::get_left(n);
               }
            }
            else {
               do{
                  if (n == subtree){
                     return count;
                  }
                  m = n;
                  n = NodeTraits::get_parent(n);
               }while(NodeTraits::get_left(n) != m);
            }
         }
      }
      return count;
   }

   static bool is_left_child(node_ptr p) noexcept
   {  return NodeTraits::get_left(NodeTraits::get_parent(p)) == p;  }

   static bool is_right_child(node_ptr p) noexcept
   {  return NodeTraits::get_right(NodeTraits::get_parent(p)) == p;  }

   static void insert_before_check
      (node_ptr header, node_ptr pos
      , insert_commit_data &commit_data
         , std::size_t *pdepth = 0
      )
   {
      node_ptr prev(pos);
      if(pos != NodeTraits::get_left(header))
         prev = base_type::prev_node(pos);
      bool link_left = unique(header) || !NodeTraits::get_left(pos);
      commit_data.link_left = link_left;
      commit_data.node = link_left ? pos : prev;
      if(pdepth){
         *pdepth = commit_data.node == header ? 0 : depth(commit_data.node) + 1;
      }
   }

   static void push_back_check
      (node_ptr header, insert_commit_data &commit_data
         , std::size_t *pdepth = 0
      ) noexcept
   {
      node_ptr prev(NodeTraits::get_right(header));
      if(pdepth){
         *pdepth = prev == header ? 0 : depth(prev) + 1;
      }
      commit_data.link_left = false;
      commit_data.node = prev;
   }

   static void push_front_check
      (node_ptr header, insert_commit_data &commit_data
         , std::size_t *pdepth = 0
      ) noexcept
   {
      node_ptr pos(NodeTraits::get_left(header));
      if(pdepth){
         *pdepth = pos == header ? 0 : depth(pos) + 1;
      }
      commit_data.link_left = true;
      commit_data.node = pos;
   }

   template<class NodePtrCompare>
   static void insert_equal_check
      (node_ptr header, node_ptr hint, node_ptr new_node, NodePtrCompare comp
      , insert_commit_data &commit_data
      , std::size_t *pdepth = 0
      )
   {
      if(hint == header || !comp(hint, new_node)){
         node_ptr prev(hint);
         if(hint == NodeTraits::get_left(header) ||
            !comp(new_node, (prev = base_type::prev_node(hint)))){
            bool link_left = unique(header) || !NodeTraits::get_left(hint);
            commit_data.link_left = link_left;
            commit_data.node = link_left ? hint : prev;
            if(pdepth){
               *pdepth = commit_data.node == header ? 0 : depth(commit_data.node) + 1;
            }
         }
         else{
            insert_equal_upper_bound_check(header, new_node, comp, commit_data, pdepth);
         }
      }
      else{
         insert_equal_lower_bound_check(header, new_node, comp, commit_data, pdepth);
      }
   }

   template<class NodePtrCompare>
   static void insert_equal_upper_bound_check
      (node_ptr h, node_ptr new_node, NodePtrCompare comp, insert_commit_data & commit_data, std::size_t *pdepth = 0)
   {
      std::size_t depth = 0;
      node_ptr y(h);
      node_ptr x(NodeTraits::get_parent(y));

      while(x){
         ++depth;
         y = x;
         x = comp(new_node, x) ?
               NodeTraits::get_left(x) : NodeTraits::get_right(x);
      }
      if(pdepth)  *pdepth = depth;
      commit_data.link_left = (y == h) || comp(new_node, y);
      commit_data.node = y;
   }

   template<class NodePtrCompare>
   static void insert_equal_lower_bound_check
      (node_ptr h, node_ptr new_node, NodePtrCompare comp, insert_commit_data & commit_data, std::size_t *pdepth = 0)
   {
      std::size_t depth = 0;
      node_ptr y(h);
      node_ptr x(NodeTraits::get_parent(y));

      while(x){
         ++depth;
         y = x;
         x = !comp(x, new_node) ?
               NodeTraits::get_left(x) : NodeTraits::get_right(x);
      }
      if(pdepth)  *pdepth = depth;
      commit_data.link_left = (y == h) || !comp(y, new_node);
      commit_data.node = y;
   }

   static void insert_commit
      (node_ptr header, node_ptr new_node, const insert_commit_data &commit_data) noexcept
   {
      //Check if commit_data has not been initialized by a insert_unique_check call.
      assert(commit_data.node != node_ptr());
      node_ptr parent_node(commit_data.node);
      if(parent_node == header){
         NodeTraits::set_parent(header, new_node);
         NodeTraits::set_right(header, new_node);
         NodeTraits::set_left(header, new_node);
      }
      else if(commit_data.link_left){
         NodeTraits::set_left(parent_node, new_node);
         if(parent_node == NodeTraits::get_left(header))
             NodeTraits::set_left(header, new_node);
      }
      else{
         NodeTraits::set_right(parent_node, new_node);
         if(parent_node == NodeTraits::get_right(header))
             NodeTraits::set_right(header, new_node);
      }
      NodeTraits::set_parent(new_node, parent_node);
      NodeTraits::set_right(new_node, node_ptr());
      NodeTraits::set_left(new_node, node_ptr());
   }

   static void set_child(node_ptr header, node_ptr new_child, node_ptr new_parent, const bool link_left) noexcept
   {
      if(new_parent == header)
         NodeTraits::set_parent(header, new_child);
      else if(link_left)
         NodeTraits::set_left(new_parent, new_child);
      else
         NodeTraits::set_right(new_parent, new_child);
   }

   static void rotate_left_no_parent_fix(node_ptr p, node_ptr p_right) noexcept
   {
      node_ptr p_right_left(NodeTraits::get_left(p_right));
      NodeTraits::set_right(p, p_right_left);
      if(p_right_left){
         NodeTraits::set_parent(p_right_left, p);
      }
      NodeTraits::set_left(p_right, p);
      NodeTraits::set_parent(p, p_right);
   }

   static void rotate_left(node_ptr p, node_ptr p_right, node_ptr p_parent, node_ptr header) noexcept
   {
      const bool p_was_left(NodeTraits::get_left(p_parent) == p);
      rotate_left_no_parent_fix(p, p_right);
      NodeTraits::set_parent(p_right, p_parent);
      set_child(header, p_right, p_parent, p_was_left);
   }

   static void rotate_right_no_parent_fix(node_ptr p, node_ptr p_left) noexcept
   {
      node_ptr p_left_right(NodeTraits::get_right(p_left));
      NodeTraits::set_left(p, p_left_right);
      if(p_left_right){
         NodeTraits::set_parent(p_left_right, p);
      }
      NodeTraits::set_right(p_left, p);
      NodeTraits::set_parent(p, p_left);
   }

   static void rotate_right(node_ptr p, node_ptr p_left, node_ptr p_parent, node_ptr header) noexcept
   {
      const bool p_was_left(NodeTraits::get_left(p_parent) == p);
      rotate_right_no_parent_fix(p, p_left);
      NodeTraits::set_parent(p_left, p_parent);
      set_child(header, p_left, p_parent, p_was_left);
   }

private:

   static void subtree_to_vine(node_ptr vine_tail, std::size_t &size) noexcept
   {
      //Inspired by LibAVL:
      //It uses a clever optimization for trees with parent pointers.
      //No parent pointer is updated when transforming a tree to a vine as
      //most of them will be overriten during compression rotations.
      //A final pass must be made after the rebalancing to updated those
      //pointers not updated by tree_to_vine + compression calls
      std::size_t len = 0;
      node_ptr remainder = NodeTraits::get_right(vine_tail);
      while(remainder){
         node_ptr tempptr = NodeTraits::get_left(remainder);
         if(!tempptr){   //move vine-tail down one
            vine_tail = remainder;
            remainder = NodeTraits::get_right(remainder);
            ++len;
         }
         else{ //rotate
            NodeTraits::set_left(remainder, NodeTraits::get_right(tempptr));
            NodeTraits::set_right(tempptr, remainder);
            remainder = tempptr;
            NodeTraits::set_right(vine_tail, tempptr);
         }
      }
      size = len;
   }

   static void compress_subtree(node_ptr scanner, std::size_t count) noexcept
   {
      while(count--){   //compress "count" spine nodes in the tree with pseudo-root scanner
         node_ptr child = NodeTraits::get_right(scanner);
         node_ptr child_right = NodeTraits::get_right(child);
         NodeTraits::set_right(scanner, child_right);
         //Avoid setting the parent of child_right
         scanner = child_right;
         node_ptr scanner_left = NodeTraits::get_left(scanner);
         NodeTraits::set_right(child, scanner_left);
         if(scanner_left)
            NodeTraits::set_parent(scanner_left, child);
         NodeTraits::set_left(scanner, child);
         NodeTraits::set_parent(child, scanner);
      }
   }

   static void vine_to_subtree(node_ptr super_root, std::size_t count) noexcept
   {
      const std::size_t one_szt = 1u;
      std::size_t leaf_nodes = count + one_szt - std::size_t(one_szt << floor_log2(count + one_szt));
      compress_subtree(super_root, leaf_nodes);  //create deepest leaves
      std::size_t vine_nodes = count - leaf_nodes;
      while(vine_nodes > 1){
         vine_nodes /= 2;
         compress_subtree(super_root, vine_nodes);
      }

      //Update parents of nodes still in the in the original vine line
      //as those have not been updated by subtree_to_vine or compress_subtree
      for ( node_ptr q = super_root, p = NodeTraits::get_right(super_root)
          ; p
          ; q = p, p = NodeTraits::get_right(p)){
         NodeTraits::set_parent(p, q);
      }
   }

   static node_ptr get_root(node_ptr n) noexcept
   {
      assert((!inited(n)));
      node_ptr x = NodeTraits::get_parent(n);
      if(x){
         while(!base_type::is_header(x)){
            x = NodeTraits::get_parent(x);
         }
         return x;
      }
      else{
         return n;
      }
   }

   template<class KeyType, class KeyNodePtrCompare>
   static node_ptr lower_bound_loop
      (node_ptr x, node_ptr y, const KeyType &key, KeyNodePtrCompare comp)
   {
      while(x){
         if(comp(x, key)){
            x = NodeTraits::get_right(x);
         }
         else{
            y = x;
            x = NodeTraits::get_left(x);
         }
      }
      return y;
   }

   template<class KeyType, class KeyNodePtrCompare>
   static node_ptr upper_bound_loop
      (node_ptr x, node_ptr y, const KeyType &key, KeyNodePtrCompare comp)
   {
      while(x){
         if(comp(key, x)){
            y = x;
            x = NodeTraits::get_left(x);
         }
         else{
            x = NodeTraits::get_right(x);
         }
      }
      return y;
   }
};

template<class NodeTraits>
class rbtree_algorithms
   : public bstree_algorithms<NodeTraits>
{
public:
   typedef NodeTraits                           node_traits;
   typedef typename NodeTraits::node            node;
   typedef typename NodeTraits::node_ptr        node_ptr;
   typedef typename NodeTraits::const_node_ptr  const_node_ptr;
   typedef typename NodeTraits::color           color;

private:
   typedef bstree_algorithms<NodeTraits>  bstree_algo;

public:
   typedef typename bstree_algo::insert_commit_data insert_commit_data;

   static void replace_node(node_ptr node_to_be_replaced, node_ptr new_node) noexcept
   {
      if(node_to_be_replaced == new_node)
         return;
      replace_node(node_to_be_replaced, bstree_algo::get_header(node_to_be_replaced), new_node);
   }

   static void replace_node(node_ptr node_to_be_replaced, node_ptr header, node_ptr new_node) noexcept
   {
      bstree_algo::replace_node(node_to_be_replaced, header, new_node);
      NodeTraits::set_color(new_node, NodeTraits::get_color(node_to_be_replaced));
   }

   static void unlink(node_ptr n) noexcept
   {
      node_ptr x = NodeTraits::get_parent(n);
      if(x){
         while(!is_header(x))
            x = NodeTraits::get_parent(x);
         erase(x, n);
      }
   }

   static void init_header(node_ptr header) noexcept
   {
      bstree_algo::init_header(header);
      NodeTraits::set_color(header, NodeTraits::red());
   }

   static node_ptr erase(node_ptr header, node_ptr z) noexcept
   {
      typename bstree_algo::data_for_rebalance info;
      bstree_algo::erase(header, z, info);
      rebalance_after_erasure(header, z, info);
      return z;
   }

   template<class NodePtrCompare>
   static node_ptr insert_equal_upper_bound
      (node_ptr h, node_ptr new_node, NodePtrCompare comp)
   {
      bstree_algo::insert_equal_upper_bound(h, new_node, comp);
      rebalance_after_insertion(h, new_node);
      return new_node;
   }

   template<class NodePtrCompare>
   static node_ptr insert_equal_lower_bound
      (node_ptr h, node_ptr new_node, NodePtrCompare comp)
   {
      bstree_algo::insert_equal_lower_bound(h, new_node, comp);
      rebalance_after_insertion(h, new_node);
      return new_node;
   }

   template<class NodePtrCompare>
   static node_ptr insert_equal
      (node_ptr header, node_ptr hint, node_ptr new_node, NodePtrCompare comp)
   {
      bstree_algo::insert_equal(header, hint, new_node, comp);
      rebalance_after_insertion(header, new_node);
      return new_node;
   }

   static node_ptr insert_before
      (node_ptr header, node_ptr pos, node_ptr new_node) noexcept
   {
      bstree_algo::insert_before(header, pos, new_node);
      rebalance_after_insertion(header, new_node);
      return new_node;
   }

   static void push_back(node_ptr header, node_ptr new_node) noexcept
   {
      bstree_algo::push_back(header, new_node);
      rebalance_after_insertion(header, new_node);
   }

   static void push_front(node_ptr header, node_ptr new_node) noexcept
   {
      bstree_algo::push_front(header, new_node);
      rebalance_after_insertion(header, new_node);
   }

   static void insert_unique_commit
      (node_ptr header, node_ptr new_value, const insert_commit_data &commit_data) noexcept
   {
      bstree_algo::insert_unique_commit(header, new_value, commit_data);
      rebalance_after_insertion(header, new_value);
   }

   static bool is_header(const_node_ptr p) noexcept
   {
      return NodeTraits::get_color(p) == NodeTraits::red() &&
            bstree_algo::is_header(p);
   }

private:

   static void rebalance_after_erasure
      ( node_ptr header, node_ptr z, const typename bstree_algo::data_for_rebalance &info) noexcept
   {
      color new_z_color;
      if(info.y != z){
         new_z_color = NodeTraits::get_color(info.y);
         NodeTraits::set_color(info.y, NodeTraits::get_color(z));
      }
      else{
         new_z_color = NodeTraits::get_color(z);
      }
      //Rebalance rbtree if needed
      if(new_z_color != NodeTraits::red()){
         rebalance_after_erasure_restore_invariants(header, info.x, info.x_parent);
      }
   }

   static void rebalance_after_erasure_restore_invariants(node_ptr header, node_ptr x, node_ptr x_parent) noexcept
   {
      while(1){
         if(x_parent == header || (x && NodeTraits::get_color(x) != NodeTraits::black())){
            break;
         }
         //Don't cache x_is_leftchild or similar because x can be null and
         //equal to both x_parent_left and x_parent_right
         const node_ptr x_parent_left(NodeTraits::get_left(x_parent));
         if(x == x_parent_left){ //x is left child
            node_ptr w = NodeTraits::get_right(x_parent);
            assert(w);
            if(NodeTraits::get_color(w) == NodeTraits::red()){
               NodeTraits::set_color(w, NodeTraits::black());
               NodeTraits::set_color(x_parent, NodeTraits::red());
               bstree_algo::rotate_left(x_parent, w, NodeTraits::get_parent(x_parent), header);
               w = NodeTraits::get_right(x_parent);
               assert(w);
            }
            node_ptr const w_left (NodeTraits::get_left(w));
            node_ptr const w_right(NodeTraits::get_right(w));
            if((!w_left  || NodeTraits::get_color(w_left)  == NodeTraits::black()) &&
               (!w_right || NodeTraits::get_color(w_right) == NodeTraits::black())){
               NodeTraits::set_color(w, NodeTraits::red());
               x = x_parent;
               x_parent = NodeTraits::get_parent(x_parent);
            }
            else {
               if(!w_right || NodeTraits::get_color(w_right) == NodeTraits::black()){
                  NodeTraits::set_color(w_left, NodeTraits::black());
                  NodeTraits::set_color(w, NodeTraits::red());
                  bstree_algo::rotate_right(w, w_left, NodeTraits::get_parent(w), header);
                  w = NodeTraits::get_right(x_parent);
                  assert(w);
               }
               NodeTraits::set_color(w, NodeTraits::get_color(x_parent));
               NodeTraits::set_color(x_parent, NodeTraits::black());
               const node_ptr new_wright(NodeTraits::get_right(w));
               if(new_wright)
                  NodeTraits::set_color(new_wright, NodeTraits::black());
               bstree_algo::rotate_left(x_parent, NodeTraits::get_right(x_parent), NodeTraits::get_parent(x_parent), header);
               break;
            }
         }
         else {
            // same as above, with right_ <-> left_.
            node_ptr w = x_parent_left;
            if(NodeTraits::get_color(w) == NodeTraits::red()){
               NodeTraits::set_color(w, NodeTraits::black());
               NodeTraits::set_color(x_parent, NodeTraits::red());
               bstree_algo::rotate_right(x_parent, w, NodeTraits::get_parent(x_parent), header);
               w = NodeTraits::get_left(x_parent);
               assert(w);
            }
            node_ptr const w_left (NodeTraits::get_left(w));
            node_ptr const w_right(NodeTraits::get_right(w));
            if((!w_right || NodeTraits::get_color(w_right) == NodeTraits::black()) &&
               (!w_left  || NodeTraits::get_color(w_left)  == NodeTraits::black())){
               NodeTraits::set_color(w, NodeTraits::red());
               x = x_parent;
               x_parent = NodeTraits::get_parent(x_parent);
            }
            else {
               if(!w_left || NodeTraits::get_color(w_left) == NodeTraits::black()){
                  NodeTraits::set_color(w_right, NodeTraits::black());
                  NodeTraits::set_color(w, NodeTraits::red());
                  bstree_algo::rotate_left(w, w_right, NodeTraits::get_parent(w), header);
                  w = NodeTraits::get_left(x_parent);
                  assert(w);
               }
               NodeTraits::set_color(w, NodeTraits::get_color(x_parent));
               NodeTraits::set_color(x_parent, NodeTraits::black());
               const node_ptr new_wleft(NodeTraits::get_left(w));
               if(new_wleft)
                  NodeTraits::set_color(new_wleft, NodeTraits::black());
               bstree_algo::rotate_right(x_parent, NodeTraits::get_left(x_parent), NodeTraits::get_parent(x_parent), header);
               break;
            }
         }
      }
      if(x)
         NodeTraits::set_color(x, NodeTraits::black());
   }

   static void rebalance_after_insertion(node_ptr header, node_ptr p) noexcept
   {
      NodeTraits::set_color(p, NodeTraits::red());
      while(1){
         node_ptr p_parent(NodeTraits::get_parent(p));
         const node_ptr p_grandparent(NodeTraits::get_parent(p_parent));
         if(p_parent == header || NodeTraits::get_color(p_parent) == NodeTraits::black() || p_grandparent == header){
            break;
         }

         NodeTraits::set_color(p_grandparent, NodeTraits::red());
         node_ptr const p_grandparent_left (NodeTraits::get_left (p_grandparent));
         bool const p_parent_is_left_child = p_parent == p_grandparent_left;
         node_ptr const x(p_parent_is_left_child ? NodeTraits::get_right(p_grandparent) : p_grandparent_left);

         if(x && NodeTraits::get_color(x) == NodeTraits::red()){
            NodeTraits::set_color(x, NodeTraits::black());
            NodeTraits::set_color(p_parent, NodeTraits::black());
            p = p_grandparent;
         }
         else{ //Final step
            const bool p_is_left_child(NodeTraits::get_left(p_parent) == p);
            if(p_parent_is_left_child){ //p_parent is left child
               if(!p_is_left_child){ //p is right child
                  bstree_algo::rotate_left_no_parent_fix(p_parent, p);
                  //No need to link p and p_grandparent:
                  //    [NodeTraits::set_parent(p, p_grandparent) + NodeTraits::set_left(p_grandparent, p)]
                  //as p_grandparent is not the header, another rotation is coming and p_parent
                  //will be the left child of p_grandparent
                  p_parent = p;
               }
               bstree_algo::rotate_right(p_grandparent, p_parent, NodeTraits::get_parent(p_grandparent), header);
            }
            else{  //p_parent is right child
               if(p_is_left_child){ //p is left child
                  bstree_algo::rotate_right_no_parent_fix(p_parent, p);
                  //No need to link p and p_grandparent:
                  //    [NodeTraits::set_parent(p, p_grandparent) + NodeTraits::set_right(p_grandparent, p)]
                  //as p_grandparent is not the header, another rotation is coming and p_parent
                  //will be the right child of p_grandparent
                  p_parent = p;
               }
               bstree_algo::rotate_left(p_grandparent, p_parent, NodeTraits::get_parent(p_grandparent), header);
            }
            NodeTraits::set_color(p_parent, NodeTraits::black());
            break;
         }
      }
      NodeTraits::set_color(NodeTraits::get_parent(header), NodeTraits::black());
   }
};

}

#endif
