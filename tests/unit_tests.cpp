
#include <type_traits>
#include <ranges>
#include <iostream>

#include "../include/poly_ops/poly_ops.hpp"

#define BOOST_TEST_MODULE PolyOpsTests
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>


typedef int32_t coord_t;
typedef uint16_t index_t;

#include "stream_output.hpp"

using namespace poly_ops;
namespace bdata = boost::unit_test::data;

auto make_lp(coord_t x,coord_t y,index_t next) {
    return detail::loop_point<index_t,coord_t>(point_t<coord_t>(x,y),next);
}

auto make_lp(coord_t x,coord_t y,index_t next,int line_bal) {
    return detail::loop_point<index_t,coord_t>(point_t<coord_t>(x,y),next,line_bal);
}

namespace poly_ops {
std::ostream &operator<<(std::ostream &os,const point_t<coord_t> &x) {
    return os << pp(x,0);
}

namespace detail {
std::ostream &operator<<(std::ostream &os,const loop_point<index_t,coord_t> &x) {
    return os << '{' << x.data << "}," << ',' << x.next;
}

std::ostream &operator<<(std::ostream &os,const segment<index_t> &x) {
    return os << x.a << " - " << x.b;
}
}
}
template<typename... T> std::ostream &std::operator<<(std::ostream &os,const std::tuple<T...> &x) {
    return os << '{' << arg_delimited<T...>::from_tuple(x) << '}';
}

template<typename R,typename Cmp> boost::test_tools::predicate_result check_total_order(R &&items,Cmp less) {
    std::vector<std::ranges::range_value_t<R>> sorted;
    for(auto &&item : items) {
        for(auto sitr=sorted.begin(); sitr!=sorted.end(); ++sitr) {
            if(less(item,*sitr)) {
                auto inserted = sorted.insert(sitr,std::forward<decltype(item)>(item));
                sitr = std::next(inserted); // sitr is invalidated by "insert"
                auto after_inserted = sitr;
                while(++sitr != sorted.end()) {
                    if(!less(*inserted,*sitr)) {
                        boost::test_tools::predicate_result res(false);
                        res.message() << "items (" << *inserted << "), ("
                            << *after_inserted << ") and (" << *sitr
                            << ") do not have a proper order";
                        return res;
                    }
                }
                goto found;
            }
        }
        sorted.push_back(std::forward<decltype(item)>(item));
      found: ;
    }
    return true;
}

point_t<coord_t> mirror(const point_t<coord_t> &x) {
    return {-x[0],x[1]};
}

template<typename R> struct range_dataset {
    using iterator = std::ranges::iterator_t<const R>;

    static const int arity = std::tuple_size_v<std::ranges::range_value_t<R>>;

    R r;

    bdata::size_t size() const { return std::ranges::size(r); }
    iterator begin() const { return std::begin(r); }
};

template<typename R> struct bdata::monomorphic::is_dataset<range_dataset<R>> : boost::mpl::true_ {};

auto make_test_ray_intersections_dataset() {
    using result_t = std::tuple<point_t<coord_t>,point_t<coord_t>,point_t<coord_t>,coord_t,point_t<coord_t>,bool,bool>;

    // these lines are in order by slope, decreasing
    static const point_t<coord_t> lines[][2] = {
        {{100,100},{10,10}},
        {{100,100},{40,30}},
        {{100,100},{60,40}},
        {{100,100},{20,-10}},
        {{100,100},{10,-30}}
    };

    const size_t line_count = sizeof(lines)/sizeof(lines[0]);
    const size_t x_count = line_count * 4 + 2;
    const size_t total = line_count * x_count;

    return range_dataset(std::views::iota(size_t(0),total) | std::views::transform([=](size_t i) -> result_t {
        const size_t y = i / x_count;
        auto &l1 = lines[y];
        size_t x = i % x_count;

        switch(x) {
        case 0:
            return result_t(l1[0],l1[0]+l1[1],l1[0],1,point_t<coord_t>(0,-1),false,false);
        case 1:
            return result_t(l1[0],l1[0]+l1[1],l1[0],1,point_t<coord_t>(0,1),false,true);
        default:
            {
                x -= 2;
                size_t z = x/4;
                auto &l2 = lines[z];
                switch(x % 4) {
                case 0:
                    return result_t(l1[0],l1[0]+l1[1],l2[0],1,l2[1],false,y>=z);
                case 1:
                    return result_t(l1[0]+mirror(l1[1]),l1[0],l2[0],1,l2[1],false,false);
                case 2:
                    return result_t(l1[0]+mirror(l1[1]),l1[0],l2[0],-1,mirror(l2[1]),false,y>=z);
                default:
                    return result_t(l1[0],l1[0]+l1[1],l2[0],-1,mirror(l2[1]),false,false);
                }
            }
        }
    }));
}

BOOST_DATA_TEST_CASE(
    test_ray_intersections,
    make_test_ray_intersections_dataset(),
    sa,sb,p,hsign,dp,tie_breaker,expected)
{
    BOOST_TEST(detail::line_segment_up_ray_intersection(sa,sb,p,hsign,dp,tie_breaker) == expected);
    BOOST_TEST(detail::dual_line_segment_up_ray_intersection(sa,sb,p,hsign,dp,tie_breaker,hsign,dp,tie_breaker) == std::tuple(expected,expected));
}

BOOST_AUTO_TEST_CASE(test_sweep_cmp) {
    /* these lines are in order from high-Y to low (the order sweep_cmp is
    supposed to enforce) */
    point_t<coord_t> lines[][2] = {
        {{60,100},{60,90}},
        {{50,100},{60,80}},
        {{50,70},{100,110}},
        {{60,60},{60,70}},
        {{0,90},{70,50}},
        {{40,40},{70,40}},
        {{30,10},{130,80}}
    };
    index_t order[] = {6,0,5,1,4,2,3};
    size_t o_size = std::extent_v<decltype(order)>;

    std::pmr::vector<detail::loop_point<index_t,coord_t>> lpoints;
    detail::sweep_cmp<index_t,coord_t> cmp{lpoints};
    detail::sweep_t<index_t,coord_t> sweep(cmp);
    std::pmr::vector<detail::segment<index_t>> ordered_seg;
    ordered_seg.resize(o_size);

    for(index_t i : order) {
        index_t li = lpoints.size();
        lpoints.emplace_back(lines[i][0],li+1,static_cast<int>(i));
        lpoints.emplace_back(lines[i][1],li,static_cast<int>(i));
        detail::segment<index_t> s(li,li+1);
        sweep.insert(s);
        ordered_seg[i] = s;
    }

    for(size_t i=0; i<o_size; ++i) {
        for(size_t j=0; j<o_size; ++j) {
            detail::segment<index_t> s1 = ordered_seg[i];
            detail::segment<index_t> s2 = ordered_seg[j];
            BOOST_CHECK((i < j) == cmp(s1,s2));
            /*if((i < j) != cmp(s1,s2)) {
                std::cout << "comparison fail: [" << lpoints[s1.a].data
                    << " - " << lpoints[s1.b].data << "] > ["
                    << lpoints[s2.a].data << " - " << lpoints[s2.b].data
                    << "]\n";
            }*/
        }
    }

    auto itr = sweep.begin();
    for(size_t i=0; i<o_size; ++i, ++itr) {
        if(lpoints[itr->a].line_bal != static_cast<int>(i)) {
            std::cout << "sweep sort failure\norder is: ";
            for(auto &p : sweep) std::cout << lpoints[p.a].line_bal << ' ';
            std::cout << '\n';
            break;
        }
    }
}

BOOST_AUTO_TEST_CASE(test_sweep_cmp_2) {
    std::pmr::vector<detail::loop_point<index_t,coord_t>> lpoints{
        make_lp(127,969,1),
        make_lp(914,221,0),
        make_lp(97,547,0),
        make_lp(359,430,2),
        make_lp(632,308,3),
        make_lp(278,188,6),
        make_lp(359,430,7),
        make_lp(547,993,0)};
    detail::sweep_t<index_t,coord_t> sweep({
            detail::segment<index_t>(0,1),
            detail::segment<index_t>(2,3),
            detail::segment<index_t>(3,4),
            detail::segment<index_t>(5,6)},
        detail::sweep_cmp<index_t,coord_t>{lpoints});
    BOOST_CHECK(!sweep.count(detail::segment<index_t>(6,7)));
}

BOOST_AUTO_TEST_CASE(test_sweep_cmp_3) {
    std::pmr::vector<detail::loop_point<index_t,coord_t>> lpoints{
        make_lp(20,20,1),   //  0
        make_lp(60,40,0),   //  1
        make_lp(30,70,3),   //  2
        make_lp(40,30,4),   //  3
        make_lp(50,50,0),   //  4
        make_lp(40,30,6),   //  5
        make_lp(40,40,0),   //  6
        make_lp(20,40,8),   //  7
        make_lp(60,20,0),   //  8
        make_lp(30,50,10),  //  9
        make_lp(40,40,0),   // 10
        make_lp(0,196,12),  // 11
        make_lp(727,782,0), // 12
        make_lp(310,59,14), // 13
        make_lp(695,129,0), // 14
        make_lp(520,97,16), // 15
        make_lp(609,946,0)  // 16
    };
    detail::sweep_cmp<index_t,coord_t> cmp{lpoints};
    detail::segment<index_t> s1(0,1), s2(7,8), s3(2,3), s4(3,4), s5(5,6),
        s6(9,10), s7(11,12), s8(15,16), s9(13,14);
    BOOST_CHECK(cmp(s3,s1));
    BOOST_CHECK(!cmp(s1,s3));
    BOOST_CHECK(cmp(s4,s1));
    BOOST_CHECK(!cmp(s1,s4));
    BOOST_CHECK(cmp(s5,s1));
    BOOST_CHECK(!cmp(s1,s5));
    BOOST_CHECK(cmp(s6,s1));
    BOOST_CHECK(!cmp(s1,s6));
    BOOST_CHECK(cmp(s3,s2));
    BOOST_CHECK(!cmp(s2,s3));
    BOOST_CHECK(cmp(s4,s2));
    BOOST_CHECK(!cmp(s2,s4));
    BOOST_CHECK(cmp(s5,s2));
    BOOST_CHECK(!cmp(s2,s5));
    BOOST_CHECK(cmp(s6,s2));
    BOOST_CHECK(!cmp(s2,s6));

    BOOST_CHECK(cmp(s8,s9));
    BOOST_CHECK(!cmp(s9,s8));
    BOOST_CHECK(cmp(s7,s9));
    BOOST_CHECK(!cmp(s9,s7));
    BOOST_CHECK(cmp(s7,s8));
    BOOST_CHECK(!cmp(s8,s7));
}

BOOST_AUTO_TEST_CASE(test_sweep_cmp4) {
    std::pmr::vector<detail::loop_point<index_t,coord_t>> lpoints{
        make_lp(472,447,1),   //  0
        make_lp(0,0,0),       //  1
        make_lp(373,575,3),   //  2
        make_lp(513,69,4),    //  3
        make_lp(59,978,0),    //  4
        make_lp(373,574,6),   //  5
        make_lp(373,574,0),   //  6
        make_lp(462,171,8),   //  7
        make_lp(0,0,0),       //  8
        make_lp(373,573,10),  //  9
        make_lp(373,573,0),   // 10
        make_lp(373,574,12)   // 11
    };

    auto segments = {
        detail::segment<index_t>(2,11),
        detail::segment<index_t>(5,2),
        detail::segment<index_t>(5,7),
        detail::segment<index_t>(6,10),
        detail::segment<index_t>(11,9),
        detail::segment<index_t>(10,0),
        detail::segment<index_t>(9,3),
        detail::segment<index_t>(4,8)};

    /* The exact order between overlapping lines is not important. It just needs
    to be consistent. */

    BOOST_TEST(check_total_order(segments,detail::sweep_cmp<index_t,coord_t>{lpoints}));
}

BOOST_AUTO_TEST_CASE(test_sweep_cmp_overlap) {
    std::pmr::vector<detail::loop_point<index_t,coord_t>> lpoints{
        make_lp(419,489,1),
        make_lp(420,490,0),
        make_lp(419,489,3),
        make_lp(420,490,2)
    };
    detail::sweep_cmp<index_t,coord_t> cmp{lpoints};

    detail::segment<index_t> s1{0,1}, s2{2,3};
    BOOST_CHECK_NE(cmp(s1,s2),cmp(s2,s1));
}

BOOST_AUTO_TEST_CASE(test_sweep_cmp_vert_overlap) {
    std::pmr::vector<detail::loop_point<index_t,coord_t>> lpoints{
        make_lp(630,829,4),
        make_lp(573,177,2),
        make_lp(621,839,3),
        make_lp(630,824,6),
        make_lp(630,824,1),
        make_lp(630,823,0),
        make_lp(630,823,7),
        make_lp(0,0,6)
    };

    auto segments = {
        detail::segment<index_t>(2,6),
        detail::segment<index_t>(4,5),
        detail::segment<index_t>(3,6),
        detail::segment<index_t>(5,8),
        detail::segment<index_t>(7,2)};

    /* The exact order between overlapping lines is not important. It just needs
    to be consistent. */

    BOOST_TEST(check_total_order(segments,detail::sweep_cmp<index_t,coord_t>{lpoints}));
}

BOOST_AUTO_TEST_CASE(test_sweep_tee) {
    /*  4---e---6
        |
        a
        |
       1/2--c---3
        |
        b
        |
        0---d---5
    */
    std::pmr::vector<detail::loop_point<index_t,coord_t>> lpoints{
        make_lp(0,1,0),   //  0
        make_lp(0,0,0),   //  1
        make_lp(0,0,0),   //  2
        make_lp(1,0,0),   //  3
        make_lp(0,-1,0),  //  4
        make_lp(1,1,0),   //  5
        make_lp(1,-1,0)   //  6
    };

    detail::segment<index_t> a(2,4), b(1,0), c(1,3), d(0,5), e(4,6);
    detail::sweep_cmp<index_t,coord_t> cmp{lpoints};

    BOOST_TEST(cmp(b,a));
    BOOST_TEST(cmp(c,a));
    BOOST_TEST(cmp(b,c));
    BOOST_TEST(cmp(d,c));
    BOOST_TEST(cmp(c,e));
}

BOOST_AUTO_TEST_CASE(test_point_on_edge_intersect) {
    // This was a problem at one point. One check succeeded; the other failed.

    std::pmr::vector<detail::loop_point<index_t,coord_t>> lpoints{
        make_lp(465,583,1),
        make_lp(915,248,0),
        make_lp(837,79,3),
        make_lp(933,287,2)
    };

    detail::segment<index_t> s1{0,1}, s2{2,3};
    point_t<coord_t> intr;
    detail::at_edge_t at_edge[2];
    BOOST_CHECK(intersects(s1,s2,lpoints.data(),intr,at_edge));
    BOOST_CHECK(intersects(s2,s1,lpoints.data(),intr,at_edge));
}

BOOST_AUTO_TEST_CASE(test_overlap_intersect) {
    std::pmr::vector<detail::loop_point<index_t,coord_t>> lpoints{
        make_lp(630,823,1),
        make_lp(630,829,0),
        make_lp(630,824,3),
        make_lp(630,823,2)
    };

    detail::segment<index_t> s1{0,1}, s2{2,3};
    point_t<coord_t> intr;
    detail::at_edge_t at_edge[2];
    BOOST_CHECK(intersects(s1,s2,lpoints.data(),intr,at_edge));
}

BOOST_AUTO_TEST_CASE(test_follow_balance) {
    std::pmr::vector<detail::loop_point<index_t,coord_t>> lpoints{
        make_lp(0,1,1),
        make_lp(2,3,2,0),
        make_lp(4,5,3),
        make_lp(2,3,4,-1),
        make_lp(6,7,5),
        make_lp(8,9,0),
    };

    detail::broken_starts_t<index_t,coord_t> broken_starts;
    std::pmr::vector<index_t> broken_ends;

    follow_balance<index_t,coord_t>(lpoints.data(),1,broken_ends,broken_starts,nullptr);
    follow_balance<index_t,coord_t>(lpoints.data(),3,broken_ends,broken_starts,nullptr);

    BOOST_CHECK(broken_starts.size() == 1);
    auto itr = broken_starts.find(point_t<coord_t>(2,3));
    BOOST_CHECK(itr != broken_starts.end() && itr->second.size() == 1 && itr->second[0] == 1);
    BOOST_CHECK(broken_ends.size() == 1 && broken_ends[0] == 2);
}
