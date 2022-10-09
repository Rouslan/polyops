
#include <type_traits>
#include <iostream>

#include "poly_ops.hpp"
#include "stream_output.hpp"

#define BOOST_TEST_MODULE PolyOpsTests
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

using namespace poly_ops;

typedef int32_t coord_t;
typedef uint16_t index_t;


auto make_lp(coord_t x,coord_t y,index_t next) {
    return detail::loop_point<index_t,coord_t>(point_t<coord_t>(x,y),0,next);
}

auto make_lp(coord_t x,coord_t y,index_t next,int line_bal) {
    return detail::loop_point<index_t,coord_t>(point_t<coord_t>(x,y),0,next,line_bal);
}

std::ostream &operator<<(std::ostream &os,const point_t<coord_t> &x) {
    return os << x[0] << ',' << x[1];
}

std::ostream &operator<<(std::ostream &os,const detail::loop_point<index_t,coord_t> &x) {
    return os << '{' << x.data << "}," << x.original_set << ',' << x.next;
}

std::ostream &operator<<(std::ostream &os,const detail::segment<index_t> &x) {
    return os << x.a << " - " << x.b;
}

template<typename Actual,typename Expected,typename... Args>
void test_func(const char *fun_name,Actual &&actual,Expected &&expected,Args&&... args) {
    BOOST_CHECK(actual == expected);
    /*if(actual != expected) {
        std::cout << '\n' << fun_name << '(' << arg_delimited{args...} <<
            ") failed\nexpected: " << pp{expected} << "\nactual: " << pp{actual}
            << '\n';
    }*/
}

#define TEST_FUNC(F,expected,...) \
    test_func(#F,F(__VA_ARGS__),expected __VA_OPT__(,) __VA_ARGS__)

void test_dual_line_segment_up_ray_intersection(
    point_t<coord_t> sa,
    point_t<coord_t> sb,
    point_t<coord_t> p,
    coord_t hsign1,
    point_t<coord_t> dp1,
    coord_t hsign2,
    point_t<coord_t> dp2,
    bool expected1,
    bool expected2)
{
    TEST_FUNC(
        detail::dual_line_segment_up_ray_intersection,
        std::tuple(expected1,expected2),
        sa,sb,p,hsign1,dp1,hsign2,dp2);
}

void test_line_segment_up_ray_intersection(
    point_t<coord_t> sa,
    point_t<coord_t> sb,
    point_t<coord_t> p,
    coord_t hsign,
    point_t<coord_t> dp,
    bool expected)
{
    TEST_FUNC(
        detail::line_segment_up_ray_intersection,
        expected,
        sa,sb,p,hsign,dp);
    test_dual_line_segment_up_ray_intersection(sa,sb,p,hsign,dp,hsign,dp,expected,expected);
}

point_t<coord_t> mirror(const point_t<coord_t> &x) {
    return {-x[0],x[1]};
}

BOOST_AUTO_TEST_CASE(test_ray_intersections) {
    test_line_segment_up_ray_intersection({-5,5},{40,50},{8,82},1,{10,10},true);

    // these lines are in order by slope, decreasing
    point_t<coord_t> right_lines[][2] = {
        {{100,100},{10,10}},
        {{100,100},{40,30}},
        {{100,100},{60,40}},
        {{100,100},{20,-10}},
        {{100,100},{10,-30}}
    };
    const int a_size = sizeof(right_lines)/sizeof(right_lines[0]);
    for(int i=0; i<a_size; ++i) {
        auto &l1 = right_lines[i];
        for(int j=0; j<a_size; ++j) {
            auto &l2 = right_lines[j];
            test_line_segment_up_ray_intersection(l1[0],l1[0]+l1[1],l2[0],1,l2[1],i>=j);
            test_line_segment_up_ray_intersection(l1[0]+mirror(l1[1]),l1[0],l2[0],1,l2[1],false);
            test_line_segment_up_ray_intersection(l1[0]+mirror(l1[1]),l1[0],l2[0],-1,mirror(l2[1]),i>=j);
            test_line_segment_up_ray_intersection(l1[0],l1[0]+l1[1],l2[0],-1,mirror(l2[1]),false);
        }
        test_line_segment_up_ray_intersection(l1[0],l1[0]+l1[1],l1[0],1,{0,-1},false);
        test_line_segment_up_ray_intersection(l1[0],l1[0]+l1[1],l1[0],1,{0,1},true);
    }
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
        lpoints.emplace_back(lines[i][0],i,li+1);
        lpoints.emplace_back(lines[i][1],i,li);
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
        if(lpoints[itr->a].original_set != i) {
            std::cout << "sweep sort failure\norder is: ";
            for(auto &p : sweep) std::cout << lpoints[p.a].original_set << ' ';
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

BOOST_AUTO_TEST_CASE(test_follow_balance) {
    std::pmr::vector<detail::loop_point<index_t,coord_t>> lpoints{
        make_lp(0,1,1),
        make_lp(2,3,2,0),
        make_lp(4,5,3),
        make_lp(2,3,4,-1),
        make_lp(6,7,5),
        make_lp(8,9,0),
    };

    detail::original_sets_t<index_t> original_sets(std::pmr::get_default_resource());
    original_sets.emplace_back();

    detail::broken_starts_t<index_t,coord_t> broken_starts;
    std::pmr::vector<index_t> broken_ends;

    follow_balance<index_t,coord_t>(lpoints.data(),1,broken_ends,broken_starts,original_sets);
    follow_balance<index_t,coord_t>(lpoints.data(),3,broken_ends,broken_starts,original_sets);

    BOOST_CHECK(broken_starts.size() == 1);
    auto itr = broken_starts.find(point_t<coord_t>(2,3));
    BOOST_CHECK(itr != broken_starts.end() && itr->second.size() == 1 && itr->second[0] == 1);
    BOOST_CHECK(broken_ends.size() == 1 && broken_ends[0] == 2);
}
