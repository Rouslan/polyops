#include <vector>
#include <algorithm>
#include <ostream>
#include <numbers>

#define BOOST_UT_DISABLE_MODULE
#include "third_party/boost/ut.hpp"

#include "../include/poly_ops/poly_ops.hpp"


namespace poly_ops {
    template<typename T> std::ostream &operator<<(std::ostream &os,const point_t<T> &p) {
        return os << '{' << p.x() << ',' << p.y() << '}';
    }
}

template<typename R> struct any_range {
    R val;
};
template<typename R> any_range(R&&) -> any_range<R>;
template<typename T,std::size_t N> any_range(T (&)[N]) -> any_range<std::span<T>>;
template<typename R1,typename R2> bool operator==(any_range<R1> a,any_range<R2> b) {
    return std::ranges::equal(a.val,b.val);
}
template<typename R> std::ostream &operator<<(std::ostream &os,const any_range<R> &p) {
    os << '{';
    bool first = true;
    for(auto &&item : p.val) {
        if(!first) os << ',';
        first = false;
        os << item;
    }
    return os << '}';
}

template<typename R1,typename R2> auto expect_range_eq(R1 &&a,R2 &&b,std::source_location loc=std::source_location::current()) {
    return boost::ut::expect(boost::ut::eq(any_range{a},any_range{b}),loc);
}

template<typename T,typename U> bool point_approx_equal(poly_ops::point_t<T> a,poly_ops::point_t<U> b,double delta) {
    return std::abs(a.x() - b.x()) <= delta && std::abs(a.y() - b.y()) < delta;
}

template<typename T> struct point_with_delta {
    poly_ops::point_t<T> p;
    double delta;
};
template<typename T,typename U> bool operator==(const point_with_delta<T> &a,poly_ops::point_t<U> b) {
    return point_approx_equal(a.p,b,a.delta);
}
template<typename R> std::ostream &operator<<(std::ostream &os,const point_with_delta<R> &p) {
    return os << p.p;
}

template<typename T,typename U> auto expect_range_eq(poly_ops::point_t<T> a,poly_ops::point_t<U> b,double delta,std::source_location loc=std::source_location::current()) {
    return boost::ut::expect(boost::ut::eq(point_with_delta{a,delta},b),loc);
}

struct p {
    int x;
    int y;

    friend auto operator==(poly_ops::point_t<int> a,p b) {
        return boost::ut::eq(a,poly_ops::point_t{b.x,b.y});
    }
    friend auto operator==(p a,poly_ops::point_t<int> b) {
        return boost::ut::eq(poly_ops::point_t{a.x,a.y},b);
    }
};

struct point_array_tree {
    std::vector<poly_ops::point_t<int>> items;
    std::vector<point_array_tree> inner_loops;

    std::size_t size() const { return items.size(); }
};

template<typename T,typename R> auto make_vector(R &&src) {
    return std::vector<T>{src.begin(),src.end()};
}

std::vector<point_array_tree> make_array_trees(auto &&x);

point_array_tree make_array_tree(auto &&x) {
    return point_array_tree{
        make_vector<poly_ops::point_t<int>>(x),
        make_array_trees(x.inner_loops())};
}

std::vector<point_array_tree> make_array_trees(auto &&x) {
    return make_vector<point_array_tree>(x | std::views::transform([](auto &&item) { return make_array_tree(item); }));
}

void sort_by_size(auto &x) {
    std::ranges::sort(x,{},[](auto &item) { return item.items.size(); });
}

void for_each_combination(
    std::span<std::span<const poly_ops::point_t<int>>> input,
    auto &&fun,
    std::vector<std::span<const poly_ops::point_t<int>>> &buffer)
{
    if(input.empty()) fun(buffer);
    else {
        std::vector<std::span<const poly_ops::point_t<int>>> current;
        current.reserve(input.size()-1);
        for(std::size_t i=0; i<input.size(); ++i) {
            current.clear();
            current.insert(current.end(),input.begin(),input.begin()+long(i));
            current.insert(current.end(),input.begin()+(long(i)+1),input.end());
            buffer.push_back(input[i]);
            for_each_combination(current,fun,buffer);
            buffer.pop_back();
        }
    }
}
void for_each_combination(std::span<std::span<const poly_ops::point_t<int>>> input,auto &&fun)
{
    std::vector<std::span<const poly_ops::point_t<int>>> buffer;
    buffer.reserve(input.size());
    for_each_combination(input,fun,buffer);
}

int main() {
    using namespace boost::ut;
    using loop_t = std::vector<poly_ops::point_t<int>>;

    "Test decomposition"_test = [] {
        std::vector<loop_t> loops{{
            {280,220},{249,215},{221,200},{199,178},{184,150},{180,119},
            {184,89},{199,61},{221,39},{249,24},{280,19},{310,24},{338,39},
            {360,61},{375,89},{380,119},{375,150},{360,178},{338,200},{310,215}
        },{
            {120,380},{89,375},{61,360},{39,338},{24,310},{19,280},{24,249},
            {39,221},{61,199},{89,184},{120,179},{150,184},{178,199},{200,221},
            {215,249},{220,280},{215,310},{200,338},{178,360},{150,375}
        },{
            {240,300},{209,295},{181,280},{159,258},{144,230},{140,200},
            {144,169},{159,141},{181,119},{209,104},{240,99},{270,104},
            {298,119},{320,141},{335,169},{340,200},{335,230},{320,258},
            {298,280},{270,295}
        },{
            {200,340},{169,335},{141,320},{119,298},{104,270},{99,240},
            {104,209},{119,181},{141,159},{169,144},{200,139},{230,144},
            {258,159},{280,181},{295,209},{300,240},{295,270},{280,298},
            {258,320},{230,335}
        },{
            {200,260},{169,255},{141,240},{119,218},{104,190},{99,160},
            {104,129},{119,101},{141,79},{169,64},{200,59},{230,64},{258,79},
            {280,101},{295,129},{300,160},{295,190},{280,218},{258,240},
            {230,255}
        }};

        std::vector<std::span<const poly_ops::point_t<int>>> current{loops.begin(),loops.end()};
        for_each_combination(current,[](std::span<std::span<const poly_ops::point_t<int>>> data) {
            /* This shape should decompose into exactly 10 shapes and 0 holes */
            auto out = poly_ops::normalize_op<true,int>(data);
            expect(out.size() == 10_u);
            expect(std::ranges::all_of(out,[](auto loop) { return loop.inner_loops().size() == 0; }));
        });
    };

    "Test nesting"_test = [] {
        std::vector<loop_t> loops{{
            { 20, 20},{157, 43},{256, 17},{338, 30},{356, 89},{363,189},
            {130,204},{ 14,185},{ 36, 95}
        },{
            { 63,155},{ 58,103},{ 95, 66},{150, 74},{176,106},{178,146},
            {150,176},{111,179}
        },{
            {265,165},{207,165},{196,115},{212, 62},{298, 46},{336,101},{332,145}
        },{
            {118,157},{ 81,136},{ 80,103},{120, 89},{156,112},{149,144}
        },{
            {238,149},{218, 74},{287, 58},{320,123},{287,142}
        },{
            {107,130},{101,104},{138,118},{126,143}
        },{
            {188, 87},{157, 57},{197, 53}
        }};

        auto out = make_array_trees(poly_ops::normalize_op<true,int>(loops));
        expect(fatal(out.size() == 1_u));
        expect(out[0].size() == 9_u);
        expect(fatal(out[0].inner_loops.size() == 3_u));
        sort_by_size(out[0].inner_loops);

        expect(out[0].inner_loops[0].size() == 3_u);
        expect(out[0].inner_loops[0].inner_loops.size() == 0_u);

        expect(out[0].inner_loops[1].size() == 7_u);
        expect(fatal(out[0].inner_loops[1].inner_loops.size() == 1_u));
        expect(out[0].inner_loops[1].inner_loops[0].size() == 5_u);
        expect(out[0].inner_loops[1].inner_loops[0].inner_loops.size() == 0_u);

        expect(out[0].inner_loops[2].size() == 8_u);
        expect(fatal(out[0].inner_loops[2].inner_loops.size() == 1_u));
        expect(out[0].inner_loops[2].inner_loops[0].size() == 6_u);
        expect(fatal(out[0].inner_loops[2].inner_loops[0].inner_loops.size() == 1_u));
        expect(out[0].inner_loops[2].inner_loops[0].inner_loops[0].size() == 4_u);
        expect(out[0].inner_loops[2].inner_loops[0].inner_loops[0].inner_loops.size() == 0_u);
    };

    "Test basic offset"_test = [] {
        loop_t box = {{0,0},{1000,0},{1000,1000},{0,1000}};
        auto result = poly_ops::offset<false,int>(box,50,1000000,poly_ops::origin_point_tracker{});
        expect(fatal(result.size() == 1_u));
        expect(fatal(result[0u].size() == 8_u));

        std::vector<poly_ops::point_and_origin<int>> box2{result[0u].begin(),result[0u].end()};

        // rotate until {-50,0} is the first point
        auto itr = std::ranges::find(box2,poly_ops::point_t{-50,0},[](auto &x) { return x.p; });
        expect(fatal(itr != box2.end()));
        std::ranges::rotate(box2,itr);

        std::size_t o1[] = {0};
        std::size_t o2[] = {1};
        std::size_t o3[] = {2};
        std::size_t o4[] = {3};
        expect_range_eq(box2[0].original_points,o1);
        expect(box2[1].p == p{0,-50});
        expect_range_eq(box2[1].original_points,o1);

        expect(box2[2].p == p{1000,-50});
        expect_range_eq(box2[2].original_points,o2);
        expect(box2[3].p == p{1050,0});
        expect_range_eq(box2[3].original_points,o2);

        expect(box2[4].p == p{1050,1000});
        expect_range_eq(box2[4].original_points,o3);
        expect(box2[5].p == p{1000,1050});
        expect_range_eq(box2[5].original_points,o3);

        expect(box2[6].p == p{0,1050});
        expect_range_eq(box2[6].original_points,o4);
        expect(box2[7].p == p{-50,1000});
        expect_range_eq(box2[7].original_points,o4);
    };

    "Test basic inset"_test = [] {
        loop_t box = {{0,0},{1000,0},{1000,1000},{0,1000}};
        auto result = poly_ops::offset<false,int>(box,-50,1000000,poly_ops::origin_point_tracker{});
        expect(fatal(result.size() == 1_u));
        expect(fatal(result[0u].size() == 4_u));

        std::vector<poly_ops::point_and_origin<int>> box2{result[0u].begin(),result[0u].end()};

        // rotate until {50,50} is the first point
        auto itr = std::ranges::find(box2,poly_ops::point_t{50,50},[](auto &x) { return x.p; });
        expect(fatal(itr != box2.end()));
        std::ranges::rotate(box2,itr);

        std::size_t o1[] = {0};
        std::size_t o2[] = {1};
        std::size_t o3[] = {2};
        std::size_t o4[] = {3};
        expect_range_eq(box2[0].original_points,o1);

        expect(box2[1].p == p{950,50});
        expect_range_eq(box2[1].original_points,o2);

        expect(box2[2].p == p{950,950});
        expect_range_eq(box2[2].original_points,o3);

        expect(box2[3].p == p{50,950});
        expect_range_eq(box2[3].original_points,o4);
    };

    "Test compound offset"_test = [] {
        std::vector<loop_t> boxes = {
            {{0,0},{1000,0},{1000,1000},{0,1000}},
            {{2000,0},{3000,0},{3000,1000},{2000,1000}},
            {{4000,0},{5000,0},{5000,1000},{4000,1000}}};
        auto result = poly_ops::offset<false,int>(boxes,50,1000000,poly_ops::origin_point_tracker{});
        expect(fatal(result.size() == 3_u));

        for(std::size_t i=0; i<3; ++i) {
            int x_offset = int(i)*2000;
            std::size_t index_offset = i*4;
            expect(fatal(result[i].size() == 8_u));

            std::vector<poly_ops::point_and_origin<int>> box2{result[i].begin(),result[i].end()};

            // rotate until {-50,0} is the first point
            auto itr = std::ranges::find(box2,poly_ops::point_t{-50+x_offset,0},[](auto &x) { return x.p; });
            expect(fatal(itr != box2.end()));
            std::ranges::rotate(box2,itr);

            std::size_t o1[] = {index_offset+0};
            std::size_t o2[] = {index_offset+1};
            std::size_t o3[] = {index_offset+2};
            std::size_t o4[] = {index_offset+3};
            expect_range_eq(box2[0].original_points,o1);
            expect(box2[1].p == p{x_offset,-50});
            expect_range_eq(box2[1].original_points,o1);

            expect(box2[2].p == p{1000+x_offset,-50});
            expect_range_eq(box2[2].original_points,o2);
            expect(box2[3].p == p{1050+x_offset,0});
            expect_range_eq(box2[3].original_points,o2);

            expect(box2[4].p == p{1050+x_offset,1000});
            expect_range_eq(box2[4].original_points,o3);
            expect(box2[5].p == p{1000+x_offset,1050});
            expect_range_eq(box2[5].original_points,o3);

            expect(box2[6].p == p{x_offset,1050});
            expect_range_eq(box2[6].original_points,o4);
            expect(box2[7].p == p{-50+x_offset,1000});
            expect_range_eq(box2[7].original_points,o4);
        }
    };

    poly_ops::origin_tracked_clipper<int> tclipper;

    "Test offset curves"_test = [](auto values) {
        struct point_meta {
            double angle;
            poly_ops::point_t<double> curve_start;
            poly_ops::point_t<double> curve_end;
            std::size_t segments;
        };

        auto &&[loop,offset_amount,arc_step_size,tclipper] = values;

        std::size_t total_points = loop.size();

        std::vector<point_meta> meta{loop.size()};
        std::size_t prev_i = loop.size() - 1;
        for(std::size_t i=0; i<loop.size(); ++i) {
            auto &m = meta[i];
            m.angle = std::numbers::pi - poly_ops::vangle<int>(loop[prev_i] - loop[i],loop[(i+1)%loop.size()] - loop[i]);
            m.curve_start = poly_ops::perp_vector<int>(loop[prev_i],loop[i],offset_amount) + loop[i];
            m.curve_end = poly_ops::perp_vector<int>(loop[i],loop[(i+1)%loop.size()],offset_amount) + loop[i];
            m.segments = std::max(std::size_t(m.angle * offset_amount / arc_step_size),std::size_t(1));

            total_points += m.segments;
            prev_i = i;
        }

        std::vector<poly_ops::point_and_origin<int>> box2;

        /* box2 will reference data in this object (if tclipper is nullptr) */
        poly_ops::origin_point_tracker tracker{};

        if(tclipper) {
            poly_ops::add_offset_loops_subject(*tclipper,loop,offset_amount,arc_step_size);
            auto result = tclipper->execute(poly_ops::bool_op::union_);
            expect(fatal(result.size() == 1_u));
            box2.insert(box2.end(),result[0u].begin(),result[0u].end());
        } else {
            auto result = poly_ops::offset<false,int>(loop,offset_amount,arc_step_size,tracker);
            expect(fatal(result.size() == 1_u));
            box2.insert(box2.end(),result[0u].begin(),result[0u].end());
        }
        expect(fatal(eq(box2.size(),total_points)));

        std::size_t i = 0;
        for(;; ++i) {
            expect(fatal(lt(i,box2.size())));
            if(point_approx_equal(box2[i].p,meta[0].curve_start,1.0)) break;
        }

        for(std::size_t k=0; k<loop.size(); ++k) {
            auto &m = meta[k];
            expect_range_eq(box2[i].p,m.curve_start,1.0);
            for(std::size_t j=0; j<m.segments+1; ++j) {
                prev_i = i;
                auto target_op = {k};
                expect_range_eq(box2[i].original_points,target_op);
                i = (i+1) % total_points;
            }
            expect_range_eq(box2[prev_i].p,m.curve_end,1.0);
        }
    } | std::vector<std::tuple<loop_t,double,int,poly_ops::origin_tracked_clipper<int>*>>{
        {loop_t{{0,0},{1000,0},{1000,1000},{0,1000}},50.0,10,nullptr},
        {loop_t{{3225,-3225},{5450,-13525},{16000,-15450},{8025,-1575}},2000.0,100,nullptr},
        {loop_t{{20,20},{140,35},{37,142},{20,100}},20.0,15,nullptr},
        {loop_t{{0,0},{1000,0},{1000,1000},{0,1000}},50.0,10,&tclipper},
        {loop_t{{3225,-3225},{5450,-13525},{16000,-15450},{8025,-1575}},2000.0,100,&tclipper},
        {loop_t{{20,20},{140,35},{37,142},{20,100}},20.0,15,&tclipper}
    };

    return 0;
}
