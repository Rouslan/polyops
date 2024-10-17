#include <vector>
#include <algorithm>

#define BOOST_UT_DISABLE_MODULE
#include "third_party/boost/ut.hpp"

#include "../include/poly_ops/poly_ops.hpp"


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

int main() {
    using namespace boost::ut;

    "Test decomposition"_test = [] {
        std::vector<std::vector<poly_ops::point_t<int>>> loops{{
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

        /* This shape should decompose into exactly 10 shapes and 0 holes */
        auto out = poly_ops::normalize_op<true,int>(loops);
        expect(out.size() == 10);
        expect(std::ranges::all_of(out,[](auto loop) { return loop.inner_loops().size() == 0; }));
    };

    "Test nesting"_test = [] {
        std::vector<std::vector<poly_ops::point_t<int>>> loops{{
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
        expect(out.size() == 1);
        expect(out[0].size() == 9);
        sort_by_size(out[0].inner_loops);

        expect(out[0].inner_loops[0].size() == 3);
        expect(out[0].inner_loops[0].inner_loops.size() == 0);

        expect(out[0].inner_loops[1].size() == 7);
        expect(out[0].inner_loops[1].inner_loops.size() == 1);
        expect(out[0].inner_loops[1].inner_loops[0].size() == 5);
        expect(out[0].inner_loops[1].inner_loops[0].inner_loops.size() == 0);

        expect(out[0].inner_loops[2].size() == 8);
        expect(out[0].inner_loops[2].inner_loops.size() == 1);
        expect(out[0].inner_loops[2].inner_loops[0].size() == 6);
        expect(out[0].inner_loops[2].inner_loops[0].inner_loops.size() == 1);
        expect(out[0].inner_loops[2].inner_loops[0].inner_loops[0].size() == 4);
        expect(out[0].inner_loops[2].inner_loops[0].inner_loops[0].inner_loops.size() == 0);
    };

    return 0;
}
