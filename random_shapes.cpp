
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>

#include "poly_ops.hpp"
#include "server.hpp"

using namespace poly_ops;

typedef int32_t coord_t;
typedef uint16_t index_t;
std::mt19937 rand_gen;

void random_loop(std::vector<point_t<coord_t>> &loop,index_t size) {
    std::uniform_int_distribution<index_t> dist(0,10000);
    loop.resize(size);
    for(index_t i=0; i<size; ++i) loop[i] = {dist(rand_gen),dist(rand_gen)};
}

struct segment_output {
    const std::pmr::vector<detail::loop_point<index_t,coord_t>> &lpoints;
    detail::segment<index_t> s;
};

std::ostream &operator<<(std::ostream &os,const segment_output &s) {
    return os << s.s.a_x(s.lpoints) << ',' << s.s.a_y(s.lpoints) << " - " <<
        s.s.b_x(s.lpoints) << ',' << s.s.b_y(s.lpoints);
}

void normalize(const std::pmr::vector<detail::loop_point<index_t,coord_t>> &lpoints,detail::segment<index_t> &s) {
    if(s.b_x(lpoints) < s.a_x(lpoints)) std::swap(s.a,s.b);
}

void emit_line(message_canvas &mc,index_t i,const point_t<coord_t> &a,const point_t<coord_t> &b) {
    using namespace json;
    mc.message(obj(
        attr("command") = "line",
        attr("i") = i,
        attr("a") = array_range(a),
        attr("b") = array_range(b)));
}

void test_sweep_order(const char *html_file) {
    std::uniform_int_distribution<index_t> dist(20,80);

    std::pmr::vector<detail::loop_point<index_t,coord_t>> lpoints;
    lpoints.resize(6);

    detail::sweep_cmp<index_t,coord_t> cmp{lpoints};


    for(int test_i=0; test_i < 2000; ++test_i) {
        index_t x = dist(rand_gen);
        std::uniform_int_distribution<index_t> dist_l(0,x);
        std::uniform_int_distribution<index_t> dist_r(x,100);

        for(index_t i = 0; i<3; ++i) {
            lpoints[i*2] = detail::loop_point<index_t,coord_t>({dist_l(rand_gen),dist(rand_gen)},i*2,i*2+1);
            lpoints[i*2+1] = detail::loop_point<index_t,coord_t>({dist_r(rand_gen),dist(rand_gen)},i*2+1,(i*2+2)%6);
        }

        detail::segment<index_t> s[3] = {{0,1},{2,3},{4,5}};
        for(int j=0; j<3; ++j) normalize(lpoints,s[j]);

        if(cmp(s[1],s[0])) {
            if(cmp(s[2],s[1])) {
                std::swap(s[0],s[2]);
            } else {
                std::swap(s[0],s[1]);
                if(cmp(s[2],s[1])) std::swap(s[1],s[2]);
            }
        } else if(cmp(s[2],s[0])){
            std::swap(s[0],s[2]);
            std::swap(s[1],s[2]);
        } else if(cmp(s[2],s[1])) std::swap(s[1],s[2]);

        for(int i=0; i<3; ++i) {
            for(int j=0; j<3; ++j) if(!cmp(s[i],s[j]) == (i < j)) {
                //std::cout << "problem with these lines:\n";
                run_message_server(html_file,[&](message_canvas &mc){
                    for(int k=0; k<3; ++k) {
                        emit_line(mc,k,lpoints[s[k].a].data,lpoints[s[k].b].data);
                        for(int k2=0; k2<3; ++k2) {
                            bool cmp_r = cmp(s[k],s[k2]);
                            mc.console_line_stream()
                                << "cmp " << k << " & " << k2 << " - actual:"
                                << cmp_r << " expected:" << (k < k2);
                        }
                    }
                });
                return;
            }
        }
    }
}

int main(int argc,char **argv) {
    /*if(argc != 2) {
        std::cerr << "Usage: inset_test FILENAME" << std::endl;
        return 1;
    }
    test_sweep_order(argv[1]);*/
    std::vector<point_t<coord_t>> loop;
    for(int i=0; i<1000; ++i) {
        random_loop(loop,100);
        index_t ends[] = {100};
        basic_polygon<index_t,coord_t,point_t<coord_t>> polygon{loop.data(),ends};
        offset_stroke_triangulate<index_t,coord_t>(polygon,50,10);
    }
    return 0;
}
