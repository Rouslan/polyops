
#include <iostream>
#include <fstream>
#include <random>
#include <map>
#include <ranges>

#define DEBUG_STEP_BY_STEP 1

#include "graphical_test_common.hpp"


constexpr size_t LOOP_SIZE = 5;

void random_loop(std::mt19937 &rand_gen,std::vector<point_t<coord_t>> &loop,index_t size) {
    std::uniform_int_distribution<index_t> dist(0,1000);
    loop.resize(size);
    for(index_t i=0; i<size; ++i) loop[i] = {dist(rand_gen),dist(rand_gen)};
}

void do_one(std::mt19937 &rand_gen,std::vector<point_t<coord_t>> buffer) {
    random_loop(rand_gen,buffer,LOOP_SIZE);

    if(mc__) {
        mc__->message(json::obj(
        json::attr("command") = "originalpoints",
            json::attr("points") = json::array_range(buffer | std::views::join)));
    }

    index_t ends[] = {LOOP_SIZE};
    input_coords = reinterpret_cast<coord_t (*)[2]>(buffer.data());
    basic_polygon<index_t,coord_t,point_t<coord_t>> polygon{buffer.data(),ends};
    offset_stroke_triangulate<index_t,coord_t>(polygon,50,40);
}

int run_my_message_server(const char *htmlfile,const char *supportdir,const std::mt19937 &rand_gen,int i) {
    graphical_debug = true;
    try {
        run_message_server(htmlfile,supportdir,[&,i](message_canvas &mc) {
            mc__ = &mc;
            /* the RNG state is copied so each invocation of this callback
            is the same */
            auto rand_gen_ = rand_gen;
            mc.console_line_stream() << "failure on iteration " << i;
            try {
                std::vector<point_t<coord_t>> loop;
                do_one(rand_gen_,loop);
            } catch(const std::exception &e) {
                mc.console_line(e.what());
            }
        });
    } catch(const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}

int main(int argc,char **argv) {
    if(argc != 3 && argc != 4) {
        std::cerr << "Usage: random_shapes FILENAME SUPPORTDIR [RANDSTATEFILE]" << std::endl;
        return 1;
    }

    std::mt19937 rand_gen;
    int i=0;
    const char *supportdir = argv[2][0] ? argv[2] : nullptr;
    if(argc == 3) {
        // try a bunch until one fails, then redo it interactively
        try {
            std::vector<point_t<coord_t>> loop;
            for(; i<10000; ++i) {
                std::mt19937 r_state = rand_gen;
                try {
                    do_one(rand_gen,loop);
                } catch(...) {
                    rand_gen = r_state;
                    throw;
                }
            }
        } catch(const assertion_failure &e) {
            { std::ofstream("last_rand_state") << i << '\n' << rand_gen; }
            std::cout << "assertion failure: " << e.what() << std::endl;
            return run_my_message_server(argv[1],supportdir,rand_gen,i);
        }
        std::cout << "all succeeded\n";
    } else {
        { std::ifstream(argv[3]) >> i >> rand_gen; }
        return run_my_message_server(argv[1],supportdir,rand_gen,i);
    }
    return 0;
}
