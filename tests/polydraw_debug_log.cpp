
#define POLY_OPS_DRAW_DEBUG_LOG(...) \
    (std::cout << std::format(__VA_ARGS__) << '\n')

#include <format>
#include <iostream>
#include <fstream>
#include <cerrno>
#include <cstring>

#include "../include/poly_ops/polydraw.hpp"
#include "stream_output.hpp"

using coord_t = int;


int main(int argc,char **argv) {
    if(argc != 2) {
        std::cerr << "exactly one argument is required\n";
        return 1;
    }

    std::ifstream is(argv[1]);
    if(!is.is_open()) {
        std::cerr << "failed to open " << argv[1] << ": " << std::strerror(errno) << '\n';
        return 1;
    }
    std::vector<std::vector<poly_ops::point_t<coord_t>>> loops;
    read_loops(is,loops);

    poly_ops::point_t<coord_t> max_dims(0,0);
    for(const auto &loop : loops) {
        for(const auto &p : loop) {
            max_dims.x() = std::max(max_dims.x(),p.x());
            max_dims.y() = std::max(max_dims.y(),p.y());
        }
    }

    poly_ops::draw::rasterizer<coord_t> rast;
        rast.add_loops(loops);
        rast.draw(
            static_cast<unsigned int>(max_dims.x()+1),
            static_cast<unsigned int>(max_dims.y()+1),
            [](unsigned int,unsigned int,unsigned int){});

    return 0;
}
