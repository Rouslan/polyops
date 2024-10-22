
#define POLY_OPS_DEBUG_LOG(...) \
    (std::cout << std::format(__VA_ARGS__) << '\n')

#include <format>
#include <iostream>
#include <fstream>
#include <cerrno>
#include <cstring>

#include "stream_output.hpp"
#include "../include/poly_ops/poly_ops.hpp"


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

    std::cout << poly_ops::normalize_op<true,int,std::size_t>(loops).size() << std::endl;

    return 0;
}
