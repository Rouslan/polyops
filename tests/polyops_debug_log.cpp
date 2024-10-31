
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


int parse_op(const char *arg,poly_ops::bool_op &op) {
    if(std::strcmp(arg,"union")) {
        op = poly_ops::bool_op::union_;
    /*} else if(std::strcmp(arg,"intersection")) {
        op = poly_ops::bool_op::intersection;
    } else if(std::strcmp(arg,"xor")) {
        op = poly_ops::bool_op::xor_;
    } else if(std::strcmp(arg,"difference")) {
        op = poly_ops::bool_op::difference;*/
    } else if(std::strcmp(arg,"normalize")) {
        op = poly_ops::bool_op::normalize;
    } else {
        std::cerr << R"(operation must be "union" or "normalize")" "\n";
        return 1;
    }
    return 0;
}

int main(int argc,char **argv) {
    if(argc != 2 && argc != 3) {
        std::cerr << "one or two arguments are required\n";
        return 1;
    }

    poly_ops::bool_op op = poly_ops::bool_op::normalize;
    if(argc > 2 && parse_op(argv[2],op)) return 1;

    std::ifstream is(argv[1]);
    if(!is.is_open()) {
        std::cerr << "failed to open " << argv[1] << ": " << std::strerror(errno) << '\n';
        return 1;
    }
    std::vector<std::vector<poly_ops::point_t<coord_t>>> loops;
    read_loops(is,loops);

    std::cout << poly_ops::boolean_op<true,int,std::size_t>(loops,std::span<poly_ops::point_t<coord_t>>{},op).size() << std::endl;

    return 0;
}
