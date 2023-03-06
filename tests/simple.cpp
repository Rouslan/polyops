#include <vector>
#include <iostream>

#include "../include/poly_ops/poly_ops.hpp"

void random_loop(std::mt19937 &rand_gen,std::vector<poly_ops::point_t<int>> &loop,unsigned int size) {
    std::uniform_int_distribution<int> dist(0,100);
    loop.resize(size);
    for(unsigned int i=0; i<size; ++i) loop[i] = {dist(rand_gen),dist(rand_gen)};
}

int main() {
    std::vector<poly_ops::point_t<int>> loop_a{
        {58,42},
        {36,52},
        {20,34},
        {32,13},
        {55,18}};
    std::vector<poly_ops::point_t<int>> loop_b/*{
        {76,58},
        {43,56},
        {45,23},
        {78,26}}*/;
    
    poly_ops::clipper<unsigned int,int> clip;
    std::mt19937 rand_gen;

    for(int i=0; i<100000; ++i) {
        random_loop(rand_gen,loop_b,4);
        clip.add_loop_subject(loop_a);
        clip.add_loop_clip(loop_b);
        clip.execute(poly_ops::bool_op::difference);
        for(auto &&loop : clip.get_output<false>()) {
            for(auto &&p : loop) {
                std::cout << p[0] << ',' << p[1] << ' ';
            }
            std::cout << '\n';
        }
    }

    return 0;
}
