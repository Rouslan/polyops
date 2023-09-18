/* Generate a random image for which img.has_square(SQUARE_SIZE) returns false
but is mostly filled */

#include <random>

#include "bitmap.hpp"

constexpr unsigned int IMG_WIDTH = 100;
constexpr unsigned int IMG_HEIGHT = 100;
constexpr unsigned int SQUARE_SIZE = 5;

void clear_in_square(bitmap &img,unsigned int x,unsigned int y,unsigned int offset) {
    img.clear(x + offset%SQUARE_SIZE,y + offset/SQUARE_SIZE);
}

int main(int argc,char *argv[]) {
    if(argc != 2) {
        std::fprintf(stderr,"exactly one argument is required\n");
        return 1;
    }

    bitmap img{IMG_WIDTH,IMG_HEIGHT};
    img.set();

    std::default_random_engine re{std::random_device{}()};
    std::uniform_int_distribution<unsigned int> dist{0,SQUARE_SIZE*SQUARE_SIZE-1};
    for(unsigned int y=0; y<(IMG_HEIGHT-SQUARE_SIZE+1); y+=SQUARE_SIZE) {
        for(unsigned int x=0; x<(IMG_WIDTH-SQUARE_SIZE+1); x+=SQUARE_SIZE) {
            clear_in_square(img,x,y,dist(re));
        }
    }

    for(;;) {
        unsigned int x, y;
        if(!img.find_square(SQUARE_SIZE,x,y)) break;
        clear_in_square(img,x,y,dist(re));
    }

    img.to_pbm_file(argv[1]);
    return 0;
}
