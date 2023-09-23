#include <stdexcept>
#include <system_error>
#include <type_traits>

#include "bitmap.hpp"

class bad_bitmap : public std::runtime_error {
public:
    using runtime_error::runtime_error;
};

namespace {

unsigned char rev_bits(unsigned char x) {
    const unsigned char rev_nibble[16] = {0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    return static_cast<unsigned char>((rev_nibble[x & 0xf] << 4) | rev_nibble[x >> 4]);
}

void throw_invalid_pbm() {
    throw bad_bitmap("invalid PBM file");
}

char require_c(std::FILE *file) {
    int c = std::getc(file);
    if(c == EOF) {
        if(std::ferror(file)) throw std::system_error(errno,std::generic_category());
        throw_invalid_pbm();
    }
    return static_cast<char>(c);
}

void skip_comment(std::FILE *file) {
    char c;
    do { c = require_c(file); } while(c != '\n' && c != '\r');
}

bool is_blank(char c) {
    return c ==  ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

void require_blank(std::FILE *file) {
    bool found = false;
    char c;
    for(;;) {
        c = require_c(file);
        if(c == '#') skip_comment(file);
        else if(is_blank(c)) found = true;
        else break;
    }
    if(!found) throw_invalid_pbm();
    std::ungetc(c,file);
}

unsigned int require_uint(std::FILE *file) {
    unsigned int value;
    int count = std::fscanf(file,"%u",&value);
    if(count == EOF) throw std::system_error(errno,std::generic_category());
    if(count != 1) throw_invalid_pbm();
    return value;
}

void put_byte(bitmap::word_t &w,std::FILE *file) {
    if(std::putc(static_cast<char>(rev_bits(static_cast<unsigned char>(w))),file) == EOF) {
        throw std::system_error(errno,std::generic_category());
    }
    w >>= 8;
}

void take_byte(bitmap::word_t &w,unsigned int byte,std::FILE *file) {
    w |= static_cast<bitmap::word_t>(rev_bits(static_cast<unsigned char>(require_c(file)))) << (byte*8);
}

template<typename T> bool bit_test(T value,T mask) {
    return (value & mask) == mask;
}

}

void bitmap::set_row(unsigned int x1,unsigned int x2,unsigned int y) {
    assert(x1 < _width && x2 < _width && y < _height);
    
    if(x2 <= x1) return;

    unsigned int row_off = y * _stride;

    unsigned int floor_start = x1 / word_bit_size + row_off;
    unsigned int ceil_start = (x1 + word_bit_size - 1) / word_bit_size + row_off;
    unsigned int floor_end = x2 / word_bit_size + row_off;
    unsigned int ceil_end = (x2 + word_bit_size - 1) / word_bit_size + row_off;
    if(floor_start+1 == ceil_end) {
        word_t w = (word_t(1) << (x1 % word_bit_size)) - 1;
        if(x2 % word_bit_size) w ^= (word_t(1) << (x2 % word_bit_size)) - 1;
        else w = ~w;
        _data[floor_start] |= w;
    } else {
        if(x1 % word_bit_size) {
            _data[floor_start] |= ~((word_t(1) << (x1 % word_bit_size)) - 1);
        }
        for(unsigned int i=ceil_start; i < floor_end; ++i) _data[i] = ~word_t(0);
        if(x2 % word_bit_size) {
            _data[floor_end] |= (word_t(1) << (x2 % word_bit_size)) - 1;
        }
    }
}

bitmap operator^(const bitmap &a,const bitmap &b) {
    if(a._width != b._width || a._height != b._height)
        throw std::logic_error("'a' and 'b' must have the same dimensions");
    
    bitmap r{a._width,a._height};
    for(unsigned int i=0; i<a._stride*a._height; ++i) r._data[i] = a._data[i] ^ b._data[i];

    return r;
}

bitmap bitmap::from_pbm_file(const char *filename) {
    std::FILE *file = std::fopen(filename,"rb");
    if(!file) throw std::system_error(errno,std::generic_category());
    file_closer closer{file};

    try {
        if(require_c(file) != 'P' || require_c(file) != '4') throw std::runtime_error("not a binary PBM file");
        require_blank(file);

        unsigned int width = require_uint(file);
        require_blank(file);
        unsigned int height = require_uint(file);
        
        for(;;) {
            char c = require_c(file);
            if(c == '#') skip_comment(file);
            else if(is_blank(c)) break;
            else throw_invalid_pbm();
        }

        bitmap r{width,height};

        for(unsigned int y=0; y<height; ++y) {
            for(unsigned int xw=0; xw < width/word_bit_size; ++xw) {
                word_t &w = r.data()[y*r.stride() + xw];
                w = 0;
                for(unsigned int i=0; i<sizeof(word_t); ++i) take_byte(w,i,file);
            }
            if(unsigned int bits = width % word_bit_size) {
                word_t &w = r.data()[(y+1)*r.stride() - 1];
                w = 0;
                for(unsigned int i=0; i<(bits+7)/8; ++i) take_byte(w,i,file);
            }
        }

        return r;
    } catch(const bad_bitmap&) {
        throw bad_bitmap{std::string{"invalid PBM file: "} + filename};
    }
}

void bitmap::to_pbm_file(const char *filename) {
    std::FILE *file = std::fopen(filename,"wb");
    if(!file) throw std::system_error(errno,std::generic_category());
    file_closer closer{file};

    std::fprintf(file,"P4 %u %u\n",_width,_height);

    for(unsigned int y=0; y<_height; ++y) {
        for(unsigned int xw=0; xw < _width/word_bit_size; ++xw) {
            word_t w = _data[y*_stride + xw];
            for(unsigned int i=0; i<sizeof(word_t); ++i) put_byte(w,file);
        }
        if(unsigned int bits = _width % word_bit_size) {
            word_t w = _data[(y+1)*_stride - 1];
            for(unsigned int i=0; i<(bits+7)/8; ++i) put_byte(w,file);
        }
    }
}

bool bitmap::find_square(unsigned int size,unsigned int &_x,unsigned int &_y) const {
    assert(size > 0);
    for(unsigned int y=0; y < (_height-size+1); ++y) {
        for(unsigned int xw=0; xw < (_width-size+word_bit_size)/word_bit_size; ++xw) {
            if(!_data[y*_stride + xw] || (xw*word_bit_size + size - 1) > _width) continue;

            word_t mask = ~word_t(0);
            mask >>= word_bit_size - size;
            const unsigned int img_end = _width - (xw*word_bit_size + size - 1);
            unsigned int i=0;
            for(; i<std::min(word_bit_size - size,img_end); ++i) {
                for(unsigned int j=0; j<size; ++j) {
                    if(!bit_test(_data[(y+j)*_stride + xw],mask)) goto miss1;
                }
                _x = xw*word_bit_size + i;
                _y = y;
                return true;

              miss1:
                mask <<= 1;
            }
            word_t mask2 = 1;
            for(; i<std::min(word_bit_size,img_end); ++i) {
                for(unsigned int j=0; j<size; ++j) {
                    unsigned int w_i = (y+j)*_stride + xw;
                    if(!bit_test(_data[w_i],mask) || !bit_test(_data[w_i + 1],mask2)) goto miss2;
                }
                _x = xw*word_bit_size + i;
                _y = y;
                return true;

              miss2:
                mask <<= 1;
                mask2 = (word_t(1) << (i + size - word_bit_size)) - 1;
            }
        }
    }
    return false;
}

bool bitmap::has_square(unsigned int size) const {
    unsigned int x,y;
    return find_square(size,x,y);
}
