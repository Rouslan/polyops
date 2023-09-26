/* this is a 1-bit image buffer, not the Microsoft file format */

#ifndef bitmap_hpp
#define bitmap_hpp

#include <memory>
#include <cassert>
#include <cstdio>


struct file_closer {
    std::FILE *file;
    file_closer(std::FILE *file) : file{file} {}
    ~file_closer() {
        std::fclose(file);
    }
};

/* A two-dimensional packed 1-bit array.

The least significant bit is the first bit. */
class bitmap {
public:
    using word_t = unsigned long;
    static constexpr unsigned int word_bit_size = sizeof(word_t) * 8;

private:
    unsigned int _width;
    unsigned int _height;
    unsigned int _stride; // how many word_t values make up a row
    std::unique_ptr<word_t[]> _data;

public:
    bitmap(unsigned int width,unsigned int height)
        : _width{width}, _height{height}, _stride((width + word_bit_size - 1)/word_bit_size), _data{new word_t[_stride * height]} {}
    
    unsigned int width() const { return _width; }
    unsigned int height() const { return _height; }
    unsigned int stride() const { return _stride; }
    word_t *data() { return _data.get(); }
    const word_t *data() const { return _data.get(); }

    void set(unsigned int x,unsigned int y) {
        assert(x < _width && y < _height);
        _data[y * _stride + x/word_bit_size] |= (word_t(1) << (x % word_bit_size));
    }
    void clear(unsigned int x,unsigned int y) {
        assert(x < _width && y < _height);
        _data[y * _stride + x/word_bit_size] &= ~(word_t(1) << (x % word_bit_size));
    }
    void set() {
        for(std::size_t i=0; i<_stride*_height; ++i) _data[i] = ~word_t(0);
    }
    void clear() {
        for(std::size_t i=0; i<_stride*_height; ++i) _data[i] = 0;
    }
    bool get(unsigned int x,unsigned int y) const {
        assert(x < _width && y < _height);
        return (_data[y * _stride + x/word_bit_size] & (word_t(1) << (x % word_bit_size))) != 0;
    }

    void set_row(unsigned int x1,unsigned int x2,unsigned int y);

    friend bitmap operator^(const bitmap &a,const bitmap &b);

    /* This is almost a conforming binary PBM parser. It only allows comments
    around the mandatory whitespace characters, not before the magic number or
    in the middle of tokens. */
    static bitmap from_pbm_file(std::FILE *file);
    static bitmap from_pbm_file(const char *filename);

    void to_pbm_file(std::FILE *file);
    void to_pbm_file(const char *filename);

#ifdef _MSC_VER
    static bitmap from_pbm_file(const wchar_t *filename);
    void to_pbm_file(const wchar_t *filename);
#endif

    bool find_square(unsigned int size,unsigned int &x,unsigned int &y) const;
    bool has_square(unsigned int size) const;
};

#endif
