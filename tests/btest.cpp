/* Test the algorithm by painting the output and comparing to pre-rendered
images of the same operations done per-pixel.

The results will not be exactly the same because the coordinates of the added
points are rounded to integers, thus a comparison is only considered a failure
when a continuous 5x5 pixel area doesn't match.

This program runs tests specified in an ASCII text file with the following
grammar (using the same notation as W3C:
https://www.w3.org/TR/REC-xml/#sec-notation):

Start       ::= (Line? (LineEnd Line?)*)?

Line        ::= Command S? Comment? | S Comment? | Comment

Command     ::= ArrayAssign | OpAssign | Divider

ArrayAssign ::= Set S? ":" S? "[" (S? Point (S Point))? S? "]"

OpAssign    ::= Operation S? ":" S? ID

Divider     ::= "-"+

Point       ::= Integer S Integer

ID          ::= [0-9a-zA-Z]+

LineEnd     ::= "\r"? "\n" | "\r"

Comment     ::= "#" [^\n\r]*

Set         ::= "subject" | "clip"

Operation   ::= "union" | "difference" | "intersection" | "xor" | "normalize"

Integer     ::= [+-]? [0-9]+

S           ::= [ \t]+

*/

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <optional>
#include <string>
#include <filesystem>
#include <system_error>
#include <functional>
#include <iostream>

#include "bitmap.hpp"
#include "../include/poly_ops/polydraw.hpp"
#include "../include/poly_ops/base.hpp"
#include "../include/poly_ops/clip.hpp"


template<typename T> constexpr T MAX_IMG_DIM = 1000;
constexpr std::size_t MAX_INT_STR_SIZE = 9;

/* big enough to store a UUID if needed */
constexpr std::size_t MAX_ID_STR_SIZE = 36;

constexpr unsigned int DIFF_FAIL_SQUARE_SIZE = 5;
constexpr bool DUMP_FAILURE_DIFF = false;


using coord_t = long;

#include "stream_output.hpp"


class parse_input_error : public std::exception {
public:
    const char *msg;
    int line;
    
    parse_input_error(const char *msg) : msg{msg}, line{-1} {}
    parse_input_error(const char *msg,unsigned int line) : msg{msg}, line{static_cast<int>(line)} {}

    const char *what() const noexcept override { return "bad input"; }

    void print(std::FILE *file) const noexcept {
        if(line >= 0) std::fprintf(file,msg,line);
        else std::fputs(msg,file);
    }
};

template<typename Coord> struct test_case {
    std::vector<std::vector<poly_ops::point_t<Coord>>> subject;
    std::vector<std::vector<poly_ops::point_t<Coord>>> clip;
    std::string op_files[static_cast<int>(poly_ops::bool_op::normalize)+1];
};

template<typename Coord> struct parse_state {
    std::vector<test_case<Coord>> tests;

    std::FILE *file;
    unsigned int line;
    int c;
    bool touched;

    int next_c() {
        c = std::getc(file);
        if(c == EOF && std::ferror(file)) throw std::system_error(errno,std::generic_category());
        return c;
    }

    void unexpected_char() {
        switch(c) {
        case EOF:
            throw parse_input_error{"unexpected end of file"};
        case '\n':
        case '\r':
            throw parse_input_error{"unexpected line end at line %i",line};
        default:
            throw parse_input_error{"unexpected character at line %i",line};
        }
    }

    void read_newline() {
        switch(c) {
        case '\n':
            line += 1;
            next_c();
            break;
        case '\r':
            line += 1;
            next_c();
            if(c == '\n') next_c();
            break;
        case EOF:
            break;
        default:
            unexpected_char();
        }
    }

    void skip_line() {
        for(;;) {
            switch(c) {
            case '\n':
            case '\r':
            case EOF:
                return;
            default:
                next_c();
                break;
            }
        }
    }

    void skip_whitespace(bool allow_newline=false) {
        for(;;) {
            switch(c) {
            case ' ':
            case '\t':
                next_c();
                break;
            case '#':
                skip_line();
                if(!allow_newline) return;
                break;
            case '\n':
                if(!allow_newline) return;
                line += 1;
                next_c();
                break;
            case '\r':
                if(!allow_newline) return;
                line += 1;
                next_c();
                if(c == '\n') next_c();
                break;
            default:
                return;
            }
        }
    }

    void read_field(const char *name) {
        while(*name) {
            if(*name++ != next_c()) {
                throw parse_input_error{"unknown field name on line %i",line};
            }
        }
        next_c();
        if((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
            throw parse_input_error{"unknown field name on line %i",line};
        }

        skip_whitespace();
        if(c != ':') unexpected_char();
        next_c();
        skip_whitespace();
    }

    unsigned int read_uint() {
        char digits[MAX_INT_STR_SIZE];
        std::size_t str_size = 0;

        while(c >= '0' && c <= '9') {
            if(str_size == MAX_INT_STR_SIZE) throw parse_input_error{"too many digits in number at line %i",line};
            digits[str_size++] = char(c);
            next_c();
        }

        if(str_size == 0) unexpected_char();

        unsigned int value = 0;
        unsigned int factor = 1;
        while(str_size > 0) {
            --str_size;
            value += static_cast<unsigned int>(digits[str_size] - '0') * factor;
            factor *= 10;
        }
        return value;
    }

    int read_int() {
        bool neg = false;
        if(c == '-') {
            next_c();
            neg = true;
        } else if(c == '+') {
            next_c();
        }

        int value = static_cast<int>(read_uint());
        return neg ? -value : value;
    }

    std::string read_id() {
        char value[MAX_ID_STR_SIZE];
        std::size_t str_size = 0;

        while(c >= '0' && c <= '9') {
            if(str_size == MAX_ID_STR_SIZE) throw parse_input_error{"too many characters in ID at line %i",line};
            value[str_size++] = char(c);
            next_c();
        }
        return {value,str_size};
    }

    std::vector<poly_ops::point_t<Coord>> read_point_array() {
        std::vector<poly_ops::point_t<Coord>> points;

        if(c != '[') unexpected_char();
        next_c();

        poly_ops::point_t<Coord> p;
        for(;;) {
            skip_whitespace(true);
            if(c == ']') {
                next_c();
                return points;
            }
            p.x() = read_int();

            skip_whitespace(true);
            if(c == ']') throw parse_input_error{"array has odd number of items at line %i",line};
            p.y() = read_int();

            points.push_back(p);
        }
    }

    void run(std::FILE *_file) {
        file = _file;
        line = 1;
        touched = false;
        next_c();
        tests.emplace_back();

        for(;;) {
            switch(c) {
            case 's':
                read_field("ubject");
                tests.back().subject.push_back(read_point_array());
                touched = true;
                break;
            case 'c':
                read_field("lip");
                tests.back().clip.push_back(read_point_array());
                touched = true;
                break;
            case 'u':
                read_field("nion");
                tests.back().op_files[static_cast<int>(poly_ops::bool_op::union_)] = read_id();
                skip_whitespace();
                read_newline();
                touched = true;
                break;
            case 'd':
                read_field("ifference");
                tests.back().op_files[static_cast<int>(poly_ops::bool_op::difference)] = read_id();
                skip_whitespace();
                read_newline();
                touched = true;
                break;
            case 'i':
                read_field("ntersection");
                tests.back().op_files[static_cast<int>(poly_ops::bool_op::intersection)] = read_id();
                skip_whitespace();
                read_newline();
                touched = true;
                break;
            case 'x':
                read_field("or");
                tests.back().op_files[static_cast<int>(poly_ops::bool_op::xor_)] = read_id();
                skip_whitespace();
                read_newline();
                touched = true;
                break;
            case 'n':
                read_field("ormalize");
                tests.back().op_files[static_cast<int>(poly_ops::bool_op::normalize)] = read_id();
                skip_whitespace();
                read_newline();
                touched = true;
                break;
            case '-':
                do {
                    next_c();
                } while(c == '-');
                skip_whitespace();
                read_newline();
                if(touched) {
                    tests.emplace_back();
                    touched = false;
                }
                break;
            case EOF:
                if(!touched) tests.pop_back();
                return;
            case ' ':
            case '\t':
                skip_whitespace();
                read_newline();
                break;
            case '\n':
            case '\r':
                read_newline();
                break;
            case '#':
                skip_line();
                read_newline();
                break;
            default:
                unexpected_char();
            }
        }
    }
};

const char *bool_op_names[] = {
    "union",
    "intersection",
    "xor",
    "difference",
    "normalize"};
constexpr int OP_COUNT = static_cast<int>(std::extent_v<decltype(bool_op_names)>);

bool test_op(
    poly_ops::clipper<coord_t> &clip,
    poly_ops::draw::rasterizer<coord_t> &rast,
    const test_case<coord_t> &test,
    poly_ops::bool_op op,
    const std::string &file_id,
    const std::filesystem::path &folder)
{
    auto target_img = bitmap::from_pbm_file((folder / (file_id + ".pbm")).c_str());
    bitmap test_img{target_img.width(),target_img.height()};
    test_img.clear();
    
    clip.add_loops_subject(test.subject);
    clip.add_loops_clip(test.clip);

    auto clipped = clip.execute<false>(op);
    //for(auto loop : clipped) {
    //    std::cout << '[' << delimited(loop) << "]\n";
    //}

    rast.clear();
    rast.add_loops(clipped);
    rast.draw(
        test_img.width(),
        test_img.height(),
        std::bind_front(&bitmap::set_row,std::ref(test_img)),
        poly_ops::draw::fill_rule_t::positive);

    auto diff = test_img ^ target_img;
    if(diff.has_square(DIFF_FAIL_SQUARE_SIZE)) {
        std::fprintf(
            stderr,
            "failure with operation \"%s\" compared to file \"%s\"\n",
            bool_op_names[static_cast<int>(op)],
            file_id.c_str());
        if(DUMP_FAILURE_DIFF) {
            test_img.to_pbm_file((file_id + "_mine.pbm").c_str());
            diff.to_pbm_file((file_id + "_diff.pbm").c_str());
        }
        return false;
    }
    return true;
}

/* make sure bitmap::has_square works */
void check_square_finder(const std::filesystem::path &folder) {
    if(bitmap::from_pbm_file((folder / ("discont.pbm")).c_str()).has_square(DIFF_FAIL_SQUARE_SIZE)) {
        throw std::logic_error{"bitmap::has_square is faulty; found square where none should exist"};
    }
    if(!bitmap::from_pbm_file((folder / ("cont.pbm")).c_str()).has_square(DIFF_FAIL_SQUARE_SIZE)) {
        throw std::logic_error{"bitmap::has_square is faulty; did not find square where one should exist"};
    }
}

int main(int argc,char *argv[]) {
    if(argc != 2 && argc != 3) {
        std::fprintf(stderr,"exactly one or two arguments are required\n");
        return 2;
    }

    int operation = -1;
    if(argc > 2) {
        for(int i=0; i<OP_COUNT; ++i) {
            if(std::strcmp(bool_op_names[i],argv[2]) == 0) {
                operation = i;
                goto found;
            }
        }
        if(std::strcmp("all",argv[2]) != 0) {
            std::fprintf(stderr,"invalid operation name\n");
            return 2;
        }
        found: ;
    }

    std::size_t tests = 0;
    std::size_t successes = 0;

    try {
        parse_state<coord_t> state;

        std::FILE *file = std::fopen(argv[1],"rb");
        if(!file) throw std::system_error(errno,std::generic_category());

        {
            file_closer closer{file};
            state.run(file);
        }

        auto test_source = std::filesystem::path(argv[1]).remove_filename();

        check_square_finder(test_source);

        poly_ops::clipper<coord_t> clip;
        poly_ops::draw::rasterizer<coord_t> rast;

        for(auto &test : state.tests) {
            bool has_one = false;
            for(int i=0; i<OP_COUNT; ++i) {
                if(!test.op_files[i].empty()) {
                    has_one = true;
                    if(operation != -1 && operation != i) continue;
                    successes += test_op(clip,rast,test,static_cast<poly_ops::bool_op>(i),test.op_files[i],test_source);
                    tests += 1;
                }
            }
            if(!has_one) {
                std::fputs("warning: test entry found with nothing to test against\n",stderr);
            }
        }
        std::fprintf(stderr,"passed tests: %zu out of %zu\n",successes,tests);
    } catch(const parse_input_error &e) {
        e.print(stderr);
        std::fputc('\n',stderr);
        return 2;
    } catch(const std::exception &e) {
        std::fputs(e.what(),stderr);
        std::fputc('\n',stderr);
        return 3;
    }

    return successes != tests;
}
