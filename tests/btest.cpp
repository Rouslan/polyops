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
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <optional>
#include <string>
#include <filesystem>
#include <system_error>
#include <functional>
#include <iostream>
#include <charconv>

#include "bitmap.hpp"
#include "../include/poly_ops/polydraw.hpp"
#include "../include/poly_ops/base.hpp"
#include "../include/poly_ops/clip.hpp"
#include "../include/poly_ops/offset.hpp"


template<typename T> constexpr T MAX_IMG_DIM = 1000;
constexpr std::size_t MAX_INT_STR_SIZE = 9;

/* big enough to store a UUID if needed */
constexpr std::size_t MAX_ID_STR_SIZE = 36;

constexpr unsigned int DIFF_FAIL_SQUARE_SIZE = 5;

constexpr int OFFSET_ARC_STEP_SIZE = 3;

constexpr int TEST_ALL = -1;
constexpr int TEST_OFFSET = -2;


using coord_t = long;

#include "stream_output.hpp"

bool dump_failure_diff = false;

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

struct op_file_t {
    int offset;
    std::string file;
};

template<typename Coord> struct test_case {
    std::vector<std::vector<poly_ops::point_t<Coord>>> subject;
    std::vector<std::vector<poly_ops::point_t<Coord>>> clip;
    std::vector<op_file_t> op_files[static_cast<int>(poly_ops::bool_op::normalize)+1];
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

    void read_field_name(const char *end) {
        while(*end) {
            if(*end++ != next_c()) {
                throw parse_input_error{"unknown field name on line %i",line};
            }
        }
        next_c();
        if((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
            throw parse_input_error{"unknown field name on line %i",line};
        }
    }

    void read_colon() {
        skip_whitespace();
        if(c != ':') unexpected_char();
        next_c();
        skip_whitespace();
    }

    void read_op_entry(const char *end,poly_ops::bool_op op) {
        read_field_name(end);

        int offset = 0;
        if(c == '+' || c == '-') offset = read_int();

        read_colon();

        auto &file_set = tests.back().op_files[static_cast<int>(op)];
        auto i = std::ranges::lower_bound(file_set,offset,{},[](const op_file_t &o){ return o.offset; });
        if(i != file_set.end() && i->offset == offset) *i = {offset,read_id()};
        else file_set.emplace(i,offset,read_id());
        skip_whitespace();
        read_newline();
        touched = true;
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
                read_field_name("ubject");
                read_colon();
                tests.back().subject.push_back(read_point_array());
                touched = true;
                break;
            case 'c':
                read_field_name("lip");
                read_colon();
                tests.back().clip.push_back(read_point_array());
                touched = true;
                break;
            case 'u':
                read_op_entry("nion",poly_ops::bool_op::union_);
                break;
            case 'd':
                read_op_entry("ifference",poly_ops::bool_op::difference);
                break;
            case 'i':
                read_op_entry("ntersection",poly_ops::bool_op::intersection);
                break;
            case 'x':
                read_op_entry("or",poly_ops::bool_op::xor_);
                break;
            case 'n':
                read_op_entry("ormalize",poly_ops::bool_op::normalize);
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
    const op_file_t &entry,
    const std::filesystem::path &folder)
{
    auto target_img = bitmap::from_pbm_file((folder / (entry.file + ".pbm")).c_str());
    bitmap test_img{target_img.width(),target_img.height()};
    test_img.clear();

    if(entry.offset) {
        add_offset_loops_subject(clip,test.subject,entry.offset,OFFSET_ARC_STEP_SIZE);
        add_offset_loops_clip(clip,test.clip,entry.offset,OFFSET_ARC_STEP_SIZE);
    } else {
        clip.add_loops_subject(test.subject);
        clip.add_loops_clip(test.clip);
    }

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
            entry.file.c_str());
        if(dump_failure_diff) {
            test_img.to_pbm_file((entry.file + "_mine.pbm").c_str());
            diff.to_pbm_file((entry.file + "_diff.pbm").c_str());
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

bool str_to_bool(const char *str) {
    if(!str) return false;
    int x;
    const char *end = str+std::strlen(str);
    auto r = std::from_chars(str,end,x);
    return r.ptr == end && x;
}

int main(int argc,char *argv[]) {
    dump_failure_diff = str_to_bool(std::getenv("DUMP_FAILURE_DIFF"));

    if(argc != 2 && argc != 3) {
        std::fprintf(stderr,"exactly one or two arguments are required\n");
        return 2;
    }

    int operation = TEST_ALL;
    if(argc > 2) {
        if(std::strcmp("offset",argv[2]) == 0) {
            operation = TEST_OFFSET;
            goto found;
        }
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
                    if(operation > -1 && operation != i) continue;
                    for(auto &op_file : test.op_files[i]) {
                        if(operation == TEST_ALL || (op_file.offset != 0) == (operation == TEST_OFFSET)) {
                            successes += test_op(clip,rast,test,static_cast<poly_ops::bool_op>(i),op_file,test_source);
                            tests += 1;
                        }
                    }
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
