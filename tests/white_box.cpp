
#include <iostream>
#include <fstream>
#include <random>
#include <map>
#include <ranges>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <chrono>
#include <span>


#define POLY_OPS_ASSERT(X) if(!(X)) throw assertion_failure{#X,__FILE__,__LINE__}
#define POLY_OPS_ASSERT_SLOW POLY_OPS_ASSERT

#define POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_F
#define POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_FR
#define POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_B
#define POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_BR
#define POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_CALC_BALANCE
#define POLY_OPS_DEBUG_STEP_BY_STEP_EVENT_CALC_SAMPLE
#define POLY_OPS_DEBUG_STEP_BY_STEP_MISSED_INTR (void)0

#define POLY_OPS_DEBUG_ITERATION if(timeout && std::chrono::steady_clock::now() > timeout_expiry) { \
    timeout = false; \
    throw timed_out{}; \
}

constexpr unsigned int COORD_MAX = 1000;

struct test_failure : std::exception {
    virtual std::ostream &emit(std::ostream &os) const = 0;
    const char *what() const noexcept override {
        return "test failure";
    }
};
struct assertion_failure : test_failure {
    const char *assertion_str;
    const char *filename;
    int lineno;
    assertion_failure(const char *assertion,const char *filename,int lineno)
        : assertion_str{assertion}, filename{filename}, lineno{lineno} {}

    std::ostream &emit(std::ostream &os) const override {
        return os << "assertion failure at " << filename << " on line " << lineno << ": " << assertion_str;
    }
};
struct timed_out : test_failure {
    std::ostream &emit(std::ostream &os) const override {
        return os << "test timed out";
    }
};
std::ostream &operator<<(std::ostream &os,const test_failure &f) {
    return f.emit(os);
}

bool timeout = false;
std::chrono::steady_clock::time_point timeout_expiry;

#include "../include/poly_ops/poly_ops.hpp"
#include "../include/poly_ops/polydraw.hpp"
#include "stream_output.hpp"


constexpr long DEFAULT_LOOP_SIZE = 5;
constexpr long DEFAULT_LOOP_COUNT = 1;
constexpr long DEFAULT_TEST_COUNT = 1000000;


enum class op_type {unset,normalize,offset,union_,all,draw};
struct settings_t {
    op_type operation;
    long loop_size;
    long loop_count;
    long timeout;
    long test_count;
    std::unique_ptr<char[]> failout;
    std::unique_ptr<char[]> datain;

    settings_t() : operation(op_type::unset), loop_size(-1), loop_count(-1), timeout(-1), test_count(-1) {}
};

template<typename Coord,typename Index>
void random_loop(std::mt19937 &rand_gen,std::vector<poly_ops::point_t<Coord>> &loop,Index size) {
    std::uniform_int_distribution<Coord> dist(0,COORD_MAX);
    loop.resize(static_cast<std::size_t>(size));
    for(Index i=0; i<size; ++i) loop[i] = {dist(rand_gen),dist(rand_gen)};
}

template<typename Coord,typename Index>
void do_one(op_type type,std::span<const std::vector<poly_ops::point_t<Coord>>> loops) {
    bool all = type == op_type::all;
    if(all || type == op_type::normalize) {
        poly_ops::normalize_op<false,Coord,Index>(loops);
    }
    if(all || type == op_type::union_) {
        poly_ops::union_op<false,Coord,Index>(loops);
    }
    if(all || type == op_type::offset) {
        poly_ops::offset<false,Coord,Index>(loops,50,40);
    }
    if(type == op_type::draw) {
        poly_ops::draw::rasterizer<Coord> rast;
        rast.add_loops(loops);
        rast.draw(
            COORD_MAX,
            COORD_MAX,
            [](unsigned int,unsigned int,unsigned int){});
    }
}

bool show_help() {
    std::cout << R"(Usage: random_shapes [OPTIONS]

Run white-box tests for poly_ops::clipper class

OPTIONS:
-d FILENAME
    Path to file containing input to use instead of generating data randomly.

-f FILENAME
    If a test failed, dump the input data to this path. This is ignored if "-d"
    is specified.

-h, -?, --help
    Show this message.

-o OPERATION
    Operation to perform. This can be either "normalize", "offset", "union",
    "all" or "draw". The default is "all".
    
    "draw" is a special case. Instead of poly_ops::clipper,
    poly_ops::draw::rasterizer is tested. "all" does not include "draw".

-p INTEGER
    A positive integer specifying how many points per loop to generate for
    random tests. The default is 5. This is ignored if "-d" is specified.

-t INTEGER
    A positive integer specifying a time limit per test in milliseconds, or 0
    for no time limit. By default there is no time limit.

-n INTEGER
    A positive integer specifying how many times to run the test for each
    integer size (16, 32 and 64 bits). New data is randomly generated each
    time. The default is 1 million. This is ignored if "-d" is specified.

-l INTEGER
    A positive integer specifying how many loops to generate for random tests.
    The default is 1. This is ignored if "-d" is specified.
)";
    return false;
}

bool parse_fail() {
    std::cerr << "invalid command line arguments\n\n";
    return show_help();
}

void copy_str_into(std::unique_ptr<char[]> &dest,const char *src) {
    size_t s = std::strlen(src);
    dest.reset(new char[s+1]);
    std::memcpy(dest.get(),src,s);
    dest[s] = 0;
}

bool handle_pos_val(settings_t&,const char*) {
    std::cerr << "this program does not take any positional arguments\n\n";
    return show_help();
}

char *get_named_val(char **(&argv),char **end) {
    if((*argv)[2] == 0) {
        ++argv;
        if(argv == end) {
            parse_fail();
            return nullptr;
        }
        return *argv;
    }
    return *argv + 2;
}

bool handle_str_val(std::unique_ptr<char[]> &dest,char **(&argv),char **end) {
    if(dest) return parse_fail();
    char *val = get_named_val(argv,end);
    if(!val) return false;
    copy_str_into(dest,val);
    return true;
}

bool handle_nonneg_int_val(long &value,char **(&argv),char **end) {
    if(value >= 0) return parse_fail();
    const char *val = get_named_val(argv,end);
    if(!val) return false;

    char *val_end;
    value = std::strtol(val,&val_end,10);
    if(*val_end != 0 || value <= 0) return parse_fail();

    return true;
}

bool handle_operation_val(settings_t &settings,char **(&argv),char **end) {
    if(settings.operation != op_type::unset) return parse_fail();
    char *val = get_named_val(argv,end);
    if(!val) return false;
    if(strcmp(val,"normalize") == 0) {
        settings.operation = op_type::normalize;
        return true;
    }
    if(strcmp(val,"offset") == 0) {
        settings.operation = op_type::offset;
        return true;
    }
    if(strcmp(val,"union") == 0) {
        settings.operation = op_type::union_;
        return true;
    }
    if(strcmp(val,"all") == 0) {
        settings.operation = op_type::all;
        return true;
    }
    if(strcmp(val,"draw") == 0) {
        settings.operation = op_type::draw;
        return true;
    }

    return parse_fail();
}

bool parse_command_line(settings_t &settings,char **argv,char **end) {
    for(; argv != end; ++argv) {
        switch((*argv)[0]) {
        case '-':
            switch((*argv)[1]) {
            case '-':
                if((*argv)[2] == 0) {
                    ++argv;
                    for(; argv != end; ++argv) {
                        if(!handle_pos_val(settings,*argv)) return false;
                    }
                    return true;
                }

                if(strcmp(*argv+2,"help") != 0) return parse_fail();
                [[fallthrough]];
            case 'h':
            case '?':
                return show_help();
            case 'f':
                if(!handle_str_val(settings.failout,argv,end)) return false;
                break;
            case 'd':
                if(!handle_str_val(settings.datain,argv,end)) return false;
                break;
            case 'p':
                if(!handle_nonneg_int_val(settings.loop_size,argv,end)) return false;
                break;
            case 'n':
                if(!handle_nonneg_int_val(settings.test_count,argv,end)) return false;
                break;
            case 'o':
                if(!handle_operation_val(settings,argv,end)) return false;
                break;
            case 't':
                if(!handle_nonneg_int_val(settings.timeout,argv,end)) return false;
                break;
            case 'l':
                if(!handle_nonneg_int_val(settings.loop_count,argv,end)) return false;
                break;
            default:
                std::cerr << "unknown option \"" << (*argv)[1] << "\"\n\n";
                return show_help();
            }
            break;
        case 0:
            return parse_fail();
        default:
            if(!handle_pos_val(settings,*argv)) return false;
            break;
        }
    }
    return true;
}

template<typename Coord>
int read_loops_from_file(const settings_t &settings,std::vector<std::vector<poly_ops::point_t<Coord>>> &loops) {
    std::ifstream is(settings.datain.get());
    if(!is.is_open()) {
        std::cerr << "failed to open " << settings.datain << ": " << std::strerror(errno) << std::endl;
        return 2;
    }
    read_loops(is,loops);
    return 0;
}

void enable_timeout(const settings_t &settings) {
    if(settings.timeout) {
        timeout = true;
        timeout_expiry = std::chrono::steady_clock::now() + std::chrono::milliseconds(settings.timeout);
    }
}

template<typename Coord,typename Index> int run_from_file(const settings_t &settings) {
    std::vector<std::vector<poly_ops::point_t<Coord>>> loops;
    if(int r=read_loops_from_file(settings,loops)) return r;

    try {
        enable_timeout(settings);
        do_one<Coord,Index>(settings.operation,loops);
    } catch(const test_failure &e) {
        std::cout << e << '\n';
        return 1;
    }

    std::cout << "all succeeded\n";
    return 0;
}

template<typename Coord> void write_fail_file(const settings_t &settings,std::span<const std::vector<poly_ops::point_t<Coord>>> loops) {
    if(settings.failout) {
        std::ofstream out(settings.failout.get());
        if(out.is_open()) {
            write_loops<Coord>(out,loops);
        } else {
            std::cerr << "cannot write to " << settings.failout << ": " << std::strerror(errno) << std::endl;
            errno = 0;
        }
    }
}

template<typename Coord,typename Index> int run_random(const settings_t &settings) {
    std::vector<std::vector<poly_ops::point_t<Coord>>> loops;
    loops.resize(static_cast<std::size_t>(settings.loop_count));
    long i=0;
    std::mt19937 rand_gen;
    
    try {
        while(i<settings.test_count) {
            for(std::size_t j=0; j<static_cast<std::size_t>(settings.loop_count); ++j) {
                random_loop(rand_gen,loops[j],static_cast<Index>(settings.loop_size));
            }
            enable_timeout(settings);
            do_one<Coord,Index>(settings.operation,loops);
            ++i;
            if(i > 0 && i % 1000 == 0)
                std::cout << "completed " << i << " tests\n";
        }
    } catch(const test_failure &e) {
        write_fail_file<Coord>(settings,loops);
        std::cout << e << '\n';
        return 1;
    }
    std::cout << "all succeeded\n";
    return 0;
}

template<typename Coord,typename Index> int run_with_param(const settings_t &settings) {
    std::cout << "running with Coord size = " << sizeof(Coord) << " and Index size = " << sizeof(Index) << '\n';
    if(settings.datain) {
        return run_from_file<Coord,Index>(settings);
    }
    return run_random<Coord,Index>(settings);
}

int main(int argc,char **argv) {
    settings_t settings;
    if(argc < 1 || !parse_command_line(settings,argv+1,argv+argc)) return 1;

    if(settings.loop_size < 0) settings.loop_size = DEFAULT_LOOP_SIZE;
    if(settings.loop_count < 0) settings.loop_count = DEFAULT_LOOP_COUNT;
    if(settings.timeout < 0) settings.timeout = 0;
    if(settings.test_count < 0) settings.test_count = DEFAULT_TEST_COUNT;
    if(settings.operation == op_type::unset) settings.operation = op_type::all;

    if(int r=run_with_param<std::int16_t,std::uint16_t>(settings)) return r;
    if(int r=run_with_param<std::int32_t,std::uint32_t>(settings)) return r;
    return run_with_param<std::int64_t,std::uint64_t>(settings);
}
