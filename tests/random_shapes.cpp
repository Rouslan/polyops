
#include <iostream>
#include <fstream>
#include <random>
#include <map>
#include <ranges>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <cerrno>

#define DEBUG_STEP_BY_STEP 1

#include "graphical_test_common.hpp"


constexpr size_t DEFAULT_LOOP_SIZE = 5;

enum class op_type {unset,normalize,offset};
struct settings_t {
    op_type operation;
    long loop_size;
    long skip;
    long timeout;
    bool interactive;
    std::unique_ptr<char[]> htmlfile;
    std::unique_ptr<char[]> supportdir;
    std::unique_ptr<char[]> failout;
    std::unique_ptr<char[]> datain;

    settings_t() : operation(op_type::unset), loop_size(-1), skip(-1), timeout(-1), interactive(false) {}
};

void random_loop(std::mt19937 &rand_gen,std::vector<point_t<coord_t>> &loop,index_t size) {
    std::uniform_int_distribution<index_t> dist(0,1000);
    loop.resize(size);
    for(index_t i=0; i<size; ++i) loop[i] = {dist(rand_gen),dist(rand_gen)};
}

void do_one(op_type type,std::span<const std::vector<point_t<coord_t>>> loops) {
    if(type == op_type::normalize) {
        normalize<false,index_t,coord_t>(loops);
    } else {
        if(mc__) {
            mc__->message(json::obj(
            json::attr("command") = "originalpoints",
                json::attr("points") = json::array_range(loops[0] | std::views::join)));
        }

        offset<false,index_t,coord_t>(loops,50,40);
    }
}

int run_my_message_server(const settings_t &settings,std::span<const std::vector<point_t<coord_t>>> loops,int i=-1) {
    graphical_debug = true;
    try {
        run_message_server(settings.htmlfile.get(),settings.supportdir.get(),[=,&settings](message_canvas &mc) {
            mc__ = &mc;
            if(i != -1) mc.console_line_stream() << "failure on iteration " << i;
            try {
                do_one(settings.operation,loops);
            } catch(const std::exception &e) {
                mc.console_line(e.what());
            }
        });
    } catch(const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}

bool show_help() {
    std::cout << R"(Usage: random_shapes [OPTIONS] [HTMLFILE [SUPPORTDIR]]

OPTIONS:
-d FILENAME
    Path to file containing input to use instead of generating data randomly.

-f FILENAME
    If a test failed, dump the input data to this path. This is ignored if "-d"
    or "-i" is specified.

-h, -?, --help
    Show this message.

-i
    Start in interactive mode without first checking if the input fails.

-o OPERATION
    Operation to perform. This can be either "normalize" or "offset". The
    default is "offset".

-p INTEGER
    A positive integer specifying how many points to generate for random tests.
    The default is 5. This is ignored if "-d" is specified.

-s INTEGER
    A non-negative integer specifying how many random tests to skip. The tests
    are not run but the pseudo-random number generator is advanced as if they
    were. This is ignored if "-d" is specified.

-t INTEGER
    A positive integer specifying a time limit per test in milliseconds, or 0
    for no time limit. By default there is no time limit. This is ignored if
    "-i" is specified.
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

bool handle_pos_val(settings_t &settings,const char *val) {
    if(settings.htmlfile) {
        if(settings.supportdir) {
            return parse_fail();
        }
        copy_str_into(settings.supportdir,val);
    } else {
        copy_str_into(settings.htmlfile,val);
    }
    return true;
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
            case 'o':
                if(!handle_operation_val(settings,argv,end)) return false;
                break;
            case 'i':
                settings.interactive = true;
                break;
            case 's':
                if(!handle_nonneg_int_val(settings.skip,argv,end)) return false;
                if(settings.skip < 0) return false;
                break;
            case 't':
                if(!handle_nonneg_int_val(settings.timeout,argv,end)) return false;
                break;
            default:
                return parse_fail();
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

int read_loops_from_file(const settings_t &settings,std::vector<std::vector<poly_ops::point_t<coord_t>>> &loops) {
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

int run_from_file(const settings_t &settings) {
    std::vector<std::vector<poly_ops::point_t<coord_t>>> loops;
    if(int r=read_loops_from_file(settings,loops)) return r;

    try {
        enable_timeout(settings);
        do_one(settings.operation,loops);
    } catch(const test_failure &e) {
        std::cout << e << std::endl;
        if(!settings.htmlfile) return 1;
        return run_my_message_server(settings,loops);
    }

    std::cout << "all succeeded\n";
    return 0;
}

int run_from_file_interactive(const settings_t &settings) {
    std::vector<std::vector<poly_ops::point_t<coord_t>>> loops;
    if(int r=read_loops_from_file(settings,loops)) return r;
    return run_my_message_server(settings,loops);
}

void write_fail_file(const settings_t &settings,std::span<const std::vector<point_t<coord_t>>> loops) {
    if(settings.failout) {
        std::ofstream out(settings.failout.get());
        if(out.is_open()) {
            write_loops(out,loops);
        } else {
            std::cerr << "cannot write to " << settings.failout << ": " << std::strerror(errno) << std::endl;
            errno = 0;
        }
    }
}

int run_random(const settings_t &settings) {
    std::vector<point_t<coord_t>> loop;
    long i=0;
    std::mt19937 rand_gen;
    
    try {
        for(; i<100000000; ++i) {
            long tests = i - settings.skip;
            if(tests > 0 && tests % 1000 == 0)
                std::cout << "completed " << tests << " tests" << std::endl;
            random_loop(rand_gen,loop,settings.loop_size);
            if(i >= settings.skip) {
                enable_timeout(settings);
                do_one(settings.operation,std::span(&loop,1));
            }
        }
    } catch(const test_failure &e) {
        write_fail_file(settings,std::span(&loop,1));
        std::cout << e << std::endl;
        if(!settings.htmlfile) return 1;
        return run_my_message_server(settings,std::span(&loop,1),i);
    }
    std::cout << "all succeeded\n";
    return 0;
}

int run_random_interactive(const settings_t &settings) {
    std::vector<point_t<coord_t>> loop;
    int i=0;
    std::mt19937 rand_gen;
    
    for(; i<1+settings.skip; ++i) {
        random_loop(rand_gen,loop,settings.loop_size);
    }

    return run_my_message_server(settings,std::span(&loop,1),i);
}

int main(int argc,char **argv) {
    settings_t settings;
    if(argc < 1 || !parse_command_line(settings,argv+1,argv+argc)) return 1;

    if(settings.loop_size < 0) settings.loop_size = DEFAULT_LOOP_SIZE;
    if(settings.skip < 0) settings.skip = 0;
    if(settings.timeout < 0) settings.timeout = 0;

    if(settings.interactive) {
        if(!settings.htmlfile) {
            std::cerr << "HTMLFILE must be specified to run interactively" << std::endl;
            return 1;
        }
        if(settings.datain) {
            if(int r=run_from_file_interactive(settings)) return r;
        } else {
            if(int r=run_random_interactive(settings)) return r;
        }
    } else {
        if(settings.datain) {
            if(int r=run_from_file(settings)) return r;
        } else {
            if(int r=run_random(settings)) return r;
        }
    }
    
    return 0;
}
