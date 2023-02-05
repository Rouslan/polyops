
#include <exception>
#include <vector>
#include <fstream>
#include <iostream>
#include <cassert>
#include <cstring>
#include <array>
#include <cmath>
#include <numbers>
#include <chrono>

#include "SDL.h"

#include "../include/poly_ops/poly_ops.hpp"

#define CHECK(X) do { int r = X; if(r) return r; } while(false)

typedef int32_t coord_t;
typedef uint16_t index_t;


int emit_failure(const char *pre) {
    std::cerr << pre << SDL_GetError() << '\n';
    return 1;
}

using vec3 = std::array<float,3>;
vec3 cross(const vec3 &a,const vec3 &b) {
    return {a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]};
}
vec3 vmul(const vec3 &a,float b) {
    return {a[0]*b,a[1]*b,a[2]*b};
}
vec3 vmul(float a,const vec3 &b) {
    return vmul(b,a);
}
vec3 vadd(const vec3 &a,const vec3 &b) {
    return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};
}

struct orbit {
    vec3 a;
    vec3 b;

    poly_ops::point_t<coord_t> operator()(double delta) const {
        auto r = vadd(vmul(a,std::sin(delta)),vmul(b,std::cos(delta)));
        return {static_cast<coord_t>(r[0]),static_cast<coord_t>(r[1])};
    }
};

using loops_t = std::vector<std::vector<poly_ops::point_t<coord_t>>>;

struct satellite {
    loops_t loops;
    orbit orb;

    satellite() = default;
    satellite(const orbit &o) : orb(o) {}
    satellite(satellite&&) = default;
};

using shapes_t = std::vector<satellite>;

shapes_t shapes;
loops_t center_shape;
std::vector<std::vector<SDL_FPoint>> line_buffer;


auto offset_loop(const std::vector<poly_ops::point_t<coord_t>> &loop,poly_ops::point_t<coord_t> offset) {
    offset[0] += 3000;
    offset[1] += 3000;
    return loop | std::views::transform([=](auto &p) { return p + offset; });
}

void update_scene(double delta) {
    line_buffer.clear();

    /* instead of allocating memory for each transformed input loop, we pass
    range views */
    std::vector<decltype(offset_loop(shapes[0].loops[0],{}))> input;
    for(auto &shape : shapes) {
        auto offset = shape.orb(delta);
        for(auto &loop : shape.loops) {
            input.push_back(offset_loop(loop,offset));
        }
    }

    for(auto &loop : center_shape) {
        input.push_back(offset_loop(loop,{0,0}));
    }

    auto loops = poly_ops::normalize<false,index_t,coord_t>(input);

    for(auto &&loop : loops) {
        assert(loop.size());

        line_buffer.emplace_back();
        line_buffer.back().reserve(loop.size()+1);
        for(auto &p : loop) line_buffer.back().emplace_back(p[0]/10.0f,p[1]/10.0f);
        auto first = *loop.begin();
        line_buffer.back().emplace_back(first[0]/10.0f,first[1]/10.0f);
    }
}

void execute_drawing(SDL_Renderer *renderer) {
    /*const char padding = 10;

    SDL_Rect view;
    SDL_RenderGetViewport(renderer,&view);

    float scale = std::min(float(view.w)/psize[0],float(view.h)/psize[1]);*/

    SDL_SetRenderDrawColor(renderer,255,255,255,255);
    SDL_RenderClear(renderer);

    SDL_SetRenderDrawColor(renderer,0,0,0,255);

    for(auto &loop : line_buffer) {
        SDL_RenderDrawLinesF(renderer,loop.data(),static_cast<int>(loop.size()));
    }

    SDL_RenderPresent(renderer);
}

int odd_coord_count() {
    std::cerr << "input has odd number of coordinates" << std::endl;
    return 2;
}

class loop_reader {
public:
    virtual void add_coord(const poly_ops::point_t<coord_t>&) = 0;
    virtual void next_loop() = 0;
    virtual int next_shape() = 0;
};

int read_loops(const char *path,loop_reader &reader) {
    std::ifstream is(path);
    if(!is.is_open()) {
        std::cerr << "failed to open " << path << ": " << std::strerror(errno) << std::endl;
        return 2;
    }

    poly_ops::point_t<coord_t> p;
    int c_count = 0;

    while(!is.bad()) {
        is >> std::ws;
        if(is.eof()) {
            if(c_count) return odd_coord_count();
            if(shapes.back().loops.back().empty()) shapes.back().loops.pop_back();
            if(shapes.back().loops.empty()) shapes.pop_back();
            return 0;
        }
        auto c = is.peek();
        if(c == '=') {
            if(c_count) return odd_coord_count();
            is.get();
            reader.next_loop();
        } else if(c == '#') {
            if(c_count) return odd_coord_count();
            is.get();
            CHECK(reader.next_shape());
        } else {
            is >> p[c_count++];
            if(is.fail()) {
                std::cerr << "invalid value in file" << std::endl;
                return 2;
            }
            if(c_count > 1) {
                c_count = 0;
                reader.add_coord(p);
            }
        }
    }
    std::cerr << "error reading file: " << std::strerror(errno) << std::endl;
    return 2;
}

class satellite_reader : public loop_reader {
    std::mt19937 rgen;
    std::uniform_real_distribution<float> rudist;
    std::normal_distribution<float> rndist;
    std::uniform_real_distribution<float> ru2dist;
    shapes_t &shapes;

    orbit rand_orbit() {
        float theta = rudist(rgen);
        vec3 a = {std::cos(theta),std::sin(theta),0.0f};
        vec3 out = {0.0f,0.0f,1.0f};
        vec3 right = cross(a,out);
        theta = rndist(rgen);
        vec3 b = vadd(vmul(right,std::cos(theta)),vmul(out,std::sin(theta)));

        float radius = ru2dist(rgen);
        return {vmul(a,radius),vmul(b,radius)};
    }

public:
    satellite_reader(shapes_t &shapes) :
        rudist(-std::numbers::pi_v<float>,std::numbers::pi_v<float>),
        rndist(0,0.8f),
        ru2dist(1000,3000),
        shapes(shapes)
    {
        shapes.emplace_back(rand_orbit());
        shapes.back().loops.emplace_back();
    }

    void add_coord(const poly_ops::point_t<coord_t> &p) override {
        shapes.back().loops.back().push_back(p);
    }
    void next_loop() override {
        if(!shapes.back().loops.back().empty()) shapes.back().loops.emplace_back();
    }
    int next_shape() override {
        if(shapes.back().loops.size() > 1 || !shapes.back().loops.back().empty()) {
            if(shapes.back().loops.back().empty()) shapes.back().loops.pop_back();
            shapes.emplace_back(rand_orbit());
            shapes.back().loops.emplace_back();
        }
        return 0;
    }
};

int read_satellites_from_file(shapes_t &shapes) {
    satellite_reader reader{shapes};
    CHECK(read_loops("alphabet.txt",reader));

    assert(!shapes.empty() && !shapes.back().loops.empty() && !shapes.back().loops.back().empty());
    if(shapes.back().loops.back().empty()) shapes.back().loops.pop_back();
    if(shapes.back().loops.empty()) shapes.pop_back();
    if(shapes.empty()) {
        std::cerr << "no data in \"alphabet.txt\"" << std::endl;
        return 2;
    }
    return 0;
}

class shape_reader : public loop_reader {
    loops_t &loops;

public:
    shape_reader(loops_t &loops) : loops(loops)
    {
        loops.emplace_back();
    }

    void add_coord(const poly_ops::point_t<coord_t> &p) override {
        loops.back().push_back(p);
    }
    void next_loop() override {
        if(!loops.back().empty()) loops.emplace_back();
    }
    int next_shape() override {
        std::cerr << "multiple shapes not allowed for \"center.txt\"" << std::endl;
        return 2;
    }
};

int read_shape_from_file(loops_t &shape) {
    shape_reader reader{shape};
    CHECK(read_loops("center.txt",reader));

    assert(!shape.empty() && !shape.back().empty());
    if(shape.back().empty()) shape.pop_back();
    if(shape.empty()) {
        std::cerr << "no data in \"center.txt\"" << std::endl;
        return 2;
    }
    return 0;
}

template<typename Fn> struct scope_exit {
    Fn f;
    scope_exit(Fn f) : f{f} {}
    ~scope_exit() noexcept(std::is_nothrow_invocable_v<Fn>) { f(); }
};

int main(int,char**) {
    CHECK(read_satellites_from_file(shapes));
    CHECK(read_shape_from_file(center_shape));

    update_scene(0);

    SDL_SetHint(SDL_HINT_VIDEO_ALLOW_SCREENSAVER,"1");

    if(SDL_Init(SDL_INIT_VIDEO) < 0) return emit_failure("Failed to initialize SDL: ");
    scope_exit _sdl_exit{SDL_Quit};

    SDL_Window *window = SDL_CreateWindow("Inset Test",SDL_WINDOWPOS_UNDEFINED,SDL_WINDOWPOS_UNDEFINED,800,600,SDL_WINDOW_RESIZABLE);
    if(!window) return emit_failure("Failed to create window: ");
    scope_exit _window_exit{[=]{ SDL_DestroyWindow(window); }};

    SDL_Renderer *renderer = SDL_CreateRenderer(window,-1,0);
    if(!renderer) return emit_failure("Failed to create renderer: ");
    scope_exit _renderer_exit{[=]{ SDL_DestroyRenderer(renderer); }};

    //if(create_number_font(renderer)) return emit_failure("Failed to create texture: ");
    //scope_exit _tex_exit{[]{ SDL_DestroyTexture(font_tex); }};

    execute_drawing(renderer);

    SDL_Event event;
    Uint32 window_id = SDL_GetWindowID(window);
    auto start_time = std::chrono::steady_clock::now();
    for(;;) {
        while(SDL_PollEvent(&event)) {
            switch(event.type) {
            /*case SDL_KEYDOWN:
                switch(event.key.keysym.sym) {
                case SDLK_SPACE:
                    do_task(task::continue_);
                    break;
                case SDLK_s:
                    do_task(task::dump_sweep);
                    break;
                case SDLK_o:
                    do_task(task::dump_orig_points);
                    break;
                case SDLK_n:
                    show_numbers = !show_numbers;
                    execute_drawing(renderer);
                    break;
                default:
                    break;
                }
                break;*/
            case SDL_WINDOWEVENT:
                if(event.window.windowID == window_id) {
                    switch(event.window.event) {
                    case SDL_WINDOWEVENT_SIZE_CHANGED:
                        execute_drawing(renderer);
                        break;
                    case SDL_WINDOWEVENT_CLOSE:
                        return 0;
                    default:
                        break;
                    }
                }
                break;
            case SDL_QUIT:
                return 0;
            default:
                break;
            }
        }

        update_scene(std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - start_time).count());
        execute_drawing(renderer);
    }
}
