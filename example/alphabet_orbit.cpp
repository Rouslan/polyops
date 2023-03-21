
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

    satellite(loops_t &&loops,const orbit &o) : loops(std::move(loops)), orb(o) {}
    satellite(satellite&&) = default;
};

using shapes_t = std::vector<satellite>;

struct scene {
    std::vector<satellite> shapes;
    loops_t center_shape;
    std::vector<std::vector<SDL_FPoint>> line_buffer;
    poly_ops::clipper<coord_t,index_t> clip;
    poly_ops::bool_op operation = poly_ops::bool_op::union_;

    void update(double delta) {
        line_buffer.clear();
        clip.reset();

        for(auto &shape : shapes) {
            auto offset = shape.orb(delta);
            offset[0] += 3000;
            offset[1] += 3000;
            for(auto &loop : shape.loops) {
                clip.add_loop_subject(loop | std::views::transform([=](auto &p) { return p + offset; }));
            }
        }

        for(auto &loop : center_shape) {
            clip.add_loop_clip(loop | std::views::transform([=](auto &p) { return p + poly_ops::point_t<coord_t>(3000,3000); }));
        }

        auto loops = clip.execute<false>(operation);

        for(auto &&loop : loops) {
            assert(loop.size());

            line_buffer.emplace_back();
            line_buffer.back().reserve(loop.size()+1);
            for(auto &p : loop) line_buffer.back().emplace_back(p[0]/10.0f,p[1]/10.0f);
            auto first = *loop.begin();
            line_buffer.back().emplace_back(first[0]/10.0f,first[1]/10.0f);
        }
    }
};

void draw_scene(scene &sc,SDL_Renderer *renderer) {
    /*const char padding = 10;

    SDL_Rect view;
    SDL_RenderGetViewport(renderer,&view);

    float scale = std::min(float(view.w)/psize[0],float(view.h)/psize[1]);*/

    SDL_SetRenderDrawColor(renderer,255,255,255,255);
    SDL_RenderClear(renderer);

    SDL_SetRenderDrawColor(renderer,0,0,0,255);

    for(auto &loop : sc.line_buffer) {
        SDL_RenderDrawLinesF(renderer,loop.data(),static_cast<int>(loop.size()));
    }

    SDL_RenderPresent(renderer);
}

int odd_coord_count() {
    std::cerr << "input has odd number of coordinates" << std::endl;
    return 2;
}

int read_loops(const char *path,std::vector<loops_t> &shapes) {
    std::ifstream is(path);
    if(!is.is_open()) {
        std::cerr << "failed to open " << path << ": " << std::strerror(errno) << std::endl;
        return 2;
    }

    shapes.emplace_back();
    shapes.back().emplace_back();

    poly_ops::point_t<coord_t> p;
    int c_count = 0;

    while(!is.bad()) {
        is >> std::ws;
        if(is.eof()) {
            if(c_count) return odd_coord_count();
            if(shapes.back().back().empty()) shapes.back().pop_back();
            if(shapes.back().empty()) shapes.pop_back();

            if(shapes.empty()) {
                std::cerr << "no data in \"" << path << '"' << std::endl;
                return 2;
            }
            return 0;
        }
        auto c = is.peek();
        if(c == '=') {
            if(c_count) return odd_coord_count();
            is.get();
            if(!shapes.back().back().empty()) shapes.back().emplace_back();
        } else if(c == '#') {
            if(c_count) return odd_coord_count();
            is.get();
            if(shapes.back().size() > 1 || !shapes.back().back().empty()) {
                if(shapes.back().back().empty()) shapes.back().pop_back();
                shapes.emplace_back();
                shapes.back().emplace_back();
            }
        } else {
            is >> p[c_count++];
            if(is.fail()) {
                std::cerr << "invalid value in \"" << path << '"' << std::endl;
                return 2;
            }
            if(c_count > 1) {
                c_count = 0;
                shapes.back().back().push_back(p);
            }
        }
    }
    std::cerr << "error reading \"" << path << "\": " << std::strerror(errno) << std::endl;
    return 2;
}

class rand_orbit_gen {
    std::mt19937 rgen;
    std::uniform_real_distribution<float> rudist;
    std::normal_distribution<float> rndist;
    std::uniform_real_distribution<float> ru2dist;

public:
    rand_orbit_gen() :
        rudist(-std::numbers::pi_v<float>,std::numbers::pi_v<float>),
        rndist(0,0.8f),
        ru2dist(1000,3000) {}

    orbit operator()() {
        float theta = rudist(rgen);
        vec3 a = {std::cos(theta),std::sin(theta),0.0f};
        vec3 out = {0.0f,0.0f,1.0f};
        vec3 right = cross(a,out);
        theta = rndist(rgen);
        vec3 b = vadd(vmul(right,std::cos(theta)),vmul(out,std::sin(theta)));

        float radius = ru2dist(rgen);
        return {vmul(a,radius),vmul(b,radius)};
    }
};

int read_satellites_from_file(std::vector<satellite> &shapes) {
    std::vector<loops_t> raw_shapes;
    CHECK(read_loops("alphabet.txt",raw_shapes));

    rand_orbit_gen ogen;
    shapes.reserve(raw_shapes.size());
    for(auto &rs : raw_shapes) shapes.emplace_back(std::move(rs),ogen());
    return 0;
}

int read_shape_from_file(loops_t &shape) {
    std::vector<loops_t> raw_shapes;
    CHECK(read_loops("center.txt",raw_shapes));

    if(raw_shapes.size() != 1) {
        std::cerr << "multiple shapes are not allowed for \"center.txt\"" << std::endl;
        return 2;
    }

    shape = std::move(raw_shapes[0]);
    return 0;
}

template<typename Fn> struct scope_exit {
    Fn f;
    scope_exit(Fn f) : f{f} {}
    ~scope_exit() noexcept(std::is_nothrow_invocable_v<Fn>) { f(); }
};

int main(int,char**) {
    scene sc;
    CHECK(read_satellites_from_file(sc.shapes));
    CHECK(read_shape_from_file(sc.center_shape));

    SDL_SetHint(SDL_HINT_VIDEO_ALLOW_SCREENSAVER,"1");

    if(SDL_Init(SDL_INIT_VIDEO) < 0) return emit_failure("Failed to initialize SDL: ");
    scope_exit _sdl_exit{SDL_Quit};

    SDL_Window *window = SDL_CreateWindow("Inset Test",SDL_WINDOWPOS_UNDEFINED,SDL_WINDOWPOS_UNDEFINED,800,600,SDL_WINDOW_RESIZABLE);
    if(!window) return emit_failure("Failed to create window: ");
    scope_exit _window_exit{[=]{ SDL_DestroyWindow(window); }};

    SDL_Renderer *renderer = SDL_CreateRenderer(window,-1,0);
    if(!renderer) return emit_failure("Failed to create renderer: ");
    scope_exit _renderer_exit{[=]{ SDL_DestroyRenderer(renderer); }};

    SDL_Event event;
    Uint32 window_id = SDL_GetWindowID(window);
    auto start_time = std::chrono::steady_clock::now();
    for(;;) {
        while(SDL_PollEvent(&event)) {
            switch(event.type) {
            case SDL_KEYDOWN:
                if(event.key.keysym.sym == SDLK_SPACE) {
                    int o = static_cast<int>(sc.operation) + 1;
                    if(o > static_cast<int>(poly_ops::bool_op::difference))
                        o = static_cast<int>(poly_ops::bool_op::union_);
                    sc.operation = static_cast<poly_ops::bool_op>(o);
                }
                break;
            case SDL_WINDOWEVENT:
                if(event.window.windowID == window_id) {
                    switch(event.window.event) {
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

        sc.update(std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - start_time).count());
        draw_scene(sc,renderer);
    }
}
