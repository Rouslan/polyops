#define DEBUG_STEP_BY_STEP 1

constexpr bool graphical_debug = true;

#include graphical_test_common.hpp


int main(int argc,char **argv) {
    if(argc != 2) {
        std::cerr << "Usage: inset_test FILENAME" << std::endl;
        return 1;
    }

    try {
        run_message_server(argv[1],[](message_canvas &mc) {
            cblob data = mc.get_binary();
            if(data.size % sizeof(coord_t)) throw std::runtime_error("invalid data received");
            mc__ = &mc;
            input_sizes[0] = data.size / sizeof(coord_t[2]);
            std::unique_ptr<coord_t[2]> _input_coords(new coord_t[input_sizes[0]][2]);
            std::memcpy(_input_coords.get(),data.data,data.size);
            input_coords = _input_coords.get();
            offset_stroke_triangulate<index_t,coord_t>(
                basic_polygon<index_t,coord_t,coord_t[2]>(
                    input_coords,
                    std::span(input_sizes)),
                -140,
                80);
        });
    } catch(const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
