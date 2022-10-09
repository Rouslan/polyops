#ifndef server_hpp
#define server_hpp

#include <functional>
#include <iterator>
#include <ranges>
#include <string_view>
#include <concepts>
#include <tuple>
#include <type_traits>
#include <ostream>

#if defined _WIN32 || defined __CYGWIN__
  #ifdef SERVER_CPP
    #define VISIBLE __declspec(dllexport)
  #else
    #define VISIBLE __declspec(dllimport)
  #endif
#elif __GNUC__
  #define VISIBLE __attribute__ ((visibility ("default")))
#else
  #define VISIBLE
#endif

class message_canvas;

namespace json {
    namespace detail {
        template<typename T> struct is_obj : std::false_type {};
    }

    template<typename T> concept json_value =
        detail::is_obj<T>::value ||
        std::same_as<T,std::string_view> ||
        std::floating_point<T> ||
        std::integral<T> ||
        std::same_as<T,bool> ||
        std::same_as<T,decltype(nullptr)>;

    namespace detail {
        template<typename T> struct array { T values; };
        template<json_value... T> struct is_obj<array<std::tuple<T...>>> : std::true_type {};

        template<typename T> concept has_to_json_value =
            requires(T x) {
                { to_json_value(x) } -> json_value;
            };

        template<typename T> requires json_value<std::remove_cvref_t<T>> constexpr T normalize(T &&x) { return x; }
        constexpr auto normalize(decltype(nullptr) x) { return x; }
        template<std::convertible_to<std::string_view> T>
        constexpr std::string_view normalize(const T &x) { return x; }

        template<has_to_json_value T> constexpr auto normalize(T &&x) {
            return to_json_value(std::forward<T>(x));
        }

        template<typename... T> constexpr auto normalize(const std::tuple<T...> &x);
        template<typename... T> constexpr auto normalize(std::tuple<T...> &&x);

        template<typename T> using normalized = decltype(normalize(std::declval<T>()));

        template<std::ranges::input_range R>
        requires json_value<detail::normalized<std::ranges::range_value_t<R>>>
        struct is_obj<array<R>> : std::true_type {};

        template<json_value T> struct attr_value {
            std::string_view name;
            T value;
        };

        template<json_value... T> struct obj {
            std::tuple<attr_value<T>...> values;
            explicit constexpr obj(attr_value<T>... values) : values{values...} {}
        };
        template<typename... T> struct is_obj<obj<T...>> : std::true_type {};
    }

    template<typename T> concept json_value_convertable = json_value<detail::normalized<T>>;

    namespace detail {
        struct array_tuple_t {
            template<json_value_convertable... T>
            constexpr auto operator()(const T&... values) {
                return detail::array{std::tuple(normalize(values)...)};
            }
        };
        template<typename... T> constexpr auto normalize(const std::tuple<T...> &x) {
            return std::apply(array_tuple_t{},x);
        }
        template<typename... T> constexpr auto normalize(std::tuple<T...> &&x) {
            return std::apply(array_tuple_t{},x);
        }
    }

    struct attr {
        std::string_view name;

        template<json_value_convertable T>
        constexpr detail::attr_value<detail::normalized<T>> operator=(const T &val) const {
            return {name,detail::normalize(val)};
        }
    };

    template<json_value... T> constexpr auto obj(detail::attr_value<T>... values) {
        return detail::obj<T...>{values...};
    }

    constexpr inline detail::array_tuple_t array_tuple;

    template<std::ranges::input_range R>
    requires json_value_convertable<std::ranges::range_value_t<R>>
    constexpr auto array_range(R &&r) {
        return detail::array<R>{std::forward<R>(r)};
    }

    /* escapes all input to an underlying stream buffer */
    class escaped_stream_buf : public std::streambuf {
        friend class ::message_canvas;

        std::streambuf *_raw;
        std::streambuf *raw();

        escaped_stream_buf() : _raw(nullptr) {}
    protected:
        std::streamsize xsputn(const char_type *s,std::streamsize count) override;
        int_type overflow(int_type ch=traits_type::eof()) override;

        escaped_stream_buf(const escaped_stream_buf&) = delete;
        escaped_stream_buf &operator=(const escaped_stream_buf&) = delete;
    };
}

struct cblob {
    const void *data;
    size_t size;
};

class websocket_session;

class message_canvas {
    websocket_session &data;
    json::escaped_stream_buf esb;

    VISIBLE void emit_raw(char);
    VISIBLE void emit_value(bool);
    VISIBLE void emit_value(decltype(nullptr));
    VISIBLE void emit_value(long);
    VISIBLE void emit_value(unsigned long);
    VISIBLE void emit_value(float);
    VISIBLE void emit_value(double);
    VISIBLE void emit_value(std::string_view);

    void emit_value(int x) { return emit_value(static_cast<long>(x)); }
    void emit_value(unsigned int x) { return emit_value(static_cast<unsigned long>(x)); }

    void emit_value(short x) { return emit_value(static_cast<long>(x)); }
    void emit_value(unsigned short x) { return emit_value(static_cast<long>(x)); }

    template<typename T> void emit_value(json::detail::attr_value<T> value) {
        emit_value(value.name);
        emit_raw(':');
        emit_value(value.value);
    }

    void emit_delimited() const {}
    template<typename T> void emit_delimited(const T &value) { emit_value(value); }
    template<typename T1,typename T2,typename... T> void emit_delimited(const T1 &value1,const T2 &value2,const T&... values) {
        emit_value(value1);
        emit_raw(',');
        emit_delimited(value2,values...);
    }
    constexpr auto delimiter() {
        return [this]<typename... T>(const T&... values){ emit_delimited(values...); };
    }

    template<typename... T> void emit_value(const json::detail::obj<T...> &obj) {
        emit_raw('{');
        std::apply(delimiter(),obj.values);
        emit_raw('}');
    }

    template<typename... T> void emit_array(const std::tuple<T...> &values) {
        std::apply(delimiter(),values);
    }
    template<std::ranges::input_range R> void emit_array(const R &values) {
        bool started = false;
        for(auto &&v : values) {
            if(started) emit_raw(',');
            started = true;
            emit_value(json::detail::normalize(v));
        }
    }
    template<typename T> void emit_value(const json::detail::array<T> &obj) {
        emit_raw('[');
        emit_array(obj.values);
        emit_raw(']');
    }

    VISIBLE void send();
    VISIBLE void check_ready();

public:
    class console_line_stream_obj : public std::ostream {
        friend class message_canvas;

        message_canvas &owner;

        VISIBLE console_line_stream_obj(message_canvas &owner);
    public:
        VISIBLE ~console_line_stream_obj();

        console_line_stream_obj(const console_line_stream_obj&) = delete;
        console_line_stream_obj &operator=(const console_line_stream_obj&) = delete;
    };

    message_canvas(websocket_session &data) : data(data) {}
    VISIBLE ~message_canvas();

    template<json::json_value_convertable T>
    void message(T m) {
        check_ready();
        emit_value(json::detail::normalize(m));
        send();
    }

    VISIBLE void console_line(std::string_view msg);

    VISIBLE console_line_stream_obj console_line_stream();

    VISIBLE std::u8string_view get_text();

    VISIBLE cblob get_binary();
};

using input_fun = std::function<void(message_canvas&)>;

VISIBLE void run_message_server(const char *html_file,const char *support_dir,input_fun fun);

#undef VISIBLE

#endif
