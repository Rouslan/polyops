
#define BOOST_BEAST_USE_STD_STRING_VIEW 1
#define BOOST_ASIO_HAS_STD_INVOKE_RESULT
#define SERVER_CPP

#include <boost/predef.h>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/signal_set.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <exception>
#include <string_view>
#include <span>
#include <functional>
#include <cstdlib>
#include <stdexcept>
#include <streambuf>

#include "server.hpp"

#if BOOST_OS_WINDOWS
#define DIR_SEPARATOR '\\'
#else
#define DIR_SEPARATOR '/'
#endif

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;
namespace http = boost::beast::http;

//extern const char server_html_content[];
//extern const std::size_t server_html_content_size;

namespace {
void report_error(const char *msg) {
    std::cerr << "Error: " << msg << std::endl;
}
void report_error(const std::string &msg) {
    std::cerr << "Error: " << msg << std::endl;
}
void report_error(beast::error_code ec) {
    report_error(ec.message());
}

void post_error(auto exc,beast::error_code ec) {
    net::post(exc,[=]{ report_error(ec); });
}

void post_error(auto exc,const std::string &msg) {
    net::post(exc,[=]{ report_error(msg); });
}
}

using my_http_request = http::request<http::string_body>;

/* Each WebSocket session runs on a seperate thread so that "fun" can be
suspended and resumed like a coroutine without requiring it to be one. */
class websocket_session {
    friend class message_canvas;

    websocket::stream<tcp::socket> ws;
    beast::flat_buffer ibuffer;
    std::ostringstream obuffer;

    std::thread thread;
    std::mutex mut;
    std::condition_variable cond;

    /* needed to keep ourselves alive between an async handler and the next
    async call */
    std::unique_ptr<websocket_session> self;

    unsigned long cond_version;
    bool quitting;

    struct quit_exc {};

    static void continue_(std::unique_ptr<websocket_session> &&self) {
        auto ptr = self.get();
        {
            std::unique_lock ul(ptr->mut);
            ptr->self = std::move(self);
            ++ptr->cond_version;
        }
        ptr->cond.notify_all();
    }

    void thread_run(input_fun fun) {
        try {
            message_canvas mc(*this);
            fun(mc);

            for(;;) {
                ibuffer.consume(ibuffer.size());

                read(ibuffer);

                ws.text(ws.got_text());
                write(ibuffer.data());
            }
        } catch(const std::exception &e) {
            post_error(ws.get_executor(),e.what());
        } catch(quit_exc) {}
    }

    void cond_wait(std::unique_lock<std::mutex> &ul) {
        cond.wait(ul,[last_v=cond_version,this]{ return last_v != cond_version; });
        if(quitting) throw quit_exc{};
    }

    template<typename Buffer> void read(Buffer &buffer) {
        std::unique_lock ul(mut);
        ws.async_read(
            buffer,
            [self=std::move(self)](beast::error_code ec,size_t) mutable {
                if(ec != websocket::error::closed) {
                    if(ec) return post_error(self->ws.get_executor(),ec);
                    continue_(std::move(self));
                }
            });
        cond_wait(ul);
    }

    template<typename Buffer> void write(const Buffer &buffer) {
        std::unique_lock ul(mut);
        ws.async_write(
            buffer,
            [self=std::move(self)](beast::error_code ec,size_t) mutable {
                if(ec) return post_error(self->ws.get_executor(),ec);
                continue_(std::move(self));
            });
        cond_wait(ul);
    }

    void deliver_obuffer() {
        ws.text(true);
        write(net::buffer(obuffer.str().data(),obuffer.tellp()));
        obuffer.seekp(0);
    }

    void quit() {
        {
            std::unique_lock ul(mut);
            ++cond_version;
            quitting = true;
        }
        cond.notify_all();
    }

public:
    explicit websocket_session(tcp::socket &&socket)
        : ws(std::move(socket)),
          cond_version(0),
          quitting(false) {}

    void run(my_http_request &&req,input_fun fun,std::unique_ptr<websocket_session> &&self) {
        ws.async_accept(
            req,
            [=,self=std::move(self)](beast::error_code ec) mutable {
                if(ec) return post_error(self->ws.get_executor(),ec);
                auto ptr = self.get();
                ptr->self = std::move(self);
                ptr->thread = std::thread(&websocket_session::thread_run,ptr,fun);
            });
    }

    ~websocket_session() {
        if(thread.joinable()) {
            quit();
            thread.join();
        }
    }
};

namespace {
template<typename Body,typename Allocator>
http::response<http::string_body> bad_request(
    const http::request<Body,http::basic_fields<Allocator>> &req,
    std::string_view why)
{
    http::response<http::string_body> res{http::status::bad_request,req.version()};
    res.set(http::field::server,BOOST_BEAST_VERSION_STRING);
    res.set(http::field::content_type,"text/html");
    res.keep_alive(req.keep_alive());
    res.body() = std::string(why);
    res.prepare_payload();
    return res;
}

template<typename Body,typename Allocator>
http::response<http::string_body> not_found(
    const http::request<Body,http::basic_fields<Allocator>> &req)
{
    http::response<http::string_body> res{http::status::not_found,req.version()};
    res.set(http::field::server,BOOST_BEAST_VERSION_STRING);
    res.set(http::field::content_type,"text/html");
    res.keep_alive(req.keep_alive());
    res.body() = "The resource '" + std::string(req.target()) + "' was not found.";
    res.prepare_payload();
    return res;
}

template<typename Body,typename Allocator>
http::response<http::string_body> server_error(
    const http::request<Body,http::basic_fields<Allocator>> &req,
    std::string_view what)
{
    http::response<http::string_body> res{http::status::internal_server_error,req.version()};
    res.set(http::field::server,BOOST_BEAST_VERSION_STRING);
    res.set(http::field::content_type,"text/html");
    res.keep_alive(req.keep_alive());
    res.body() = "An error occurred: '" + std::string(what) + "'";
    res.prepare_payload();
    return res;
}

// this requires suffixes to be lower-case
std::string_view mime_type(std::string_view path) {
    using namespace std::literals;

    if(path.ends_with(".html"sv)) return "text/html";
    if(path.ends_with(".css"sv)) return "text/css";
    if(path.ends_with(".js"sv)) return "application/javascript";
    return "application/text";
}

// only very basic paths are supported
bool allowed_path(std::string_view path) {
    assert(!(path.empty() || path == "/"));
    if(path[0] != '/') return false;
    if(path[1] == '.') return false;
    for(size_t i=1; i<path.size(); ++i) {
        char c = path[i];
        if(!((c >= '0' && c <= '9')
            || (c >= 'A' && c <= 'Z')
            || (c >= 'a' && c <= 'z')
            || c == '_'
            || c == '.')) return false;
    }
    return true;
}

template<typename Body,typename Allocator,typename Send>
void handle_request(
    const char *html_file,
    const char *support_dir,
    http::request<Body,http::basic_fields<Allocator>> &&req,
    Send &&send)
{
    if(req.method() != http::verb::get && req.method() != http::verb::head)
        return send(bad_request(req,"Unknown HTTP-method"));

    std::string path;
    std::string_view mime;
    if(req.target().empty() || req.target() == "/") {
        path = html_file;
        mime = "text/html";
    } else {
        if(!support_dir || !allowed_path(req.target())) return send(not_found(req));
        path = support_dir;
        if(path.size() && path.back() != DIR_SEPARATOR) path += DIR_SEPARATOR;
        path += req.target();
        mime = mime_type(req.target().substr(1));
    }

    beast::error_code ec;
    http::file_body::value_type body;
    body.open(path.c_str(),beast::file_mode::scan,ec);

    if(ec == boost::system::errc::no_such_file_or_directory)
        return send(not_found(req));
    if(ec) return send(server_error(req,ec.message()));

    const auto bsize = body.size();

    if(req.method() == http::verb::head) {
        http::response<http::empty_body> res{http::status::ok,req.version()};
        res.set(http::field::server,BOOST_BEAST_VERSION_STRING);
        res.set(http::field::content_type,mime);
        res.content_length(bsize);
        res.keep_alive(req.keep_alive());
        return send(std::move(res));
    }

    http::response<http::file_body> res{
        http::status::ok,
        req.version(),
        std::move(body)};
    res.set(http::field::server,BOOST_BEAST_VERSION_STRING);
    res.set(http::field::content_type,mime);
    res.content_length(bsize);
    res.keep_alive(req.keep_alive());
    return send(std::move(res));
}

class http_session : public std::enable_shared_from_this<http_session> {
    const char *html_file;
    const char *support_dir;
    tcp::socket socket;
    input_fun fun;
    beast::flat_buffer buffer;
    http::request<http::string_body> req;
    std::shared_ptr<void> res;

public:
    http_session(const char *html_file,const char *support_dir,tcp::socket socket,input_fun fun)
        : html_file(html_file), support_dir(support_dir), socket(std::move(socket)), fun(fun) {}

    void run() { do_read(); }

    void do_read() {
        req = {};
        http::async_read(
            socket,
            buffer,
            req,
            std::bind_front(&http_session::on_read,shared_from_this()));
    }

    void on_read(beast::error_code ec,std::size_t) {
        if(ec == http::error::end_of_stream) return do_close();

        if(ec) return post_error(socket.get_executor(),ec);

        if(websocket::is_upgrade(req)) {
            auto ptr = new websocket_session(std::move(socket));
            std::unique_ptr<websocket_session> s(ptr);
            ptr->run(std::move(req),fun,std::move(s));
            return;
        }

        handle_request(
            html_file,
            support_dir,
            std::move(req),
            [this]<bool isRequest,typename Body,typename Fields>(http::message<isRequest,Body,Fields>&& msg) {
                auto sp = std::make_shared<
                    http::message<isRequest,Body,Fields>>(std::move(msg));
                res = sp;

                http::async_write(
                    socket,
                    *sp,
                    std::bind(
                        &http_session::on_write,
                        shared_from_this(),
                        std::placeholders::_1,
                        std::placeholders::_2,
                        sp->need_eof()));
            });
    }

    void on_write(beast::error_code ec,std::size_t,bool close) {
        if(ec) return post_error(socket.get_executor(),ec);
        if(close) return do_close();
        res = nullptr;
        do_read();
    }

    void do_close() {
        beast::error_code ec;
        socket.shutdown(tcp::socket::shutdown_send,ec);
    }
};

class listener : public std::enable_shared_from_this<listener> {
    const char *html_file;
    const char *support_dir;
    net::io_context &ioc;
    tcp::acceptor acceptor;
    input_fun fun;

public:
    listener(const char *html_file,const char *support_dir,net::io_context &ioc,tcp::endpoint endpoint,input_fun fun)
        : html_file(html_file), support_dir(support_dir), ioc(ioc), acceptor(ioc), fun(fun)
    {
        acceptor.open(endpoint.protocol());
        acceptor.set_option(net::socket_base::reuse_address(true));
        acceptor.bind(endpoint);
        acceptor.listen(net::socket_base::max_listen_connections);
    }

    void run() { do_accept(); }

private:
    void do_accept() {
        acceptor.async_accept(
            ioc,
            std::bind_front(&listener::on_accept,shared_from_this()));
    }

    void on_accept(beast::error_code ec,tcp::socket socket) {
        if(ec) post_error(ioc.get_executor(),ec);
        else std::make_shared<http_session>(html_file,support_dir,std::move(socket),fun)->run();

        do_accept();
    }
};
}

void run_message_server(const char *html_file,const char *support_dir,input_fun fun) {
    net::io_context ioc{1};

    tcp::endpoint ep{net::ip::address_v4::loopback(),8080};
    std::make_shared<listener>(html_file,support_dir,ioc,ep,fun)->run();

    net::signal_set signals(ioc,SIGINT,SIGTERM);
    signals.async_wait([&](const beast::error_code&,int) { ioc.stop(); });

#if BOOST_OS_WINDOWS
    std::system("explorer \"http://localhost:8080\"");
#elif BOOST_OS_LINUX
    std::system("xdg-open \"http://localhost:8080\"");
#elif BOOST_OS_MACOS
    std::system("open \"http://localhost:8080\"");
#else
    std::cout << "Don't know how to launch the browser on this OS. "
        "Open \"http://localhost:8080\" manually." << std::endl;
#endif

    for(;;) {
        try {
            ioc.run();
            break;
        } catch(const std::exception& e) {
            report_error(e.what());
        }
    }
}

namespace {
char hex_c(unsigned int x) {
    assert(x < 16);
    return "0123456789abcdef"[x];
}

void output_escaped(std::streambuf *buf,char c) {
    auto uc = static_cast<unsigned char>(c);
    switch(c) {
    case '"':
    case '\\':
        buf->sputc('\\');
        buf->sputc(c);
        break;
    case '\b': buf->sputn("\\b",2); break;
    case '\f': buf->sputn("\\f",2); break;
    case '\n': buf->sputn("\\n",2); break;
    case '\r': buf->sputn("\\r",2); break;
    case '\t': buf->sputn("\\t",2); break;
    default:
        if(uc < 32 || uc == 127) {
            buf->sputn("\\u00",4);
            buf->sputc(hex_c(uc >> 4));
            buf->sputc(hex_c(uc & 0xf));
        } else buf->sputc(c);
        break;
    }
}
void output_escaped(std::streambuf *buf,std::string_view str) {
    for(char c : str) output_escaped(buf,c);
}
struct json_escape_str { std::string_view value; };
std::ostream &operator<<(std::ostream &os,const json_escape_str &str) {
    output_escaped(os.rdbuf(),str.value);
    return os;
}
}

namespace json {
std::streambuf *escaped_stream_buf::raw() {
    assert(_raw);
    return _raw;
}

std::streamsize escaped_stream_buf::xsputn(const char_type *s,std::streamsize count) {
    output_escaped(raw(),std::string_view(s,count));
    return count;
}
escaped_stream_buf::int_type escaped_stream_buf::overflow(int_type ch) {
    std::streambuf *r = raw();
    if(!traits_type::eq_int_type(ch,traits_type::eof())) output_escaped(r,traits_type::to_char_type(ch));
    return traits_type::not_eof(ch);
}
}

message_canvas::~message_canvas() {
    esb._raw = nullptr;
}

void message_canvas::emit_raw(char x) { data.obuffer << x; }
void message_canvas::emit_value(bool x) { data.obuffer << (x ? "true" : "false"); }
void message_canvas::emit_value(decltype(nullptr)) { data.obuffer << "null"; }
void message_canvas::emit_value(long x) { data.obuffer << x; }
void message_canvas::emit_value(unsigned long x) { data.obuffer << x; }
void message_canvas::emit_value(float x) { data.obuffer << x; }
void message_canvas::emit_value(double x) { data.obuffer << x; }
void message_canvas::emit_value(std::string_view x) {
    data.obuffer << '"' << json_escape_str{x} << '"';
}
void message_canvas::send() {
    data.deliver_obuffer();
}

void message_canvas::check_ready() {
    if(esb._raw) throw std::logic_error(
        "cannot send other messages until object returned by "
        "\"console_line_stream\" is destroyed");
}

void message_canvas::console_line(std::string_view msg) {
    check_ready();
    data.obuffer << "{\"command\":\"console\",\"text\":\"" << json_escape_str{msg} << "\"}";
    send();
}

message_canvas::console_line_stream_obj message_canvas::console_line_stream() {
    check_ready();
    data.obuffer << "{\"command\":\"console\",\"text\":\"";
    esb._raw = data.obuffer.rdbuf();
    return console_line_stream_obj(*this);
}

message_canvas::console_line_stream_obj::console_line_stream_obj(message_canvas &owner) : owner(owner) {
    init(&owner.esb);
}

message_canvas::console_line_stream_obj::~console_line_stream_obj() {
    owner.esb._raw = nullptr;
    owner.data.obuffer << "\"}";
    owner.send();
}

std::u8string_view message_canvas::get_text() {
    data.ibuffer.consume(data.ibuffer.size());

    data.read(data.ibuffer);

    if(!data.ws.got_text()) throw std::runtime_error(
        "received binary instead of text data from WebSocket");
    return {static_cast<const char8_t*>(data.ibuffer.data().data()),data.ibuffer.size()};
}

cblob message_canvas::get_binary() {
    data.ibuffer.consume(data.ibuffer.size());

    data.read(data.ibuffer);

    if(data.ws.got_text()) throw std::runtime_error(
        "received text instead of binary data from WebSocket");
    return {data.ibuffer.data().data(),data.ibuffer.size()};
}
