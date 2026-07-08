// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

struct mg_context;
struct mg_connection;

namespace pt::log { enum class Level; }

namespace pt::console {

// Portable socket handle: POSIX uses signed int (-1 = invalid); Winsock
// uses SOCKET (UINT_PTR). Pulling <winsock2.h> into a public header is
// noisy, so we declare a uintptr_t-shaped opaque type here and let the
// .cpp do the real reinterpretation. On POSIX it's still effectively an
// int; on Windows the upper bits hold the kernel handle.
#if defined(_WIN32)
using socket_handle_t = std::uintptr_t;
inline constexpr socket_handle_t kInvalidSocket = static_cast<socket_handle_t>(~0ULL);
#else
using socket_handle_t = int;
inline constexpr socket_handle_t kInvalidSocket = -1;
#endif

class Console;

// Hosts the WebSocket+HTTP server (civetweb) and the raw line-protocol TCP
// listener.  Network threads own the I/O; commands queue onto Console for
// main-thread execution.
class ConsoleServer {
public:
    struct Config {
        std::string  bind_address  = "127.0.0.1";
        std::uint16_t http_port    = 27960;
        std::uint16_t line_port    = 27961;
    };

    ConsoleServer();
    ~ConsoleServer();
    ConsoleServer(const ConsoleServer&) = delete;
    ConsoleServer& operator=(const ConsoleServer&) = delete;

    bool Start(const Config& cfg, Console* console);
    void Stop();

    // Push an event JSON object body to all connections subscribed to topic.
    // `data_json` is the contents of the "data" field as a JSON object body
    // (e.g. R"({"fps":142.3})").
    void BroadcastEvent(std::string_view topic, std::string_view data_json);

    // --- Editor backend (agent-19) ---------------------------------------
    // Hooks for the editor-mode WebSocket protocol. Each callback runs
    // on the civetweb worker thread that handled the inbound message,
    // so implementations MUST either be lock-free + thread-safe (the
    // selection setter can do this trivially -- it's a single (kind,id)
    // pair) or queue a console command via Console::QueueExecute for
    // main-thread execution. The serializer returns a JSON string body
    // ready to embed in the WebSocket reply.
    //
    // Both hooks default to no-op when unset: `list_scene` replies with
    // an empty object, `select` becomes a parse-only acknowledgement.
    // The engine wires both in Init() right after Start().
    using SelectFn = std::function<void(std::string_view kind, std::uint32_t id)>;
    using SceneDumpFn = std::function<std::string()>;
    void SetSelectHandler(SelectFn fn)        { select_handler_     = std::move(fn); }
    void SetSceneDumpHandler(SceneDumpFn fn)  { scene_dump_handler_ = std::move(fn); }
    // --- end Editor backend ----------------------------------------------

    // Forward a log line to subscribers of the "log" topic.  Wired as a
    // pt::log::Sink at startup.
    static void OnLog(pt::log::Level level, const std::string& body);
    static void SetGlobalInstance(ConsoleServer* s);

private:
    // ----- WebSocket per-connection state ---------------------------------
    struct WsClient {
        mg_connection* conn = nullptr;
        std::set<std::string, std::less<>> topics;
    };

    // ----- civetweb callbacks (static -> dispatch via user_data) ---------
    static int  HttpHandler(mg_connection* conn, void* cbdata);
    static int  WsConnectHandler(const mg_connection* conn, void* cbdata);
    static void WsReadyHandler(mg_connection* conn, void* cbdata);
    static int  WsDataHandler(mg_connection* conn, int op, char* data,
                              std::size_t len, void* cbdata);
    static void WsCloseHandler(const mg_connection* conn, void* cbdata);

    void HandleWsMessage(mg_connection* conn, std::string_view payload);
    void SendWs(mg_connection* conn, std::string_view text);

    // ----- Line protocol thread -------------------------------------------
    void LineAcceptLoop();
    void LineClientLoop(socket_handle_t fd);

    // ----- State ----------------------------------------------------------
    Config                config_;
    Console*              console_ = nullptr;
    mg_context*           ctx_     = nullptr;

    std::mutex            ws_mutex_;
    std::unordered_map<mg_connection*, WsClient> ws_clients_;

    std::atomic<bool>     line_run_{false};
    socket_handle_t       line_fd_  = kInvalidSocket;
    std::thread           line_thread_;
    std::vector<std::thread> line_workers_;
    // Live per-client sockets, guarded by line_clients_mutex_. Each
    // worker inserts its fd on entry and erases it on exit; Stop()
    // walks the set and ::shutdown()s every fd to break workers out
    // of a blocking recv(). Closing the LISTEN socket alone doesn't
    // wake accepted clients, so without this a connected-but-idle
    // client (a forgotten `nc` session) would park its worker in
    // recv() forever and the joins in Stop() would hang engine
    // shutdown.
    std::mutex            line_clients_mutex_;
    std::set<socket_handle_t> line_clients_;

    // --- Editor backend (agent-19) ---------------------------------------
    // Set via SetSelectHandler / SetSceneDumpHandler. Read inside
    // HandleWsMessage on the civetweb worker thread; the engine sets
    // these once at Init time so a mutex is unnecessary (write-once
    // before any WebSocket clients can connect, since the listener
    // only accepts after Start() returns).
    SelectFn              select_handler_;
    SceneDumpFn           scene_dump_handler_;
    // --- end Editor backend ----------------------------------------------
};

}  // namespace pt::console
