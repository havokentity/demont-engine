// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Rajesh D'Monte
#pragma once

#include <atomic>
#include <cstdint>
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
    void LineClientLoop(int fd);

    // ----- State ----------------------------------------------------------
    Config                config_;
    Console*              console_ = nullptr;
    mg_context*           ctx_     = nullptr;

    std::mutex            ws_mutex_;
    std::unordered_map<mg_connection*, WsClient> ws_clients_;

    std::atomic<bool>     line_run_{false};
    int                   line_fd_  = -1;
    std::thread           line_thread_;
    std::vector<std::thread> line_workers_;
};

}  // namespace pt::console
