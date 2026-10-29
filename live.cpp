// live.cpp — 実データ受信エンジン(Phase 2)
//
// アダプタ(adapters/bitflyer_adapter.py 等)から localhost TCP でバイナリフィードを
// 受信し、板更新→OFI z-scoreのリアルタイム計算と、全フレームの収録を行う。
//
// 使い方:
//   live [port=9401]        ※ -DHFT_PX_WINDOW=1048576 でビルドすること(build.bat参照)
//
// ストリーム形式(wire.py と対): [u64 ts_ns] + ワイヤメッセージ
//   A/X/E/D は feed.hpp と同一。拡張:
//   'R' u8,i64 base_jpy : 板リセット+アンカー(実価格 = base + tick - PX_LO)
//   'T' u8,u8,i32,u32   : 約定(板には適用しない。収録・表示のみ)
//
// 収録: captures/capture_YYYYMMDD_HHMMSS.bin に受信フレームをそのまま書く。
// Ctrl+C で終了(収録は1秒毎にflush済み)。
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <direct.h>
#include "common.hpp"
#include "orderbook.hpp"
#include "feed.hpp"
#include "signals.hpp"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

using namespace hft;

static std::atomic<bool> g_stop{false};
static BOOL WINAPI on_ctrl(DWORD) { g_stop.store(true); return TRUE; }

// メッセージ種別ごとのワイヤ長(タイムスタンプ8Bを除く)。0 = 不明タイプ
static std::size_t msg_size(u8 t) {
    switch (t) {
    case 'A': return 14;
    case 'X': case 'E': case 'R': return 9;
    case 'D': return 5;
    case 'T': return 10;
    default:  return 0;
    }
}

static void fmt_time(u64 ts_ns, char* buf, std::size_t n) {
    const std::time_t sec = static_cast<std::time_t>(ts_ns / 1'000'000'000ull);
    std::tm tm_{};
    localtime_s(&tm_, &sec);
    std::strftime(buf, n, "%H:%M:%S", &tm_);
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // リダイレクト時もログを即時に出す
    int port = 9401;
    if (argc > 1) port = std::stoi(argv[1]);

    constexpr std::size_t W = 64;       // OFI集計窓(イベント数)
    constexpr double Z_TH = 2.5;        // 発火閾値
    constexpr std::size_t COOLDOWN = W; // 発火後の最小間隔(イベント数)

    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) { std::printf("WSAStartup failed\n"); return 1; }
    SetConsoleCtrlHandler(on_ctrl, TRUE);

    SOCKET lst = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<u_short>(port));
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    BOOL yes = TRUE;
    setsockopt(lst, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<char*>(&yes), sizeof yes);
    if (bind(lst, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0 || listen(lst, 1) != 0) {
        std::printf("bind/listen failed on port %d (WSA %d)\n", port, WSAGetLastError());
        return 1;
    }

    // 収録ファイル
    _mkdir("captures");
    char fname[128];
    {
        const std::time_t now = std::time(nullptr);
        std::tm tm_{};
        localtime_s(&tm_, &now);
        std::strftime(fname, sizeof fname, "captures/capture_%Y%m%d_%H%M%S.bin", &tm_);
    }
    std::FILE* cap = std::fopen(fname, "wb");
    if (!cap) { std::printf("cannot open %s\n", fname); return 1; }

    std::printf("=== live engine (PX_WINDOW=%d) ===\n", PX_WINDOW);
    std::printf("listening 127.0.0.1:%d | capture: %s\n", port, fname);
    std::printf("start adapter: python adapters/bitflyer_adapter.py --port %d\n\n", port);

    OrderBook book;
    RollingZ roll(W);
    BestState prev{};
    i64 base_jpy = 0;
    bool anchored = false;
    u64 n_msgs = 0, n_trades = 0, n_resets = 0, n_signals = 0, msgs_this_sec = 0;
    std::size_t last_fire = 0, n_events = 0;
    double last_z = 0;
    u64 last_ts = 0;

    std::vector<u8> acc;
    acc.reserve(1 << 20);
    std::vector<char> rbuf(1 << 16);
    SOCKET cli = INVALID_SOCKET;
    auto last_flush = std::chrono::steady_clock::now();

    while (!g_stop.load()) {
        // 接続待ち/受信を select でタイムアウト付き監視(Ctrl+C応答用)
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(cli == INVALID_SOCKET ? lst : cli, &fds);
        timeval tv{0, 200'000};
        const int r = select(0, &fds, nullptr, nullptr, &tv);
        if (r <= 0) goto tick;

        if (cli == INVALID_SOCKET) {
            cli = accept(lst, nullptr, nullptr);
            std::printf("[tcp] adapter connected\n");
            acc.clear();
            goto tick;
        }
        {
            const int got = recv(cli, rbuf.data(), static_cast<int>(rbuf.size()), 0);
            if (got <= 0) {
                std::printf("[tcp] adapter disconnected; waiting for reconnect\n");
                closesocket(cli);
                cli = INVALID_SOCKET;
                goto tick;
            }
            acc.insert(acc.end(), rbuf.data(), rbuf.data() + got);
        }

        // フレーム消費
        {
            std::size_t off = 0;
            while (acc.size() - off >= 9) {
                const u8 type = acc[off + 8];
                const std::size_t msz = msg_size(type);
                if (msz == 0) {
                    std::printf("[fatal] unknown message type 0x%02x — protocol mismatch\n", type);
                    g_stop.store(true);
                    break;
                }
                if (acc.size() - off < 8 + msz) break;
                const u8* p = acc.data() + off;
                std::memcpy(&last_ts, p, 8);
                std::fwrite(p, 1, 8 + msz, cap);          // 収録(受信フレームそのまま)
                ++n_msgs;
                ++msgs_this_sec;

                const u8* mp = p + 8;
                if (type == 'R') {
                    std::memcpy(&base_jpy, mp + 1, 8);
                    book = OrderBook();
                    roll = RollingZ(W);
                    prev = BestState{};
                    anchored = true;
                    ++n_resets;
                    char tbuf[16];
                    fmt_time(last_ts, tbuf, sizeof tbuf);
                    std::printf("[%s] RESET/anchor base=%lld JPY (resets=%llu)\n", tbuf,
                                static_cast<long long>(base_jpy),
                                static_cast<unsigned long long>(n_resets));
                } else if (type == 'T') {
                    ++n_trades;
                } else {
                    NormMsg m;
                    decode(mp, m);
                    if (m.type == 'A' && (m.px < PX_LO || m.px >= PX_HI)) {
                        std::printf("[fatal] px %d out of window [%d,%d) — "
                                    "adapter/build window mismatch\n", m.px, PX_LO, PX_HI);
                        g_stop.store(true);
                        break;
                    }
                    switch (m.type) {
                    case 'A': book.add(m.id, static_cast<Side>(m.side), m.px, m.qty); break;
                    case 'X': case 'E': book.reduce(m.id, m.qty); break;
                    case 'D': book.remove(m.id); break;
                    }
                    const BestState cur = snap_best(book);
                    last_z = roll.push(ofi_contribution(prev, cur));
                    prev = cur;
                    ++n_events;
                    if (roll.warmed() && cur.valid && (last_z > Z_TH || last_z < -Z_TH) &&
                        n_events - last_fire >= COOLDOWN) {
                        last_fire = n_events;
                        ++n_signals;
                        char tbuf[16];
                        fmt_time(last_ts, tbuf, sizeof tbuf);
                        const double mid = base_jpy + 0.5 * (cur.pb + cur.pa) - PX_LO;
                        std::printf("[%s] SIGNAL %s z=%+.2f mid=%.1f JPY spread=%d\n", tbuf,
                                    last_z > 0 ? "BUY " : "SELL", last_z, mid, cur.pa - cur.pb);
                    }
                }
                off += 8 + msz;
            }
            if (off > 0) acc.erase(acc.begin(), acc.begin() + off);
        }

    tick:
        const auto now = std::chrono::steady_clock::now();
        if (now - last_flush >= std::chrono::seconds(1)) {
            std::fflush(cap);
            last_flush = now;
            if (anchored && book.has_side(BID) && book.has_side(ASK)) {
                const i64 bb = base_jpy + book.best_px(BID) - PX_LO;
                const i64 ba = base_jpy + book.best_px(ASK) - PX_LO;
                std::printf("[stat] %llu msg/s | bid %lld x %lld | ask %lld x %lld | "
                            "spread %lld | z %+.2f | trades %llu | signals %llu\n",
                            static_cast<unsigned long long>(msgs_this_sec),
                            static_cast<long long>(bb), static_cast<long long>(book.best_qty(BID)),
                            static_cast<long long>(ba), static_cast<long long>(book.best_qty(ASK)),
                            static_cast<long long>(ba - bb), last_z,
                            static_cast<unsigned long long>(n_trades),
                            static_cast<unsigned long long>(n_signals));
            }
            msgs_this_sec = 0;
        }
    }

    std::fclose(cap);
    if (cli != INVALID_SOCKET) closesocket(cli);
    closesocket(lst);
    WSACleanup();
    std::printf("\n=== done: %llu msgs, %llu trades, %llu signals -> %s ===\n",
                static_cast<unsigned long long>(n_msgs),
                static_cast<unsigned long long>(n_trades),
                static_cast<unsigned long long>(n_signals), fname);
    return 0;
}
