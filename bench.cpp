// bench.cpp — フィードリプレイのベンチマークと板の整合性検証
//
// 使い方:
//   bench [メッセージ数=1000000] [--threaded]
//
// 実行内容:
//  1. 合成フィード生成(計測外)
//  2. 計測パス: decode+板更新をメッセージ毎にrdtscで計測 → 分位点レポート
//  3. 検証パス: std::mapベースの参照実装と10万メッセージ毎に板を突き合わせ
//  4. --threaded: フィードスレッド→SPSCキュー→戦略スレッドの2スレッド構成を計測
#include "common.hpp"
#include "orderbook.hpp"
#include "feed.hpp"
#include "spsc_queue.hpp"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace hft;

// ---- 参照実装: std::map ベースの素朴な板(検証専用) ----
struct RefBook2 {
    struct RefOrder { u8 side; i32 px; u32 qty; };
    std::map<i32, i64> levels[2];  // [BID], [ASK] 昇順格納(bestは端を見る)
    std::unordered_map<u32, RefOrder> orders;

    void apply(const NormMsg& m) {
        if (m.type == 'A') {
            orders[m.id] = {m.side, m.px, m.qty};
            levels[m.side][m.px] += m.qty;
            return;
        }
        auto it = orders.find(m.id);
        if (it == orders.end()) return;
        const u32 dec = (m.type == 'D') ? it->second.qty : std::min(m.qty, it->second.qty);
        auto& mp = levels[it->second.side];
        auto lv = mp.find(it->second.px);
        lv->second -= dec;
        if (lv->second == 0) mp.erase(lv);
        it->second.qty -= dec;
        if (it->second.qty == 0) orders.erase(it);
    }
    bool best(u8 s, i32& px, i64& q) const {
        if (levels[s].empty()) return false;
        auto it = (s == BID) ? std::prev(levels[s].end()) : levels[s].begin();
        px = it->first; q = it->second;
        return true;
    }
};

static bool verify(const OrderBook& fast, const RefBook2& ref, std::size_t at) {
    bool ok = true;
    for (u8 s : {u8(BID), u8(ASK)}) {
        i32 rpx; i64 rq;
        const bool rhas = ref.best(s, rpx, rq);
        const bool fhas = fast.has_side(static_cast<Side>(s));
        if (rhas != fhas) { ok = false; break; }
        if (rhas && (fast.best_px(static_cast<Side>(s)) != rpx ||
                     fast.best_qty(static_cast<Side>(s)) != rq)) { ok = false; break; }
        // best側から10レベル分の数量を突き合わせ
        if (rhas) {
            i32 idx = fast.best_idx(static_cast<Side>(s));
            for (int k = 0; k < 10; ++k) {
                const i32 i = (s == BID) ? idx - k : idx + k;
                if (i < 0 || i >= PX_WINDOW) break;
                const i64 fq = fast.qty_at_idx(static_cast<Side>(s), i);
                auto it = ref.levels[s].find(i + PX_LO);
                const i64 q2 = (it == ref.levels[s].end()) ? 0 : it->second;
                if (fq != q2) { ok = false; break; }
            }
        }
    }
    if (!ok) std::printf("  [FAIL] message %zu で参照実装と不一致\n", at);
    return ok;
}

int main(int argc, char** argv) {
    std::size_t n = 1'000'000;
    bool threaded = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--threaded") threaded = true;
        else n = std::stoull(argv[i]);
    }

    std::printf("=== HFT order book engine bench ===\n");
    const double ghz = calibrate_tsc_ghz();
    std::printf("TSC: %.3f GHz (calibrated)\n", ghz);

    // 1. フィード生成(計測外)
    FeedGenerator gen(42);
    std::vector<u8> buf = gen.take_initial();
    gen.generate(n, buf);
    // 総メッセージ数をカウント
    std::size_t total = 0;
    for (std::size_t off = 0; off < buf.size();) {
        NormMsg m;
        off += decode(buf.data() + off, m);
        ++total;
    }
    std::printf("feed: %zu messages, %.1f MB\n", total, buf.size() / 1e6);

    // 2. 計測パス(decode + 板更新)
    {
        OrderBook book;
        std::vector<u32> lat(total);
        std::size_t k = 0;
        const auto w0 = std::chrono::steady_clock::now();
        for (std::size_t off = 0; off < buf.size();) {
            const u64 c0 = rdtsc();
            NormMsg m;
            off += decode(buf.data() + off, m);
            switch (m.type) {
            case 'A': book.add(m.id, static_cast<Side>(m.side), m.px, m.qty); break;
            case 'X': case 'E': book.reduce(m.id, m.qty); break;
            case 'D': book.remove(m.id); break;
            }
            lat[k++] = static_cast<u32>(rdtsc() - c0);
        }
        const auto w1 = std::chrono::steady_clock::now();
        const double sec = std::chrono::duration<double>(w1 - w0).count();

        std::sort(lat.begin(), lat.end());
        auto ns_at = [&](double q) { return lat[std::size_t(q * (total - 1))] / ghz; };
        std::printf("\n-- single thread replay --\n");
        std::printf("throughput : %.2f M msgs/sec (wall %.3f s)\n", total / sec / 1e6, sec);
        std::printf("latency/msg: p50 %.0f ns | p90 %.0f ns | p99 %.0f ns | p99.9 %.0f ns | max %.0f ns\n",
                    ns_at(0.50), ns_at(0.90), ns_at(0.99), ns_at(0.999), lat.back() / ghz);
        if (book.has_side(BID) && book.has_side(ASK))
            std::printf("final book : bid %d x %lld | ask %d x %lld | spread %d ticks\n",
                        book.best_px(BID), static_cast<long long>(book.best_qty(BID)),
                        book.best_px(ASK), static_cast<long long>(book.best_qty(ASK)), book.spread());
    }

    // 3. 検証パス(参照実装との突き合わせ)
    {
        OrderBook book;
        RefBook2 ref;
        bool all_ok = true;
        std::size_t k = 0;
        for (std::size_t off = 0; off < buf.size();) {
            NormMsg m;
            off += decode(buf.data() + off, m);
            switch (m.type) {
            case 'A': book.add(m.id, static_cast<Side>(m.side), m.px, m.qty); break;
            case 'X': case 'E': book.reduce(m.id, m.qty); break;
            case 'D': book.remove(m.id); break;
            }
            ref.apply(m);
            if (++k % 100000 == 0) all_ok &= verify(book, ref, k);
        }
        all_ok &= verify(book, ref, k);
        std::printf("\n-- integrity check vs std::map reference --\n");
        std::printf("%s (%zu checkpoints)\n", all_ok ? "PASS" : "FAIL", k / 100000 + 1);
        if (!all_ok) return 1;
    }

    // 4. 2スレッド構成(フィードハンドラ → SPSCキュー → 戦略スレッド)
    if (threaded) {
        OrderBook book;
        SpscQueue<NormMsg> q(1 << 14);
        std::atomic<bool> done{false};
        const auto w0 = std::chrono::steady_clock::now();
        std::thread producer([&] {
            for (std::size_t off = 0; off < buf.size();) {
                NormMsg m;
                off += decode(buf.data() + off, m);
                while (!q.try_push(m)) { /* busy spin: HFTではsleepしない */ }
            }
            done.store(true, std::memory_order_release);
        });
        std::size_t got = 0;
        NormMsg m;
        while (true) {
            if (q.try_pop(m)) {
                switch (m.type) {
                case 'A': book.add(m.id, static_cast<Side>(m.side), m.px, m.qty); break;
                case 'X': case 'E': book.reduce(m.id, m.qty); break;
                case 'D': book.remove(m.id); break;
                }
                ++got;
            } else if (done.load(std::memory_order_acquire)) {
                while (q.try_pop(m)) ++got; // 残りを排出
                break;
            }
        }
        producer.join();
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
        std::printf("\n-- 2-thread (feed -> SPSC queue -> strategy) --\n");
        std::printf("processed  : %zu msgs, %.2f M msgs/sec end-to-end\n", got, got / sec / 1e6);
    }
    return 0;
}
