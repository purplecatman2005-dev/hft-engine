// backtest.cpp — OFI(オーダーフロー不均衡)シグナルのイベント駆動バックテスト
//
// 使い方:
//   backtest [メッセージ数=1000000] [seed=42]
//
// ロジック:
//   各イベントでbest差分からOFI寄与を計算し、直近W=64イベントの窓合計をz-score化。
//   |z| > 2.5 でシグナル発火(クールダウン H イベント)、H=500イベント先のmid変動で採点。
//   結果は backtest_signals.csv に出力。
#include "common.hpp"
#include "orderbook.hpp"
#include "feed.hpp"
#include "signals.hpp"
#include <cstdio>
#include <string>
#include <vector>

using namespace hft;

int main(int argc, char** argv) {
    std::size_t n = 1'000'000;
    u64 seed = 42;
    if (argc > 1) n = std::stoull(argv[1]);
    if (argc > 2) seed = std::stoull(argv[2]);

    constexpr std::size_t W = 64;        // OFI集計窓(イベント数)
    constexpr double Z_TH = 2.5;         // 発火閾値
    constexpr std::size_t H = 500;       // 評価ホライズン(イベント数)
    constexpr std::size_t WARMUP = 20'000;

    std::printf("=== OFI signal backtest (n=%zu, seed=%llu) ===\n", n,
                static_cast<unsigned long long>(seed));

    FeedGenerator gen(seed);
    std::vector<u8> buf = gen.take_initial();
    gen.generate(n, buf);

    OrderBook book;
    RollingZ roll(W);
    BestState prev{};

    struct Signal { std::size_t idx; double z; int dir; double mid; };
    std::vector<Signal> signals;
    std::vector<double> mids;            // イベント毎のmid(前方評価用)
    mids.reserve(n + 1024);
    std::size_t last_fire = 0;
    double spread_sum = 0;
    std::size_t spread_cnt = 0;

    std::size_t k = 0;
    for (std::size_t off = 0; off < buf.size(); ++k) {
        NormMsg m;
        off += decode(buf.data() + off, m);
        switch (m.type) {
        case 'A': book.add(m.id, static_cast<Side>(m.side), m.px, m.qty); break;
        case 'X': case 'E': book.reduce(m.id, m.qty); break;
        case 'D': book.remove(m.id); break;
        }
        const BestState cur = snap_best(book);
        mids.push_back(cur.valid ? 0.5 * (cur.pb + cur.pa) : (mids.empty() ? 0 : mids.back()));
        const double z = roll.push(ofi_contribution(prev, cur));
        prev = cur;
        if (cur.valid) { spread_sum += cur.pa - cur.pb; ++spread_cnt; }

        if (k >= WARMUP && roll.warmed() && cur.valid &&
            (z > Z_TH || z < -Z_TH) && k - last_fire >= H) {
            signals.push_back({k, z, z > 0 ? +1 : -1, mids.back()});
            last_fire = k;
        }
    }

    // 前方mid変動で採点
    std::FILE* f = std::fopen("hft-engine/backtest_signals.csv", "w");
    if (!f) f = std::fopen("backtest_signals.csv", "w");
    std::fprintf(f, "event_idx,z,dir,mid,fwd_mid,dmid_ticks,captured_ticks\n");
    std::size_t scored = 0, hits = 0, zeros = 0;
    double captured = 0;
    for (const Signal& s : signals) {
        if (s.idx + H >= mids.size()) break;
        const double fwd = mids[s.idx + H];
        const double dmid = fwd - s.mid;
        const double cap = dmid * s.dir;
        std::fprintf(f, "%zu,%.3f,%d,%.1f,%.1f,%.2f,%.2f\n",
                     s.idx, s.z, s.dir, s.mid, fwd, dmid, cap);
        ++scored;
        if (dmid == 0) ++zeros;
        else if (cap > 0) ++hits;
        captured += cap;
    }
    std::fclose(f);

    const double avg_spread = spread_cnt ? spread_sum / spread_cnt : 0;
    std::printf("signals    : %zu fired, %zu scored\n", signals.size(), scored);
    if (scored) {
        const std::size_t decisive = scored - zeros;
        std::printf("hit rate   : %.1f%% (%zu/%zu, mid不変 %zu 件除く)\n",
                    decisive ? 100.0 * hits / decisive : 0.0, hits, decisive, zeros);
        std::printf("avg capture: %+.3f ticks/signal (方向×mid変動の平均)\n", captured / scored);
        std::printf("avg spread : %.3f ticks (往復コストの目安 = spread + 手数料)\n", avg_spread);
        std::printf("判定       : capture %s spread/2 → エッジ%s\n",
                    captured / scored > avg_spread / 2 ? ">" : "<=",
                    captured / scored > avg_spread / 2 ? "あり(コスト控除後も正)" : "なし(コスト負け)");
    }
    std::printf("csv        : backtest_signals.csv\n");
    return 0;
}
