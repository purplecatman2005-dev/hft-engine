// backtest_ml.cpp — GBDT(コンパイル済み決定木)シグナルのバックテスト+レイテンシ計測
//
// 使い方:
//   backtest_ml.exe [メッセージ数=1000000] [seed=777]
//   backtest_ml.exe --verify ml_verify.csv    (Python予測値との一致検証)
//
// 学習は seed=42〜46 のフィードで行うため、既定 seed=777 はアウトオブサンプル。
// 同一リプレイ上で従来の z-score ルール(|z64|>2.5)も採点し、直接比較する。
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common.hpp"
#include "feed.hpp"
#include "ml_features.hpp"
#include "ml_model.hpp"
#include "orderbook.hpp"

using namespace hft;

static int verify(const char* path) {
    std::FILE* fp = std::fopen(path, "r");
    if (!fp) { std::fprintf(stderr, "cannot open %s\n", path); return 2; }
    char line[4096];
    if (!std::fgets(line, sizeof line, fp)) { std::fclose(fp); return 2; }  // header
    double f[ML_NFEAT], maxdiff = 0.0;
    std::size_t rows = 0;
    while (std::fgets(line, sizeof line, fp)) {
        char* s = line;
        for (int i = 0; i < ML_NFEAT; ++i) {
            f[i] = std::strtod(s, &s);
            if (*s == ',') ++s;
        }
        const double py = std::strtod(s, &s);
        const double cpp = mlmodel::predict(f);
        maxdiff = std::max(maxdiff, std::abs(py - cpp));
        ++rows;
    }
    std::fclose(fp);
    std::printf("verify: %zu rows, max |python - cpp| = %.3e -> %s\n",
                rows, maxdiff, maxdiff < 1e-9 ? "OK" : "MISMATCH");
    return maxdiff < 1e-9 ? 0 : 1;
}

struct Scorer {
    struct Sig { std::size_t idx; int dir; double mid; };
    std::vector<Sig> sigs;
    std::size_t last_fire = 0;
    std::size_t cooldown;

    explicit Scorer(std::size_t cd) : cooldown(cd) {}

    void maybe_fire(std::size_t k, int dir, double mid) {
        if (k - last_fire < cooldown) return;
        sigs.push_back({k, dir, mid});
        last_fire = k;
    }

    void report(const char* name, const std::vector<double>& mids,
                std::size_t H, double avg_spread) const {
        std::size_t scored = 0, hits = 0, zeros = 0;
        double captured = 0;
        for (const Sig& s : sigs) {
            if (s.idx + H >= mids.size()) break;
            const double cap = (mids[s.idx + H] - s.mid) * s.dir;
            ++scored;
            if (cap == 0) ++zeros;
            else if (cap > 0) ++hits;
            captured += cap;
        }
        std::printf("--- %s ---\n", name);
        if (!scored) { std::printf("  no signals\n"); return; }
        const std::size_t dec = scored - zeros;
        std::printf("  signals     : %zu\n", scored);
        std::printf("  hit rate    : %.1f%% (%zu/%zu)\n",
                    dec ? 100.0 * hits / dec : 0.0, hits, dec);
        std::printf("  avg capture : %+.3f ticks/signal (spread/2 = %.3f -> エッジ%s)\n",
                    captured / scored, avg_spread / 2,
                    captured / scored > avg_spread / 2 ? "あり" : "なし");
    }
};

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--verify") == 0) {
        return verify(argc > 2 ? argv[2] : "ml_verify.csv");
    }
    std::size_t n = 1'000'000;
    u64 seed = 777;
    if (argc > 1) n = std::stoull(argv[1]);
    if (argc > 2) seed = std::stoull(argv[2]);

    constexpr std::size_t H = 500;
    constexpr std::size_t WARMUP = 20'000;
    constexpr double Z_TH = 2.5;

    std::printf("=== GBDT vs z-score backtest (n=%zu, seed=%llu, out-of-sample) ===\n",
                n, static_cast<unsigned long long>(seed));

    FeedGenerator gen(seed);
    std::vector<u8> buf = gen.take_initial();
    gen.generate(n, buf);

    OrderBook book;
    FeatureEngine fe;
    BestState prev{};
    Scorer ml(H), zr(H);
    std::vector<double> mids;
    mids.reserve(n + 1024);

    u64 pred_cycles = 0;
    std::size_t pred_calls = 0;
    double f[ML_NFEAT];
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
        if (cur.valid) { spread_sum += cur.pa - cur.pb; ++spread_cnt; }
        const bool ok = fe.update(prev, cur, f);
        prev = cur;
        if (!ok || k < WARMUP) continue;

        const u64 t0 = rdtsc();
        const double pred = mlmodel::predict(f);
        pred_cycles += rdtsc() - t0;
        ++pred_calls;

        if (std::abs(pred) > mlmodel::THRESHOLD)
            ml.maybe_fire(k, pred > 0 ? +1 : -1, mids.back());
        if (std::abs(f[0]) > Z_TH)
            zr.maybe_fire(k, f[0] > 0 ? +1 : -1, mids.back());
    }

    const double avg_spread = spread_cnt ? spread_sum / spread_cnt : 0;

    ml.report("GBDT (compiled trees)", mids, H, avg_spread);
    zr.report("z-score rule (|z64| > 2.5)", mids, H, avg_spread);

    const double ghz = calibrate_tsc_ghz();
    const double cyc = pred_calls ? double(pred_cycles) / pred_calls : 0;
    std::printf("--- inference latency ---\n");
    std::printf("  predict calls : %zu\n", pred_calls);
    std::printf("  avg latency   : %.0f cycles = %.1f ns @ %.2f GHz (rdtsc往復込み)\n",
                cyc, cyc / ghz, ghz);
    return 0;
}
