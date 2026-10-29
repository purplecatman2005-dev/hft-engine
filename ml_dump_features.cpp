// ml_dump_features.cpp — GBDT学習用の特徴量+前方ラベルをCSV出力
//
// 使い方: ml_dump_features.exe [メッセージ数=2000000] [seed=42] [出力=ml_features.csv]
//
// 合成フィードをリプレイし、STRIDEイベント毎に特徴量7種と
// H=500イベント先のmid変動(ティック)を1行として書き出す。
#include <cstdio>
#include <string>
#include <vector>

#include "common.hpp"
#include "feed.hpp"
#include "ml_features.hpp"
#include "orderbook.hpp"

using namespace hft;

int main(int argc, char** argv) {
    std::size_t n = 2'000'000;
    u64 seed = 42;
    const char* out_path = "ml_features.csv";
    if (argc > 1) n = std::stoull(argv[1]);
    if (argc > 2) seed = std::stoull(argv[2]);
    if (argc > 3) out_path = argv[3];

    constexpr std::size_t H = 500;        // ラベルの評価ホライズン(イベント数)
    constexpr std::size_t STRIDE = 16;    // サンプリング間隔(自己相関の低減)
    constexpr std::size_t WARMUP = 20'000;

    std::printf("=== dump features (n=%zu, seed=%llu) -> %s ===\n", n,
                static_cast<unsigned long long>(seed), out_path);

    FeedGenerator gen(seed);
    std::vector<u8> buf = gen.take_initial();
    gen.generate(n, buf);

    OrderBook book;
    FeatureEngine fe;
    BestState prev{};

    struct Row { std::size_t idx; double f[ML_NFEAT]; };
    std::vector<Row> rows;
    std::vector<double> mids;
    mids.reserve(n + 1024);

    std::size_t k = 0;
    double f[ML_NFEAT];
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
        const bool ok = fe.update(prev, cur, f);
        prev = cur;
        if (ok && k >= WARMUP && k % STRIDE == 0) {
            Row r;
            r.idx = k;
            for (int i = 0; i < ML_NFEAT; ++i) r.f[i] = f[i];
            rows.push_back(r);
        }
    }

    std::FILE* fp = std::fopen(out_path, "w");
    if (!fp) { std::fprintf(stderr, "cannot open %s\n", out_path); return 1; }
    std::fprintf(fp, "event_idx");
    for (const char* name : ML_FEAT_NAMES) std::fprintf(fp, ",%s", name);
    std::fprintf(fp, ",dmid\n");
    std::size_t written = 0;
    for (const Row& r : rows) {
        if (r.idx + H >= mids.size()) break;
        std::fprintf(fp, "%zu", r.idx);
        for (int i = 0; i < ML_NFEAT; ++i) std::fprintf(fp, ",%.10g", r.f[i]);
        std::fprintf(fp, ",%.4f\n", mids[r.idx + H] - mids[r.idx]);
        ++written;
    }
    std::fclose(fp);
    std::printf("rows: %zu\n", written);
    return 0;
}
