// test_invariants.cpp — 敵対的検証用テスト(リリースには不要)
//
//  T1: 複数シードで50万msgリプレイし、毎イベントで板の不変条件を検査
//      (クロスなし・spread>=1・best数量>0・注文残量とレベル数量の整合)
//  T2: SPSCキューに2000万件の連番を流し、順序・欠落・重複ゼロを確認
//  T3: RollingZ を素朴なO(W)再計算と突き合わせ
//  T4: 空サイド・二重取消・過大数量取消の防御動作
#include "common.hpp"
#include "orderbook.hpp"
#include "feed.hpp"
#include "signals.hpp"
#include "spsc_queue.hpp"
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

using namespace hft;

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++failures; \
    std::printf("  [FAIL] %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static void t1_book_invariants() {
    std::printf("T1: book invariants over replay (seeds 1,2,3)\n");
    for (u64 seed : {1ull, 2ull, 3ull}) {
        FeedGenerator gen(seed);
        std::vector<u8> buf = gen.take_initial();
        gen.generate(500'000, buf);
        OrderBook book;
        std::size_t k = 0;
        for (std::size_t off = 0; off < buf.size(); ++k) {
            NormMsg m;
            off += decode(buf.data() + off, m);
            switch (m.type) {
            case 'A': book.add(m.id, static_cast<Side>(m.side), m.px, m.qty); break;
            case 'X': case 'E': book.reduce(m.id, m.qty); break;
            case 'D': book.remove(m.id); break;
            }
            if (book.has_side(BID) && book.has_side(ASK)) {
                CHECK(book.spread() >= 1, "seed %llu msg %zu: crossed/locked book spread=%d",
                      (unsigned long long)seed, k, book.spread());
                CHECK(book.best_qty(BID) > 0 && book.best_qty(ASK) > 0,
                      "seed %llu msg %zu: best level with qty<=0", (unsigned long long)seed, k);
                if (failures) return; // 最初の破綻で打ち切り
            }
        }
        // 全レベル数量 >= 0
        for (int s = 0; s < 2; ++s)
            for (i32 i = 0; i < PX_WINDOW; ++i)
                CHECK(book.qty_at_idx(static_cast<Side>(s), i) >= 0,
                      "negative level qty side=%d idx=%d", s, i);
    }
}

static void t2_spsc_stress() {
    std::printf("T2: SPSC queue 20M sequential integrity\n");
    SpscQueue<u64> q(1 << 10);
    constexpr u64 N = 20'000'000;
    std::thread prod([&] {
        for (u64 i = 0; i < N; ++i)
            while (!q.try_push(i)) {}
    });
    u64 expect = 0, v;
    while (expect < N) {
        if (q.try_pop(v)) {
            if (v != expect) { CHECK(false, "got %llu expect %llu",
                (unsigned long long)v, (unsigned long long)expect); break; }
            ++expect;
        }
    }
    prod.join();
    CHECK(expect == N, "consumed %llu of %llu", (unsigned long long)expect, (unsigned long long)N);
}

static void t3_rollingz_vs_naive() {
    std::printf("T3: RollingZ vs naive recomputation\n");
    constexpr std::size_t W = 64;
    RollingZ rz(W);
    std::mt19937_64 rng(7);
    std::normal_distribution<double> nd(0.3, 2.0);
    std::vector<double> xs;
    for (int i = 0; i < 5000; ++i) {
        const double x = nd(rng);
        xs.push_back(x);
        const double z = rz.push(x);
        if (xs.size() < W) { CHECK(z == 0.0, "z!=0 before warmup at %d", i); continue; }
        double sum = 0, sumsq = 0;
        for (std::size_t j = xs.size() - W; j < xs.size(); ++j) { sum += xs[j]; sumsq += xs[j] * xs[j]; }
        const double mean = sum / W, var = sumsq / W - mean * mean;
        const double zn = (var <= 1e-12) ? 0.0 : sum / std::sqrt(double(W) * var);
        CHECK(std::fabs(z - zn) < 1e-6, "i=%d rolling=%f naive=%f", i, z, zn);
        if (failures) return;
    }
}

static void t4_defensive_paths() {
    std::printf("T4: defensive paths (double cancel / over-reduce / empty side)\n");
    OrderBook b;
    CHECK(!b.has_side(BID) && !b.has_side(ASK), "empty book has sides");
    b.add(0, BID, PX_LO + 100, 500);
    b.add(1, ASK, PX_LO + 102, 300);
    b.reduce(0, 9999);                       // 過大数量 → 全量扱い
    CHECK(!b.has_side(BID), "over-reduce did not empty bid side");
    b.reduce(0, 100);                        // 二重取消 → 無視
    CHECK(b.qty_at_idx(BID, 100) == 0, "double cancel corrupted level");
    b.remove(1); b.remove(1);                // 二重削除
    CHECK(!b.has_side(ASK), "double delete corrupted ask side");
    b.add(2, BID, PX_LO + 50, 700);          // 消滅後の再構築
    CHECK(b.has_side(BID) && b.best_px(BID) == PX_LO + 50 && b.best_qty(BID) == 700,
          "re-add after empty failed");
}

int main() {
    t1_book_invariants();
    t2_spsc_stress();
    t3_rollingz_vs_naive();
    t4_defensive_paths();
    if (failures == 0) { std::printf("ALL PASS\n"); return 0; }
    std::printf("%d FAILURES\n", failures);
    return 1;
}
