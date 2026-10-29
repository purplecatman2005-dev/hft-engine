// replay.cpp — 収録フィード(live.exeのcaptures/*.bin)のリプレイと検証・採点
//
// 使い方:
//   replay <capture.bin> [horizon_ms=5000] [z_th=2.5] [window=64]
//
// 実行内容:
//  1. 収録フレームを順に適用し、std::map参照実装と定期的に板を突き合わせ
//  2. OFI z-scoreを再計算し、|z|>閾値でシグナル発火(クールダウン=窓長イベント)
//  3. 実時間ホライズン(horizon_ms)先のmid変動で採点 → replay_signals.csv
//
// backtest.cpp との違い: 合成フィードではなく実収録、採点がイベント数でなく実時間。
// -DHFT_PX_WINDOW=1048576 でビルドすること(収録時のlive.exeと同じ窓サイズ)。
#include "common.hpp"
#include "orderbook.hpp"
#include "feed.hpp"
#include "signals.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

using namespace hft;

// std::mapベースの参照板(bench.cppのRefBook2と同方式+リセット対応)
struct RefBook {
    struct RefOrder { u8 side; i32 px; u32 qty; };
    std::map<i32, i64> levels[2];
    std::unordered_map<u32, RefOrder> orders;

    void clear() { levels[0].clear(); levels[1].clear(); orders.clear(); }

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

static bool verify(const OrderBook& fast, const RefBook& ref, std::size_t at) {
    bool ok = true;
    for (u8 s : {u8(BID), u8(ASK)}) {
        i32 rpx; i64 rq;
        const bool rhas = ref.best(s, rpx, rq);
        const bool fhas = fast.has_side(static_cast<Side>(s));
        if (rhas != fhas) { ok = false; break; }
        if (rhas && (fast.best_px(static_cast<Side>(s)) != rpx ||
                     fast.best_qty(static_cast<Side>(s)) != rq)) { ok = false; break; }
        // best側から10レベル分の数量を突き合わせ(bench.cppと同基準)
        if (rhas) {
            const i32 idx = fast.best_idx(static_cast<Side>(s));
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

static std::size_t msg_size(u8 t) {
    switch (t) {
    case 'A': return 14;
    case 'X': case 'E': case 'R': return 9;
    case 'D': return 5;
    case 'T': return 10;
    default:  return 0;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: replay <capture.bin> [horizon_ms=5000] [z_th=2.5] [window=64]\n");
        return 1;
    }
    const char* path = argv[1];
    u64 horizon_ms = 5000;
    double z_th = 2.5;
    std::size_t W = 64;
    if (argc > 2) horizon_ms = std::stoull(argv[2]);
    if (argc > 3) z_th = std::stod(argv[3]);
    if (argc > 4) W = std::stoull(argv[4]);

    std::FILE* f = std::fopen(path, "rb");
    if (!f) { std::printf("cannot open %s\n", path); return 1; }
    std::fseek(f, 0, SEEK_END);
    const long fsz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<u8> buf(static_cast<std::size_t>(fsz));
    if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
        std::printf("read error\n");
        return 1;
    }
    std::fclose(f);

    std::printf("=== replay %s (%.1f MB, PX_WINDOW=%d) ===\n", path, fsz / 1e6, PX_WINDOW);
    std::printf("params: horizon=%llums z_th=%.2f window=%zu\n",
                static_cast<unsigned long long>(horizon_ms), z_th, W);

    OrderBook book;
    RefBook ref;
    RollingZ roll(W);
    BestState prev{};
    i64 base_jpy = 0;

    struct Ev { u64 ts; double mid; bool valid; };
    std::vector<Ev> evs;                 // 板イベント毎のmid(実JPY, 前方評価用)
    struct Signal { std::size_t ev_idx; u64 ts; double z; int dir; double mid; };
    std::vector<Signal> signals;

    std::size_t n_msgs = 0, n_trades = 0, n_resets = 0, n_events = 0, last_fire = 0;
    u64 ts_first = 0, ts_last = 0;
    double spread_sum = 0;
    std::size_t spread_cnt = 0;
    bool all_ok = true;

    for (std::size_t off = 0; off + 9 <= buf.size();) {
        u64 ts;
        std::memcpy(&ts, buf.data() + off, 8);
        const u8 type = buf[off + 8];
        const std::size_t msz = msg_size(type);
        if (msz == 0 || off + 8 + msz > buf.size()) {
            std::printf("[warn] truncated/unknown frame at byte %zu — stop\n", off);
            break;
        }
        const u8* mp = buf.data() + off + 8;
        off += 8 + msz;
        ++n_msgs;
        if (ts_first == 0) ts_first = ts;
        ts_last = ts;

        if (type == 'R') {
            std::memcpy(&base_jpy, mp + 1, 8);
            book = OrderBook();
            ref.clear();
            roll = RollingZ(W);
            prev = BestState{};
            ++n_resets;
            continue;
        }
        if (type == 'T') { ++n_trades; continue; }

        NormMsg m;
        decode(mp, m);
        if (m.type == 'A' && (m.px < PX_LO || m.px >= PX_HI)) {
            std::printf("[fatal] px %d out of window [%d,%d) — 収録時と窓サイズ不一致\n",
                        m.px, PX_LO, PX_HI);
            return 1;
        }
        switch (m.type) {
        case 'A': book.add(m.id, static_cast<Side>(m.side), m.px, m.qty); break;
        case 'X': case 'E': book.reduce(m.id, m.qty); break;
        case 'D': book.remove(m.id); break;
        }
        ref.apply(m);
        if (n_msgs % 10000 == 0) all_ok &= verify(book, ref, n_msgs);

        const BestState cur = snap_best(book);
        const double mid = cur.valid ? base_jpy + 0.5 * (cur.pb + cur.pa) - PX_LO : 0;
        evs.push_back({ts, mid, cur.valid});
        const double z = roll.push(ofi_contribution(prev, cur));
        prev = cur;
        ++n_events;
        if (cur.valid) { spread_sum += cur.pa - cur.pb; ++spread_cnt; }

        if (roll.warmed() && cur.valid && (z > z_th || z < -z_th) &&
            n_events - last_fire >= W) {
            signals.push_back({evs.size() - 1, ts, z, z > 0 ? +1 : -1, mid});
            last_fire = n_events;
        }
    }
    all_ok &= verify(book, ref, n_msgs);

    std::printf("\n-- integrity check vs std::map reference --\n%s\n", all_ok ? "PASS" : "FAIL");

    // 実時間ホライズンで採点
    std::FILE* csv = std::fopen("replay_signals.csv", "w");
    if (!csv) { std::printf("cannot open replay_signals.csv\n"); return 1; }
    std::fprintf(csv, "time,z,dir,mid_jpy,fwd_mid_jpy,dmid_jpy,captured_jpy\n");
    const u64 hz_ns = horizon_ms * 1'000'000ull;
    std::size_t scored = 0, hits = 0, zeros = 0;
    double captured = 0;
    for (const Signal& s : signals) {
        // 発火時刻+ホライズン以降で最初の有効midを前方値とする
        std::size_t j = s.ev_idx;
        while (j < evs.size() && (evs[j].ts < s.ts + hz_ns || !evs[j].valid)) ++j;
        if (j >= evs.size()) break;      // 収録末尾: 採点不能
        const double dmid = evs[j].mid - s.mid;
        const double cap = dmid * s.dir;
        char tbuf[16];
        const std::time_t sec = static_cast<std::time_t>(s.ts / 1'000'000'000ull);
        std::tm tm_{};
        localtime_s(&tm_, &sec);
        std::strftime(tbuf, sizeof tbuf, "%H:%M:%S", &tm_);
        std::fprintf(csv, "%s,%.3f,%d,%.1f,%.1f,%.1f,%.1f\n",
                     tbuf, s.z, s.dir, s.mid, evs[j].mid, dmid, cap);
        ++scored;
        if (dmid == 0) ++zeros;
        else if (cap > 0) ++hits;
        captured += cap;
    }
    std::fclose(csv);

    const double dur_s = (ts_last - ts_first) / 1e9;
    const double avg_spread = spread_cnt ? spread_sum / spread_cnt : 0;
    std::printf("\n-- replay stats --\n");
    std::printf("duration   : %.1f s | %zu msgs (%.1f msg/s) | %zu trades | %zu resets\n",
                dur_s, n_msgs, dur_s > 0 ? n_msgs / dur_s : 0, n_trades, n_resets);
    std::printf("avg spread : %.1f JPY\n", avg_spread);
    std::printf("signals    : %zu fired, %zu scored\n", signals.size(), scored);
    if (scored) {
        const std::size_t decisive = scored - zeros;
        std::printf("hit rate   : %.1f%% (%zu/%zu, mid不変 %zu 件除く)\n",
                    decisive ? 100.0 * hits / decisive : 0.0, hits, decisive, zeros);
        std::printf("avg capture: %+.1f JPY/signal (%llums先, 方向×mid変動)\n",
                    captured / scored, static_cast<unsigned long long>(horizon_ms));
        std::printf("判定       : capture %s spread/2 → エッジ%s(手数料除く)\n",
                    captured / scored > avg_spread / 2 ? ">" : "<=",
                    captured / scored > avg_spread / 2 ? "あり" : "なし");
    }
    std::printf("csv        : replay_signals.csv\n");
    return all_ok ? 0 : 1;
}
