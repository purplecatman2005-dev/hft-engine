// eval_ml.cpp — 収録フィードに対するGBDTモデルの予測vs実測の照合(evolve用)
//
// 使い方:
//   eval_ml <capture.bin> [options]
//     --horizon-ms N     実測評価ホライズン(既定 5000)
//     --cooldown N       発火後の最小間隔イベント数(既定 64)
//     --stride N         整合性サンプルの間引き間隔(既定 16)
//     --fee-bps X        片道テイカー手数料bps(既定 15 = 0.15%)
//     --tag S            サマリに記録するモデルバージョン名
//     --out PATH         発火シグナル明細CSV(既定 eval_signals.csv)
//     --features-out PATH 再学習用の実データ特徴量CSV(既定 live_features.csv)
//     --summary PATH     サマリJSON(既定 eval_summary.json)
//
// replay.cpp と同じ収録形式を読み、ml_features.hpp の FeatureEngine と
// バージョン指定モデル(-DML_MODEL_HEADER)で予測。実時間ホライズン先の
// midと突き合わせて「整合性(corr/方向的中)」「発火シグナルのペーパー収支」
// 「再学習用ラベル付き特徴量」を出力する。
// -DHFT_PX_WINDOW=1048576 でビルドすること(収録時のlive.exeと同じ窓)。
#include "common.hpp"
#include "orderbook.hpp"
#include "feed.hpp"
#include "ml_features.hpp"

#ifndef ML_MODEL_HEADER
#define ML_MODEL_HEADER "ml_model.hpp"
#endif
#include ML_MODEL_HEADER

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

using namespace hft;

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

// JSON文字列用: バックスラッシュをスラッシュに寄せる(パス用途のみ)
static std::string json_path(const char* s) {
    std::string r(s);
    for (char& c : r) if (c == '\\') c = '/';
    return r;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: eval_ml <capture.bin> [--horizon-ms 5000] [--cooldown 64]\n"
                    "       [--stride 16] [--fee-bps 15] [--tag NAME]\n"
                    "       [--out eval_signals.csv] [--features-out live_features.csv]\n"
                    "       [--summary eval_summary.json]\n");
        return 1;
    }
    const char* path = argv[1];
    u64 horizon_ms = 5000;
    std::size_t cooldown = 64, stride = 16;
    double fee_bps = 15.0;
    const char* tag = "unknown";
    const char* out_path = "eval_signals.csv";
    const char* feat_path = "live_features.csv";
    const char* sum_path = "eval_summary.json";
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        const char* v = argv[i + 1];
        if      (k == "--horizon-ms")   horizon_ms = std::stoull(v);
        else if (k == "--cooldown")     cooldown = std::stoull(v);
        else if (k == "--stride")       stride = std::stoull(v);
        else if (k == "--fee-bps")      fee_bps = std::stod(v);
        else if (k == "--tag")          tag = v;
        else if (k == "--out")          out_path = v;
        else if (k == "--features-out") feat_path = v;
        else if (k == "--summary")      sum_path = v;
        else { std::printf("unknown option: %s\n", k.c_str()); return 1; }
    }

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

    std::printf("=== eval_ml %s (%.1f MB, model=%s, th=%.5g) ===\n",
                path, fsz / 1e6, tag, mlmodel::THRESHOLD);

    OrderBook book;
    FeatureEngine fe;
    BestState prev{};
    i64 base_jpy = 0;

    struct Ev { u64 ts; double mid; bool valid; };
    std::vector<Ev> evs;                 // 板イベント毎のmid(実JPY, 前方評価用)
    struct Sample { std::size_t ev_idx; u64 ts; double pred, mid; double f[ML_NFEAT]; };
    std::vector<Sample> samples;         // stride間引きの整合性サンプル
    struct Signal { std::size_t ev_idx; u64 ts; double pred; int dir; double mid, spread; };
    std::vector<Signal> signals;

    std::size_t n_msgs = 0, n_trades = 0, n_resets = 0, n_events = 0, last_fire = 0;
    u64 ts_first = 0, ts_last = 0;
    double spread_sum = 0, mid_sum = 0;
    std::size_t spread_cnt = 0;
    double fx[ML_NFEAT];

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
            fe = FeatureEngine();        // 再アンカー時は特徴量もウォームアップし直す
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

        const BestState cur = snap_best(book);
        const double mid = cur.valid ? base_jpy + 0.5 * (cur.pb + cur.pa) - PX_LO : 0;
        evs.push_back({ts, mid, cur.valid});
        const bool ok = fe.update(prev, cur, fx);
        prev = cur;
        ++n_events;
        if (cur.valid) {
            spread_sum += cur.pa - cur.pb;
            mid_sum += mid;
            ++spread_cnt;
        }
        if (!ok || !cur.valid) continue;

        const double pred = mlmodel::predict(fx);
        if (n_events % stride == 0) {
            Sample s;
            s.ev_idx = evs.size() - 1;
            s.ts = ts; s.pred = pred; s.mid = mid;
            for (int i = 0; i < ML_NFEAT; ++i) s.f[i] = fx[i];
            samples.push_back(s);
        }
        if (std::abs(pred) > mlmodel::THRESHOLD && n_events - last_fire >= cooldown) {
            signals.push_back({evs.size() - 1, ts, pred, pred > 0 ? +1 : -1, mid,
                               double(cur.pa - cur.pb)});
            last_fire = n_events;
        }
    }

    const u64 hz_ns = horizon_ms * 1'000'000ull;
    // 発火時刻+ホライズン以降で最初の有効midを前方値とする(replay.cppと同基準)
    auto forward_mid = [&](std::size_t ev_idx, u64 ts, double* out) -> bool {
        std::size_t j = ev_idx;
        while (j < evs.size() && (evs[j].ts < ts + hz_ns || !evs[j].valid)) ++j;
        if (j >= evs.size()) return false;
        *out = evs[j].mid;
        return true;
    };

    // -- 整合性サンプル: corr(pred, dmid) と方向的中率 + 再学習用CSV --
    std::FILE* ff = std::fopen(feat_path, "w");
    if (!ff) { std::printf("cannot open %s\n", feat_path); return 1; }
    std::fprintf(ff, "event_idx");
    for (const char* name : ML_FEAT_NAMES) std::fprintf(ff, ",%s", name);
    std::fprintf(ff, ",dmid\n");
    std::size_t n_res = 0, dir_ok = 0, dir_n = 0;
    double sp = 0, sy = 0, spp = 0, syy = 0, spy = 0;
    for (const Sample& s : samples) {
        double fwd;
        if (!forward_mid(s.ev_idx, s.ts, &fwd)) break;   // 収録末尾: 評価不能
        const double dmid = fwd - s.mid;
        std::fprintf(ff, "%zu", s.ev_idx);
        for (int i = 0; i < ML_NFEAT; ++i) std::fprintf(ff, ",%.10g", s.f[i]);
        std::fprintf(ff, ",%.4f\n", dmid);
        ++n_res;
        sp += s.pred; sy += dmid;
        spp += s.pred * s.pred; syy += dmid * dmid; spy += s.pred * dmid;
        if (dmid != 0) {
            ++dir_n;
            if ((s.pred > 0) == (dmid > 0)) ++dir_ok;
        }
    }
    std::fclose(ff);
    double corr = 0;
    if (n_res > 1) {
        const double n = double(n_res);
        const double cov = spy / n - (sp / n) * (sy / n);
        const double vp = spp / n - (sp / n) * (sp / n);
        const double vy = syy / n - (sy / n) * (sy / n);
        if (vp > 1e-15 && vy > 1e-15) corr = cov / std::sqrt(vp * vy);
    }
    const double dir_acc = dir_n ? double(dir_ok) / dir_n : 0;

    // -- 発火シグナル: 的中率とペーパー収支 --
    std::FILE* csv = std::fopen(out_path, "w");
    if (!csv) { std::printf("cannot open %s\n", out_path); return 1; }
    std::fprintf(csv, "time,pred,dir,mid_jpy,fwd_mid_jpy,dmid_jpy,capture_jpy,"
                      "spread_jpy,net_bps\n");
    std::size_t scored = 0, hits = 0, zeros = 0;
    double sum_cap = 0, sum_net_bps = 0;
    for (const Signal& s : signals) {
        double fwd;
        if (!forward_mid(s.ev_idx, s.ts, &fwd)) break;
        const double dmid = fwd - s.mid;
        const double cap = dmid * s.dir;
        // テイカー往復: 方向×mid変動 − スプレッド − 手数料2回分(bps換算)
        const double net_bps = (cap - s.spread) / s.mid * 1e4 - 2.0 * fee_bps;
        char tbuf[16];
        fmt_time(s.ts, tbuf, sizeof tbuf);
        std::fprintf(csv, "%s,%.6g,%d,%.1f,%.1f,%.1f,%.1f,%.0f,%.3f\n",
                     tbuf, s.pred, s.dir, s.mid, fwd, dmid, cap, s.spread, net_bps);
        ++scored;
        if (dmid == 0) ++zeros;
        else if (cap > 0) ++hits;
        sum_cap += cap;
        sum_net_bps += net_bps;
    }
    std::fclose(csv);

    const double dur_s = (ts_last - ts_first) / 1e9;
    const double avg_spread = spread_cnt ? spread_sum / spread_cnt : 0;
    const double avg_mid = spread_cnt ? mid_sum / spread_cnt : 0;
    const std::size_t decisive = scored - zeros;
    const double hit_rate = decisive ? double(hits) / decisive : 0;

    std::printf("\n-- consistency (all sampled events, stride=%zu) --\n", stride);
    std::printf("samples    : %zu resolved (of %zu)\n", n_res, samples.size());
    std::printf("corr       : %+.4f (pred vs %llums先dmid)\n", corr,
                static_cast<unsigned long long>(horizon_ms));
    std::printf("dir acc    : %.1f%% (mid不変除く %zu 件)\n", 100 * dir_acc, dir_n);
    std::printf("\n-- fired signals (|pred| > %.5g, cooldown=%zu) --\n",
                mlmodel::THRESHOLD, cooldown);
    std::printf("signals    : %zu fired, %zu scored\n", signals.size(), scored);
    if (scored) {
        std::printf("hit rate   : %.1f%% (%zu/%zu, mid不変 %zu 件除く)\n",
                    100 * hit_rate, hits, decisive, zeros);
        std::printf("avg capture: %+.1f JPY | avg net %+.2f bps (spread+fee %.0fbps控除)\n",
                    sum_cap / scored, sum_net_bps / scored, 2 * fee_bps);
    }
    std::printf("duration   : %.1f s | avg spread %.1f JPY | avg mid %.0f JPY\n",
                dur_s, avg_spread, avg_mid);

    // -- サマリJSON --
    std::FILE* js = std::fopen(sum_path, "w");
    if (!js) { std::printf("cannot open %s\n", sum_path); return 1; }
    std::fprintf(js,
        "{\n"
        "  \"tag\": \"%s\",\n"
        "  \"capture\": \"%s\",\n"
        "  \"threshold\": %.10g,\n"
        "  \"horizon_ms\": %llu,\n"
        "  \"fee_bps\": %.4g,\n"
        "  \"cooldown\": %zu,\n"
        "  \"duration_s\": %.1f,\n"
        "  \"n_msgs\": %zu,\n"
        "  \"n_events\": %zu,\n"
        "  \"n_trades\": %zu,\n"
        "  \"n_resets\": %zu,\n"
        "  \"avg_spread_jpy\": %.2f,\n"
        "  \"avg_mid_jpy\": %.1f,\n"
        "  \"n_samples\": %zu,\n"
        "  \"corr_pred_dmid\": %.6f,\n"
        "  \"dir_acc\": %.6f,\n"
        "  \"dir_n\": %zu,\n"
        "  \"n_signals\": %zu,\n"
        "  \"n_scored\": %zu,\n"
        "  \"n_zeros\": %zu,\n"
        "  \"hit_rate\": %.6f,\n"
        "  \"avg_capture_jpy\": %.4f,\n"
        "  \"sum_capture_jpy\": %.2f,\n"
        "  \"avg_net_bps\": %.4f,\n"
        "  \"sum_net_bps\": %.4f\n"
        "}\n",
        tag, json_path(path).c_str(), mlmodel::THRESHOLD,
        static_cast<unsigned long long>(horizon_ms), fee_bps, cooldown,
        dur_s, n_msgs, n_events, n_trades, n_resets, avg_spread, avg_mid,
        n_res, corr, dir_acc, dir_n,
        signals.size(), scored, zeros, hit_rate,
        scored ? sum_cap / scored : 0.0, sum_cap,
        scored ? sum_net_bps / scored : 0.0, sum_net_bps);
    std::fclose(js);
    std::printf("out        : %s / %s / %s\n", out_path, feat_path, sum_path);
    return 0;
}
