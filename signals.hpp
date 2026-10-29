#pragma once
// signals.hpp — マイクロストラクチャ・シグナル
//
//  - OFI (Order Flow Imbalance, Cont-Kukanov-Stoikov 2014):
//    best両サイドの価格・数量変化から買い圧-売り圧の純流入を1イベント毎に算出
//  - microprice: 数量加重mid。次のmid変動の不偏推定として板読みの基本
//  - ローリングz-score: 固定長リングバッファ + 累積和で O(1)/イベント更新
#include "common.hpp"
#include "orderbook.hpp"
#include <cmath>
#include <vector>

namespace hft {

// 1イベント分のOFI寄与 e_n を、best状態のスナップショット差分から計算
struct BestState {
    i32 pb = 0, pa = 0;   // best bid/ask (絶対ティック)
    i64 qb = 0, qa = 0;   // best数量
    bool valid = false;
};

inline BestState snap_best(const OrderBook& b) {
    BestState s;
    if (b.has_side(BID) && b.has_side(ASK)) {
        s.pb = b.best_px(BID); s.qb = b.best_qty(BID);
        s.pa = b.best_px(ASK); s.qa = b.best_qty(ASK);
        s.valid = true;
    }
    return s;
}

inline double ofi_contribution(const BestState& prev, const BestState& cur) {
    if (!prev.valid || !cur.valid) return 0.0;
    double e = 0.0;
    // 買い圧: bidが上がる/積まれる → +
    if (cur.pb >= prev.pb) e += double(cur.qb);
    if (cur.pb <= prev.pb) e -= double(prev.qb);
    // 売り圧: askが下がる/積まれる → -
    if (cur.pa <= prev.pa) e -= double(cur.qa);
    if (cur.pa >= prev.pa) e += double(prev.qa);
    return e;
}

inline double microprice(const BestState& s) {
    const double d = double(s.qb + s.qa);
    if (d <= 0) return 0.5 * (s.pb + s.pa);
    return (s.pb * double(s.qa) + s.pa * double(s.qb)) / d;
}

// 固定窓ローリング平均/分散(リングバッファ+累積和; O(1)更新)
class RollingZ {
public:
    explicit RollingZ(std::size_t window) : buf_(window, 0.0), w_(window) {}

    // 新しい値を入れ、窓合計のz-scoreを返す(窓が埋まるまでは0)
    double push(double x) {
        sum_ -= buf_[i_]; sumsq_ -= buf_[i_] * buf_[i_];
        buf_[i_] = x;
        sum_ += x; sumsq_ += x * x;
        if (++i_ == w_) i_ = 0;
        if (n_ < w_) ++n_;
        if (n_ < w_) return 0.0;   // 窓が埋まる(w_個目)までは0
        const double mean = sum_ / double(w_);
        const double var = sumsq_ / double(w_) - mean * mean;
        // 窓「合計」のzではなく、直近値の分布に対する窓合計の偏り:
        // sum ~ N(w*mean_all, w*var) を仮定し、全期間平均0(OFIは中心対称)として正規化
        if (var <= 1e-12) return 0.0;
        return sum_ / std::sqrt(double(w_) * var);
    }

    bool warmed() const { return n_ >= w_; }

private:
    std::vector<double> buf_;
    std::size_t w_, i_ = 0, n_ = 0;
    double sum_ = 0.0, sumsq_ = 0.0;
};

} // namespace hft
