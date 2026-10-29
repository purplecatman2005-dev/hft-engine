#pragma once
// orderbook.hpp — フラット配列ベースのL2オーダーブック
//
// HFT実務の定石:
//  - 価格を整数ティックで扱い、価格→配列インデックスの直接変換にする
//    (std::map の赤黒木を辿るポインタチェイスを排除。1レベルアクセス=1キャッシュライン)
//  - 注文IDが密な連番であることを利用し、注文格納も配列直引き(ハッシュ表すら使わない)
//  - best bid/ask はイベント毎に再計算せず増分維持。touch付近の空レベル走査のみ
#include "common.hpp"
#include <vector>
#include <cassert>

namespace hft {

enum Side : u8 { BID = 0, ASK = 1 };

// 価格窓: [PX_LO, PX_LO + PX_WINDOW) ティックのみ保持
// 実運用では日中の値幅制限(ストップ高安)が窓サイズの根拠になる
// live/replay(実データ)ビルドは -DHFT_PX_WINDOW=1048576 で拡張する
#ifndef HFT_PX_WINDOW
#define HFT_PX_WINDOW 4096
#endif
inline constexpr i32 PX_WINDOW = HFT_PX_WINDOW;
inline constexpr i32 PX_LO     = 100000;          // 窓の下端(絶対ティック)
inline constexpr i32 PX_HI     = PX_LO + PX_WINDOW;

struct Order {
    u32 px_idx;   // 価格窓内インデックス
    u32 qty;      // 残量(0 = 消滅)
    u8  side;
    u8  _pad[3];
};
static_assert(sizeof(Order) == 12);

class OrderBook {
public:
    OrderBook() : qty_(2, std::vector<i64>(PX_WINDOW, 0)) {
        orders_.reserve(1 << 20);
    }

    // ---- フィードイベント適用 ----
    void add(u32 id, Side s, i32 px, u32 qty) {
        const u32 i = static_cast<u32>(px - PX_LO);
        assert(i < static_cast<u32>(PX_WINDOW));
        if (id >= orders_.size()) orders_.resize(id + 1, Order{0, 0, 0, {}});
        orders_[id] = Order{i, qty, static_cast<u8>(s), {}};
        qty_[s][i] += qty;
        if (s == BID) { if (static_cast<i32>(i) > best_[BID]) best_[BID] = static_cast<i32>(i); }
        else          { if (static_cast<i32>(i) < best_[ASK]) best_[ASK] = static_cast<i32>(i); }
    }

    // 部分/全量取消・約定は「数量削減」として同一パスで処理
    void reduce(u32 id, u32 dec) {
        Order& o = orders_[id];
        if (HFT_UNLIKELY(o.qty == 0)) return;      // 二重取消は無視(実フィードでも防御する)
        if (dec > o.qty) dec = o.qty;
        o.qty -= dec;
        const Side s = static_cast<Side>(o.side);
        qty_[s][o.px_idx] -= dec;
        if (qty_[s][o.px_idx] == 0 && static_cast<i32>(o.px_idx) == best_[s])
            repair_best(s);
    }

    void remove(u32 id) { reduce(id, orders_[id].qty); }

    // ---- L2アクセス ----
    // 戻り値は窓内インデックス。空サイドは BID:-1 / ASK:PX_WINDOW
    i32 best_idx(Side s) const { return best_[s]; }
    bool has_side(Side s) const { return s == BID ? best_[BID] >= 0 : best_[ASK] < PX_WINDOW; }
    i32 best_px(Side s) const { return best_[s] + PX_LO; }
    i64 qty_at_idx(Side s, i32 idx) const { return qty_[s][idx]; }
    i64 best_qty(Side s) const { return qty_[s][best_[s]]; }
    double mid() const { return 0.5 * (best_px(BID) + best_px(ASK)); }
    i32 spread() const { return best_px(ASK) - best_px(BID); }
    const Order& order(u32 id) const { return orders_[id]; }
    std::size_t order_capacity() const { return orders_.size(); }

private:
    // best消滅時のみ、悪い方向へ空レベルを走査する。
    // 板の性質上ほぼtouch直下で見つかるため分岐予測が効く
    void repair_best(Side s) {
        i32 i = best_[s];
        if (s == BID) { while (i >= 0 && qty_[BID][i] == 0) --i; best_[BID] = i; }
        else          { while (i < PX_WINDOW && qty_[ASK][i] == 0) ++i; best_[ASK] = i; }
    }

    std::vector<std::vector<i64>> qty_;          // [side][px_idx] -> 合計数量
    std::vector<Order> orders_;                  // 注文ID直引き
    i32 best_[2] = {-1, PX_WINDOW};              // 空サイドの番兵
};

} // namespace hft
