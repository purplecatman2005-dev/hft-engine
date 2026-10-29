#pragma once
// feed.hpp — ITCH風バイナリワイヤフォーマットと合成フィード生成器
//
// ワイヤフォーマット(リトルエンディアン, パック):
//   'A' Add    : u8 type, u32 id, u8 side, i32 px(絶対ティック), u32 qty   (14B)
//   'X' Cancel : u8 type, u32 id, u32 dec_qty                              ( 9B)
//   'E' Exec   : u8 type, u32 id, u32 exec_qty                             ( 9B)
//   'D' Delete : u8 type, u32 id                                           ( 5B)
// 実フィード(NASDAQ ITCH / JPX FLEX)はここにタイムスタンプ等が付くが、
// 本エンジンではイベント順序のみを使う。
#include "common.hpp"
#include "orderbook.hpp"
#include <vector>
#include <random>
#include <cstring>

namespace hft {

#pragma pack(push, 1)
struct MsgAdd    { u8 type; u32 id; u8 side; i32 px; u32 qty; };
struct MsgCancel { u8 type; u32 id; u32 dec; };
struct MsgExec   { u8 type; u32 id; u32 qty; };
struct MsgDelete { u8 type; u32 id; };
#pragma pack(pop)
static_assert(sizeof(MsgAdd) == 14 && sizeof(MsgCancel) == 9 && sizeof(MsgDelete) == 5);

// パーサが使う正規化メッセージ(SPSCキューにも流す)
struct NormMsg {
    u8  type;   // 'A' 'X' 'E' 'D'
    u8  side;
    u32 id;
    i32 px;
    u32 qty;
};

// バッファ先頭のメッセージを1件デコードし、消費バイト数を返す
inline std::size_t decode(const u8* p, NormMsg& m) {
    switch (*p) {
    case 'A': { MsgAdd a;    std::memcpy(&a, p, sizeof a); m = {a.type, a.side, a.id, a.px, a.qty}; return sizeof a; }
    case 'X': { MsgCancel c; std::memcpy(&c, p, sizeof c); m = {c.type, 0, c.id, 0, c.dec};         return sizeof c; }
    case 'E': { MsgExec e;   std::memcpy(&e, p, sizeof e); m = {e.type, 0, e.id, 0, e.qty};         return sizeof e; }
    case 'D': { MsgDelete d; std::memcpy(&d, p, sizeof d); m = {d.type, 0, d.id, 0, 0};             return sizeof d; }
    default:  m = {0, 0, 0, 0, 0}; return 1; // 不明タイプ(生成器は出さない)
    }
}

// ---- 合成フィード生成器 ----
// 潜在ドリフト状態(マルコフ切替)が「発注サイドの偏り」と「フェアバリューの移動」の
// 両方を駆動する。これによりオーダーフローが将来のmid変動を弱く予測する構造になり、
// OFIシグナルのバックテストが意味を持つ。
class FeedGenerator {
public:
    explicit FeedGenerator(u64 seed) : rng_(seed) {
        fair_ = PX_LO + PX_WINDOW / 2;
        // 初期板: touch両側に数レベルずつ敷く
        for (int lvl = 1; lvl <= 5; ++lvl) {
            for (int k = 0; k < 3; ++k) {
                emit_add(BID, static_cast<i32>(fair_) - lvl, 100 * (1 + lvl));
                emit_add(ASK, static_cast<i32>(fair_) + lvl, 100 * (1 + lvl));
            }
        }
    }

    // n件のメッセージを生成してバッファに追記
    void generate(std::size_t n, std::vector<u8>& out) {
        out_ = &out;
        for (std::size_t i = 0; i < n; ++i) step();
        out_ = nullptr;
    }

    std::size_t messages_emitted() const { return n_msgs_; }

private:
    void step() {
        // ドリフト状態のマルコフ切替(平均500イベントで遷移)
        if (u01() < 1.0 / 500.0) drift_ = static_cast<int>(rng_() % 3) - 1;
        // フェアバリュー: ドリフト + ノイズ、窓端で反射
        fair_ += 0.02 * drift_ + 0.06 * norm();
        const double margin = 64;
        if (fair_ < PX_LO + margin) fair_ = PX_LO + margin;
        if (fair_ > PX_HI - margin) fair_ = PX_HI - margin;

        const double r = u01();
        if (r < 0.58 || live_.empty()) do_add();
        else if (r < 0.85)             do_cancel();
        else                           do_exec();
    }

    void do_add() {
        // ドリフト上向き→買い発注が増える(オーダーフローに情報が乗る)
        const Side s = (u01() < 0.5 + 0.18 * drift_) ? BID : ASK;
        const i32 f = static_cast<i32>(fair_);
        i32 px = (s == BID) ? f - 1 - geometric(0.45, 12)
                            : f + 1 + geometric(0.45, 12);
        // 板をクロスさせない(実フィードではクロス分は約定として現れるため)
        if (s == BID && book_.has_side(ASK) && px >= book_.best_px(ASK)) px = book_.best_px(ASK) - 1;
        if (s == ASK && book_.has_side(BID) && px <= book_.best_px(BID)) px = book_.best_px(BID) + 1;
        if (px < PX_LO + 1) px = PX_LO + 1;
        if (px > PX_HI - 2) px = PX_HI - 2;
        emit_add(s, px, 100u * (1 + rng_() % 10));
    }

    void do_cancel() {
        const u32 id = live_[rng_() % live_.size()];
        const Order& o = book_.order(id);
        if (u01() < 0.3 && o.qty > 100) {                 // 部分取消
            const u32 dec = 100u * (1 + rng_() % (o.qty / 100));
            emit_reduce('X', id, dec >= o.qty ? o.qty : dec);
        } else {
            emit_reduce('D', id, o.qty);                  // 全量削除
        }
    }

    void do_exec() {
        // ドリフト上向き→買い成行がaskを叩く
        const Side hit = (u01() < 0.5 + 0.18 * drift_) ? ASK : BID;
        if (!book_.has_side(hit)) { do_add(); return; }
        auto& q = lvl_[hit][book_.best_idx(hit)];
        if (q.empty()) { do_add(); return; }              // 起こらないはずだが防御
        const u32 id = q.front();                         // 時間優先(FIFO)
        const Order& o = book_.order(id);
        const u32 fill = (u01() < 0.6) ? o.qty : 100u * (1 + rng_() % (o.qty / 100 ? o.qty / 100 : 1));
        emit_reduce('E', id, fill >= o.qty ? o.qty : fill);
    }

    // ---- 内部ミラー板の更新 + ワイヤ書き出し ----
    void emit_add(Side s, i32 px, u32 qty) {
        const u32 id = next_id_++;
        book_.add(id, s, px, qty);
        lvl_[s][px - PX_LO].push_back(id);
        live_pos_.resize(next_id_, UINT32_MAX);
        live_pos_[id] = static_cast<u32>(live_.size());
        live_.push_back(id);
        MsgAdd m{'A', id, static_cast<u8>(s), px, qty};
        write(&m, sizeof m);
    }

    void emit_reduce(char type, u32 id, u32 dec) {
        const Order o = book_.order(id);
        book_.reduce(id, dec);
        if (dec >= o.qty) drop_from_indexes(id, static_cast<Side>(o.side), o.px_idx);
        if (type == 'X')      { MsgCancel m{'X', id, dec}; write(&m, sizeof m); }
        else if (type == 'E') { MsgExec   m{'E', id, dec}; write(&m, sizeof m); }
        else                  { MsgDelete m{'D', id};      write(&m, sizeof m); }
    }

    void drop_from_indexes(u32 id, Side s, u32 px_idx) {
        auto& q = lvl_[s][px_idx];
        for (std::size_t i = 0; i < q.size(); ++i)
            if (q[i] == id) { q.erase(q.begin() + i); break; }
        const u32 pos = live_pos_[id];
        live_pos_[live_.back()] = pos;
        live_[pos] = live_.back();
        live_.pop_back();
        live_pos_[id] = UINT32_MAX;
    }

    void write(const void* p, std::size_t n) {
        if (out_) {
            const u8* b = static_cast<const u8*>(p);
            out_->insert(out_->end(), b, b + n);
            ++n_msgs_;
        } else {
            init_buf_.insert(init_buf_.end(), static_cast<const u8*>(p),
                             static_cast<const u8*>(p) + n);
        }
    }

public:
    // 初期板構築分のメッセージ(generate前に一度だけ取り出してバッファ先頭に置く)
    std::vector<u8> take_initial() { return std::move(init_buf_); }

private:
    double u01() { return std::uniform_real_distribution<double>(0, 1)(rng_); }
    double norm() { return std::normal_distribution<double>(0, 1)(rng_); }
    i32 geometric(double p, i32 cap) {
        i32 k = 0;
        while (k < cap && u01() > p) ++k;
        return k;
    }

    std::mt19937_64 rng_;
    OrderBook book_;                                   // 生成器自身のミラー板
    std::vector<std::vector<u32>> lvl_[2] = {
        std::vector<std::vector<u32>>(PX_WINDOW),
        std::vector<std::vector<u32>>(PX_WINDOW)};     // レベル別FIFO(約定順序用)
    std::vector<u32> live_;                            // 生存注文ID(取消抽選用)
    std::vector<u32> live_pos_;                        // id -> live_内位置
    u32 next_id_ = 0;
    double fair_;
    int drift_ = 0;
    std::vector<u8>* out_ = nullptr;
    std::vector<u8> init_buf_;
    std::size_t n_msgs_ = 0;
};

} // namespace hft
