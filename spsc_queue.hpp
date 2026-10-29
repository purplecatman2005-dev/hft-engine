#pragma once
// spsc_queue.hpp — ロックフリーSPSC(単一生産者・単一消費者)リングバッファ
//
// HFTのフィードハンドラ→戦略スレッド間の標準構成:
//  - ミューテックス無し。acquire/releaseの片側同期のみ
//  - head/tailを別キャッシュラインに置き、false sharingを防ぐ
//  - 容量2の冪でマスク演算(剰余の除算を回避)
//  - 消費者/生産者はそれぞれ相手のインデックスをローカルにキャッシュし、
//    共有変数へのアトミックloadを空振り時のみに減らす
#include "common.hpp"
#include <atomic>
#include <vector>

namespace hft {

template <typename T>
class SpscQueue {
public:
    explicit SpscQueue(std::size_t capacity_pow2) : buf_(capacity_pow2), mask_(capacity_pow2 - 1) {
        static_assert(std::is_trivially_copyable_v<T>);
        // capacityは2の冪であること
    }

    bool try_push(const T& v) {
        const u64 h = head_.load(std::memory_order_relaxed);
        if (h - tail_cache_ > mask_) {
            tail_cache_ = tail_.load(std::memory_order_acquire);
            if (h - tail_cache_ > mask_) return false;   // 満杯
        }
        buf_[h & mask_] = v;
        head_.store(h + 1, std::memory_order_release);
        return true;
    }

    bool try_pop(T& out) {
        const u64 t = tail_.load(std::memory_order_relaxed);
        if (t == head_cache_) {
            head_cache_ = head_.load(std::memory_order_acquire);
            if (t == head_cache_) return false;          // 空
        }
        out = buf_[t & mask_];
        tail_.store(t + 1, std::memory_order_release);
        return true;
    }

private:
    std::vector<T> buf_;
    const u64 mask_;
    alignas(CACHELINE) std::atomic<u64> head_{0};   // 生産者が書く
    alignas(CACHELINE) u64 tail_cache_ = 0;         // 生産者ローカル
    alignas(CACHELINE) std::atomic<u64> tail_{0};   // 消費者が書く
    alignas(CACHELINE) u64 head_cache_ = 0;         // 消費者ローカル
};

} // namespace hft
