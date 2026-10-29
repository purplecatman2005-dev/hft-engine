#pragma once
// common.hpp — 型定義・計測ユーティリティ
#include <cstdint>
#include <cstddef>
#include <chrono>
#include <x86intrin.h>

// ---- Windowsコンソールを UTF-8 表示にする(日本語出力の文字化け防止) ----
// windows.h はマクロ汚染が大きいので、使う関数1つだけを宣言する
#ifdef _WIN32
extern "C" __declspec(dllimport) int __stdcall SetConsoleOutputCP(unsigned int);
namespace hft::detail {
inline const bool console_utf8 = (SetConsoleOutputCP(65001), true);
}
#endif

namespace hft {

using u8  = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

inline constexpr std::size_t CACHELINE = 64;

#if defined(__GNUC__)
#define HFT_LIKELY(x)   __builtin_expect(!!(x), 1)
#define HFT_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define HFT_LIKELY(x)   (x)
#define HFT_UNLIKELY(x) (x)
#endif

// ---- rdtsc ベースの低オーバーヘッド計測 ----
// steady_clock(QPC)は1呼び出し20-30ns。TSCは数ns+シリアライズなしで読める。
// 現代のx86はinvariant TSC(周波数固定)なので実時間換算はキャリブレーションで行う。
inline u64 rdtsc() { return __rdtsc(); }

// TSC周波数をsteady_clockと突き合わせて実測する(起動時に1回)
inline double calibrate_tsc_ghz(int ms = 100) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    const u64 c0 = rdtsc();
    const auto limit = std::chrono::milliseconds(ms);
    while (clock::now() - t0 < limit) { /* busy wait */ }
    const u64 c1 = rdtsc();
    const auto t1 = clock::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return double(c1 - c0) / ns; // cycles per ns = GHz
}

} // namespace hft
