# wire.py — C++エンジン(feed.hpp)のITCH風バイナリ形式エンコーダ
#
# ストリーム/収録の共通フレーム: [u64 ts_ns (epoch)] + [ワイヤメッセージ]
# ワイヤメッセージ(リトルエンディアン, パック):
#   'A' Add    : u8 type, u32 id, u8 side, i32 px(絶対ティック), u32 qty (14B)
#   'X' Cancel : u8 type, u32 id, u32 dec_qty                            ( 9B)
#   'E' Exec   : u8 type, u32 id, u32 exec_qty                           ( 9B)
#   'D' Delete : u8 type, u32 id                                         ( 5B)
# 実データ用の拡張(live.cpp / replay.cpp のみが解釈):
#   'R' Reset  : u8 type, i64 base_jpy  — 板全消去+アンカー設定          ( 9B)
#                実価格 = base_jpy + (px_tick - PX_LO)
#   'T' Trade  : u8 type, u8 side, i32 px, u32 qty — 約定(板には未適用)  (10B)
#
# 数量単位: 1e-6 BTC(u32で最大4294 BTC/注文)。価格ティック: 1 JPY。
import struct
import time

BID, ASK = 0, 1

PX_LO = 100_000
PX_WINDOW = 1 << 20          # live/replayビルドの -DHFT_PX_WINDOW と一致させること
PX_HI = PX_LO + PX_WINDOW

_A = struct.Struct("<BIBiI")
_X = struct.Struct("<BII")
_E = struct.Struct("<BII")
_D = struct.Struct("<BI")
_R = struct.Struct("<Bq")
_T = struct.Struct("<BBiI")
_TS = struct.Struct("<Q")


def _frame(payload: bytes, ts_ns: int | None) -> bytes:
    return _TS.pack(time.time_ns() if ts_ns is None else ts_ns) + payload


def add(oid: int, side: int, px: int, qty: int, ts_ns=None) -> bytes:
    return _frame(_A.pack(ord("A"), oid, side, px, qty), ts_ns)


def cancel(oid: int, dec: int, ts_ns=None) -> bytes:
    return _frame(_X.pack(ord("X"), oid, dec), ts_ns)


def delete(oid: int, ts_ns=None) -> bytes:
    return _frame(_D.pack(ord("D"), oid), ts_ns)


def reset(base_jpy: int, ts_ns=None) -> bytes:
    return _frame(_R.pack(ord("R"), base_jpy), ts_ns)


def trade(side: int, px: int, qty: int, ts_ns=None) -> bytes:
    return _frame(_T.pack(ord("T"), side, px, qty), ts_ns)
