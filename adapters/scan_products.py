# -*- coding: utf-8 -*-
# scan_products.py — bitFlyer全商品スクリーナ: 「予測しやすさ×収益性」で銘柄をランク付け
#
# 公開商品(現物 *_JPY + FX_BTC_JPY)の板と約定を1本のWSで同時観測し、
# エンジンと同じ best-level OFI の先行相関を商品ごとに測り、推定エッジ順に提示する。
# 上位銘柄はそのまま bitflyer_adapter.py --product に渡せる。
#
# 使い方:
#   python adapters\scan_products.py                # 10分観測してランキング表示
#   python adapters\scan_products.py --minutes 3
#   python adapters\scan_products.py --selftest     # ネット不要の指標計算の自己検証
#
# 指標(商品ごと):
#   spread_bps : スプレッド中央値(往復コストの下限)
#   vol_bps    : horizon秒先の|midリターン|中央値(値幅=収益機会)
#   ic         : 過去window秒のOFI合計と horizon秒先midリターンのSpearman相関(予測しやすさ)
#   hit        : |OFI|上位1/4のサンプルに限った方向的中率
#   edge_bps   : (2*hit - 1) * vol_bps - spread_bps - 2*fee_bps
#                (シグナル発火時にテイカーで往復した場合の1トレードあたり推定エッジ)
#
# 注意: 短時間観測のスナップショットで、時間帯・相場状況に強く依存する。
#       手数料は既定0(--fee-bps で指定)。結果は「今この瞬間の目安」であり
#       恒常的なエッジの証明ではない。
import argparse
import asyncio
import csv
import json
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
API = "https://api.bitflyer.com/v1"
WS_URL = "wss://ws.lightstream.bitflyer.com/json-rpc"
SAMPLE_DT = 0.5          # サンプリング周期(秒)。全商品共通の等間隔グリッド
MIN_SAMPLES = 100        # これ未満のサンプル数は「データ不足」として順位外


def get_products(include_fx: bool) -> list[str]:
    with urllib.request.urlopen(API + "/getmarkets", timeout=10) as r:
        markets = json.load(r)
    out = []
    for m in markets:
        code, mt = m["product_code"], m.get("market_type", "")
        if mt == "Spot" and code.endswith("_JPY"):
            out.append(code)
        elif include_fx and mt == "FX":
            out.append(code)
    return out


def rankdata(a: np.ndarray) -> np.ndarray:
    """同順位は平均ランク(scipy非依存のSpearman用)"""
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


class Tracker:
    """1商品分の板ミラーとOFI・midのサンプル列"""

    def __init__(self, code: str):
        self.code = code
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.have_snap = False
        self.prev_best = None          # (bb, qb, ba, qa)
        self.ofi_events: deque = deque()   # (t, e)
        self.samples: list = []        # (t, mid, spread, ofi_window)
        self.n_updates = 0
        self.n_trades = 0

    def reset_books(self):
        self.bids.clear()
        self.asks.clear()
        self.have_snap = False
        self.prev_best = None

    def _apply(self, m: dict):
        for book, key in ((self.bids, "bids"), (self.asks, "asks")):
            for lv in m.get(key, []):
                p, s = float(lv["price"]), float(lv["size"])
                if s > 0:
                    book[p] = s
                else:
                    book.pop(p, None)

    def _best(self):
        if not self.bids or not self.asks:
            return None
        bb, ba = max(self.bids), min(self.asks)
        if bb >= ba:               # クロス板(bitFlyerで稀に発生)は無効扱い
            return None
        return (bb, self.bids[bb], ba, self.asks[ba])

    def on_snapshot(self, m: dict):
        self.bids.clear()
        self.asks.clear()
        self._apply(m)
        self.have_snap = True
        # スナップショット跨ぎではOFIイベントを出さない(基準の取り直し)
        self.prev_best = self._best()

    def on_diff(self, m: dict, now: float):
        if not self.have_snap:
            return
        self._apply(m)
        self.n_updates += 1
        cur = self._best()
        prev = self.prev_best
        if cur and prev:
            bb, qb, ba, qa = prev
            b, Qb, a, Qa = cur
            e = 0.0
            if b > bb:
                e += Qb
            elif b == bb:
                e += Qb - qb
            else:
                e -= qb
            if a < ba:
                e -= Qa
            elif a == ba:
                e -= Qa - qa
            else:
                e += qa
            if e != 0.0:
                self.ofi_events.append((now, e))
        if cur:
            self.prev_best = cur

    def sample(self, now: float, window: float):
        if not self.have_snap:
            return
        cur = self._best()
        if cur is None:
            return
        bb, _, ba, _ = cur
        while self.ofi_events and self.ofi_events[0][0] < now - window:
            self.ofi_events.popleft()
        ofi_w = sum(e for _, e in self.ofi_events)
        self.samples.append((now, (bb + ba) / 2.0, ba - bb, ofi_w))

    def metrics(self, horizon: float, fee_bps: float):
        k = max(1, int(round(horizon / SAMPLE_DT)))
        if len(self.samples) < MIN_SAMPLES + k:
            return None
        arr = np.array(self.samples)
        t, mid, spread, ofi = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        fwd = mid[k:] / mid[:-k] - 1.0
        # 切断・欠測でグリッドが飛んだ区間はホライズンとして無効
        valid = (t[k:] - t[:-k]) < horizon * 1.6
        ofi0 = ofi[:-k]
        if valid.sum() < MIN_SAMPLES:
            return None
        spread_bps = float(np.median(spread / mid) * 1e4)
        vol_bps = float(np.median(np.abs(fwd[valid])) * 1e4)
        ic = spearman(ofi0[valid], fwd[valid])
        thr = np.percentile(np.abs(ofi0[valid]), 75)
        strong = valid & (np.abs(ofi0) >= thr) & (ofi0 != 0) & (fwd != 0)
        hit = float(np.mean(np.sign(ofi0[strong]) == np.sign(fwd[strong]))) \
            if strong.sum() >= 20 else float("nan")
        edge_bps = (2 * hit - 1) * vol_bps - spread_bps - 2 * fee_bps \
            if hit == hit else float("nan")
        dur_min = max((t[-1] - t[0]) / 60.0, 1e-9)
        return {
            "product": self.code,
            "samples": int(valid.sum()),
            "trades_per_min": self.n_trades / dur_min,
            "updates_per_min": self.n_updates / dur_min,
            "spread_bps": spread_bps,
            "vol_bps": vol_bps,
            "ic": ic,
            "hit": hit,
            "edge_bps": edge_bps,
        }


# ---- WS観測 ----

async def ws_loop(trackers: dict[str, Tracker], end: float):
    import websockets
    backoff = 1
    while time.time() < end:
        try:
            async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
                i = 0
                for code in trackers:
                    for ch in (f"lightning_board_snapshot_{code}",
                               f"lightning_board_{code}",
                               f"lightning_executions_{code}"):
                        i += 1
                        await ws.send(json.dumps({"jsonrpc": "2.0", "method": "subscribe",
                                                  "params": {"channel": ch}, "id": i}))
                print(f"[ws] connected: {len(trackers)} products", flush=True)
                backoff = 1
                while True:
                    remain = end - time.time()
                    if remain <= 0:
                        return
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=min(remain, 90))
                    except asyncio.TimeoutError:
                        if time.time() >= end:
                            return
                        print("[ws] no data for 90s; reconnecting", flush=True)
                        break
                    msg = json.loads(raw)
                    if msg.get("method") != "channelMessage":
                        continue
                    ch = msg["params"]["channel"]
                    m = msg["params"]["message"]
                    now = time.time()
                    if ch.startswith("lightning_board_snapshot_"):
                        code = ch[len("lightning_board_snapshot_"):]
                        trackers[code].on_snapshot(m)
                    elif ch.startswith("lightning_executions_"):
                        code = ch[len("lightning_executions_"):]
                        trackers[code].n_trades += len(m)
                    elif ch.startswith("lightning_board_"):
                        code = ch[len("lightning_board_"):]
                        trackers[code].on_diff(m, now)
        except Exception as e:
            if time.time() >= end:
                return
            print(f"[ws] error ({type(e).__name__}: {e}); retry in {backoff}s", flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)
        for tr in trackers.values():
            tr.reset_books()   # 再接続後はスナップショット再受信から


async def sampler_loop(trackers: dict[str, Tracker], end: float, window: float):
    last_stat = time.time()
    t0 = time.time()
    while True:
        now = time.time()
        if now >= end:
            return
        for tr in trackers.values():
            tr.sample(now, window)
        if now - last_stat >= 30:
            live = sum(1 for tr in trackers.values() if tr.have_snap)
            n = sum(len(tr.samples) for tr in trackers.values())
            print(f"[stat] {int(now - t0)}s/{int(end - t0)}s  "
                  f"board受信 {live}/{len(trackers)} 商品, 累計 {n} サンプル", flush=True)
            last_stat = now
        await asyncio.sleep(SAMPLE_DT)


async def scan(trackers: dict[str, Tracker], minutes: float, window: float):
    end = time.time() + minutes * 60
    await asyncio.gather(ws_loop(trackers, end), sampler_loop(trackers, end, window))


# ---- 出力 ----

def report(trackers: dict[str, Tracker], horizon: float, fee_bps: float, csv_path: Path):
    rows, insufficient = [], []
    for tr in trackers.values():
        m = tr.metrics(horizon, fee_bps)
        if m is None:
            insufficient.append(tr.code)
        else:
            rows.append(m)
    rows.sort(key=lambda r: (r["edge_bps"] == r["edge_bps"], r["edge_bps"]), reverse=True)

    hdr = (f"{'#':>2} {'product':<12} {'edge_bps':>8} {'ic':>6} {'hit':>5} "
           f"{'vol_bps':>7} {'sprd_bps':>8} {'tr/min':>7} {'upd/min':>8} {'smpl':>5}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for i, r in enumerate(rows, 1):
        print(f"{i:>2} {r['product']:<12} {r['edge_bps']:>8.2f} {r['ic']:>6.3f} "
              f"{r['hit']:>5.2f} {r['vol_bps']:>7.2f} {r['spread_bps']:>8.2f} "
              f"{r['trades_per_min']:>7.1f} {r['updates_per_min']:>8.0f} {r['samples']:>5}")
    for code in insufficient:
        print(f"   {code:<12} (データ不足: サンプル{MIN_SAMPLES}件未満)")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["product", "samples", "trades_per_min",
                                          "updates_per_min", "spread_bps", "vol_bps",
                                          "ic", "hit", "edge_bps"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] {csv_path}")

    best = next((r for r in rows if r["edge_bps"] == r["edge_bps"]), None)
    if best:
        print(f"\n推奨: {best['product']}  (推定エッジ {best['edge_bps']:+.2f} bps/trade, "
              f"ic {best['ic']:.3f}, 的中率 {best['hit']:.2f})")
        print(f"  live.exe を起動後:  python adapters\\bitflyer_adapter.py "
              f"--product {best['product']}")
        if best["edge_bps"] <= 0:
            print("  ※ 全商品で推定エッジが非正。この時間帯はスプレッドがシグナルの"
                  "値幅を上回っており、テイカー往復では期待値マイナス。")
    print("※ 短時間観測の目安であり、時間帯・相場状況で変わる。手数料は "
          f"--fee-bps={fee_bps} として計算。")


# ---- 自己検証(ネット不要) ----

def selftest():
    rng = np.random.default_rng(7)
    n, k = 4000, int(round(5.0 / SAMPLE_DT))

    def build(alpha: float) -> Tracker:
        tr = Tracker("TEST_JPY")
        ofi = rng.normal(size=n)
        ret = np.zeros(n)
        ret[1:] = alpha * ofi[:-1] + 1e-5 * rng.normal(size=n - 1)
        mid = 1e6 * np.cumprod(1 + ret)
        tr.samples = [(i * SAMPLE_DT, mid[i], 100.0, ofi[i]) for i in range(n)]
        tr.n_updates = tr.n_trades = n
        return tr

    m1 = build(2e-5).metrics(horizon=5.0, fee_bps=0.0)
    m0 = build(0.0).metrics(horizon=5.0, fee_bps=0.0)
    print(f"相関あり(alpha=2e-5): ic={m1['ic']:.3f} hit={m1['hit']:.2f} "
          f"vol={m1['vol_bps']:.2f}bps spread={m1['spread_bps']:.2f}bps")
    print(f"相関なし(alpha=0):    ic={m0['ic']:.3f} hit={m0['hit']:.2f}")
    assert m1["ic"] > 0.15, f"相関ありのICが低すぎる: {m1['ic']}"
    assert m1["hit"] > 0.55, f"相関ありの的中率が低すぎる: {m1['hit']}"
    assert abs(m0["ic"]) < 0.08, f"相関なしのICがゼロから離れすぎ: {m0['ic']}"
    assert 0.40 < m0["hit"] < 0.60, f"相関なしの的中率が0.5から離れすぎ: {m0['hit']}"
    assert abs(m1["spread_bps"] - 1.0) < 0.05, f"spread_bps計算異常: {m1['spread_bps']}"
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description="bitFlyer全商品を観測して銘柄をランク付け")
    ap.add_argument("--minutes", type=float, default=10.0, help="観測時間(分)")
    ap.add_argument("--window", type=float, default=5.0, help="OFI集計窓(秒)")
    ap.add_argument("--horizon", type=float, default=5.0, help="採点ホライズン(秒)")
    ap.add_argument("--fee-bps", type=float, default=0.0, help="片道手数料(bps)")
    ap.add_argument("--products", default="", help="カンマ区切りで対象を固定(既定: 自動列挙)")
    ap.add_argument("--no-fx", action="store_true", help="FX_BTC_JPYを除外")
    ap.add_argument("--selftest", action="store_true", help="合成データで指標計算を検証")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if args.products:
        codes = [c.strip() for c in args.products.split(",") if c.strip()]
    else:
        codes = get_products(include_fx=not args.no_fx)
    print(f"[scan] 対象 {len(codes)} 商品: {', '.join(codes)}")
    print(f"[scan] {args.minutes:g}分観測 (window={args.window:g}s, "
          f"horizon={args.horizon:g}s)", flush=True)

    trackers = {c: Tracker(c) for c in codes}
    try:
        asyncio.run(scan(trackers, args.minutes, args.window))
    except KeyboardInterrupt:
        print("\n[exit] 中断 — ここまでのデータで集計します")
    report(trackers, args.horizon, args.fee_bps, HERE.parent / "scan_products.csv")


if __name__ == "__main__":
    main()
