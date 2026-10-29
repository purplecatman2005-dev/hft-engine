# bitflyer_adapter.py — bitFlyer公開WebSocket → C++エンジン(live.exe)ブリッジ
#
# 認証不要の公開板情報(BTC_JPY)を購読し、レベル差分を合成注文(A/X/D)に変換して
# wire.py のバイナリ形式で localhost TCP の live.exe に流す。約定は 'T' として記録専用。
#
# 使い方:
#   1) live.exe を先に起動(TCP 9401 で待受)
#   2) python adapters/bitflyer_adapter.py [--product BTC_JPY] [--port 9401]
#
# 設計:
#   - ミラー板(全レベル, dict)は常に完全維持。ワイヤ出力は価格窓内のみ
#   - 窓は初回スナップショットのmidを中心にアンカー('R'で通知)。
#     bestが窓端に近づいたら再アンカー('R'+板再送)。OFIの連続性はそこで切れる
#   - スナップショット再受信は「範囲内の突き合わせ」で処理(リセットしない)
#   - WS切断は自動再接続し、次のスナップショットで全リセット
import argparse
import asyncio
import json
import socket
import sys
import time

sys.path.insert(0, __import__("os").path.dirname(__file__))
import wire

QTY_UNIT = 1e-6          # 1数量単位 = 1e-6 BTC
REANCHOR_MARGIN = 4096   # bestが窓端からこのティック以内なら再アンカー
ID_RESET_LIMIT = 30_000_000

BUY, SELL = 0, 1


class EngineDown(Exception):
    """live.exe へのTCP送信失敗(WS切断と区別して致命扱いにする)"""


def units(size: float) -> int:
    # u32上限ガード: 1レベル4000 BTC超は実質壁であり数量精度は不要
    return min(int(round(size / QTY_UNIT)), 4_000_000_000)


class SyntheticBook:
    """レベル数量→合成注文列の変換器。ミラー板と注文FIFOを保持する。"""

    def __init__(self):
        self.mirror = [{}, {}]        # [side][price_jpy] -> units(全レベル)
        self.orders = [{}, {}]        # [side][price_jpy] -> list[[id, qty]](窓内のみ)
        self.base = None              # 実価格 = base + (tick - PX_LO)
        self.next_id = 0

    def anchored(self) -> bool:
        return self.base is not None

    def tick(self, price: int) -> int:
        return wire.PX_LO + (price - self.base)

    def in_window(self, price: int) -> bool:
        return 0 <= price - self.base < wire.PX_WINDOW

    def best(self, side: int):
        m = self.mirror[side]
        if not m:
            return None
        return max(m) if side == wire.BID else min(m)

    def sync_level(self, side: int, price: int, target: int, out: bytearray):
        # ミラー更新
        if target > 0:
            self.mirror[side][price] = target
        else:
            self.mirror[side].pop(price, None)
        if not self.anchored() or not self.in_window(price):
            return
        q = self.orders[side].get(price)
        cur = sum(o[1] for o in q) if q else 0
        if target > cur:
            oid = self.next_id
            self.next_id += 1
            if q is None:
                q = self.orders[side][price] = []
            q.append([oid, target - cur])
            out += wire.add(oid, side, self.tick(price), target - cur)
        elif target < cur:
            need = cur - target
            while need > 0 and q:
                oid, qty = q[0]
                if qty <= need:
                    out += wire.delete(oid)
                    need -= qty
                    q.pop(0)
                else:
                    out += wire.cancel(oid, need)
                    q[0][1] = qty - need
                    need = 0
            if not q:
                self.orders[side].pop(price, None)

    def full_reset(self, out: bytearray):
        """現在のミラー板からアンカーを取り直し、'R'+全レベル再送を出力する。"""
        bb, ba = self.best(wire.BID), self.best(wire.ASK)
        if bb is None or ba is None:
            return False
        mid = (bb + ba) // 2
        self.base = mid - wire.PX_WINDOW // 2
        self.orders = [{}, {}]
        self.next_id = 0
        out += wire.reset(self.base)
        for side in (wire.BID, wire.ASK):
            for price in sorted(self.mirror[side]):
                if not self.in_window(price):
                    continue
                oid = self.next_id
                self.next_id += 1
                self.orders[side][price] = [[oid, self.mirror[side][price]]]
                out += wire.add(oid, side, self.tick(price), self.mirror[side][price])
        return True

    def needs_reanchor(self) -> bool:
        if not self.anchored():
            return False
        if self.next_id > ID_RESET_LIMIT:
            return True
        for side in (wire.BID, wire.ASK):
            b = self.best(side)
            if b is None:
                continue
            t = self.tick(b)
            if t < wire.PX_LO + REANCHOR_MARGIN or t >= wire.PX_HI - REANCHOR_MARGIN:
                return True
        return False


class Adapter:
    def __init__(self, product: str, host: str, port: int):
        self.product = product
        self.host, self.port = host, port
        self.book = SyntheticBook()
        self.sock = None
        self.force_reset = True       # 初回/再接続後: 次のスナップショットで全リセット
        self.n_trades = 0
        self.n_resets = 0

    def connect_engine(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=5)
        self.sock.settimeout(30)

    def send(self, out: bytearray):
        if not out:
            return
        try:
            self.sock.sendall(out)
        except OSError as e:
            raise EngineDown(str(e)) from e

    # ---- チャネル別ハンドラ ----
    def on_snapshot(self, m: dict, out: bytearray):
        snap = [{}, {}]
        for side, key in ((wire.BID, "bids"), (wire.ASK, "asks")):
            for lv in m[key]:
                u = units(lv["size"])
                if u > 0:
                    snap[side][int(round(lv["price"]))] = u
        if not snap[wire.BID] or not snap[wire.ASK]:
            return
        if self.force_reset or not self.book.anchored():
            self.book.mirror = snap
            if self.book.full_reset(out):
                self.force_reset = False
                self.n_resets += 1
            return
        # 範囲内突き合わせ: スナップショットは各サイド上位300レベルのみなので、
        # その価格範囲の外側(より不利な価格)のミラーは保持する
        for side in (wire.BID, wire.ASK):
            worst = min(snap[side]) if side == wire.BID else max(snap[side])
            covered = (lambda p: p >= worst) if side == wire.BID else (lambda p: p <= worst)
            for p in [p for p in self.book.mirror[side] if covered(p)]:
                if p not in snap[side]:
                    self.book.sync_level(side, p, 0, out)
            for p, u in snap[side].items():
                if self.book.mirror[side].get(p, 0) != u:
                    self.book.sync_level(side, p, u, out)

    def on_diff(self, m: dict, out: bytearray):
        if not self.book.anchored():
            return
        for side, key in ((wire.BID, "bids"), (wire.ASK, "asks")):
            for lv in m[key]:
                self.book.sync_level(side, int(round(lv["price"])), units(lv["size"]), out)

    def on_executions(self, execs: list, out: bytearray):
        if not self.book.anchored():
            return
        for e in execs:
            price = int(round(e["price"]))
            if not self.book.in_window(price):
                continue
            side = BUY if e.get("side") == "BUY" else SELL
            self.n_trades += 1
            out += wire.trade(side, self.book.tick(price), units(e["size"]))

    def handle(self, channel: str, message):
        out = bytearray()
        if channel.startswith("lightning_board_snapshot"):
            self.on_snapshot(message, out)
        elif channel.startswith("lightning_board"):
            self.on_diff(message, out)
        elif channel.startswith("lightning_executions"):
            self.on_executions(message, out)
        if self.book.needs_reanchor():
            r = bytearray()
            if self.book.full_reset(r):
                self.n_resets += 1
                out += r
        self.send(out)

    def status(self) -> str:
        bb, ba = self.book.best(wire.BID), self.book.best(wire.ASK)
        if bb is None or ba is None:
            return "book: (empty)"
        return (f"bid {bb:,} x {self.book.mirror[wire.BID][bb] * QTY_UNIT:.4f} | "
                f"ask {ba:,} x {self.book.mirror[wire.ASK][ba] * QTY_UNIT:.4f} | "
                f"spread {ba - bb:,} JPY | trades {self.n_trades} | resets {self.n_resets}")


async def run(adapter: Adapter):
    import websockets
    url = "wss://ws.lightstream.bitflyer.com/json-rpc"
    channels = [f"lightning_board_snapshot_{adapter.product}",
                f"lightning_board_{adapter.product}",
                f"lightning_executions_{adapter.product}"]
    backoff = 1
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                for i, ch in enumerate(channels):
                    await ws.send(json.dumps({"jsonrpc": "2.0", "method": "subscribe",
                                              "params": {"channel": ch}, "id": i + 1}))
                print(f"[ws] connected: {adapter.product}", flush=True)
                backoff = 1
                last_stat = time.time()
                while True:
                    # ウォッチドッグ: 購読がサイレント死する(pingは通る)ことがあるため
                    # 90秒無データで接続を張り直す
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=90)
                    except asyncio.TimeoutError:
                        print("[ws] no data for 90s; reconnecting", flush=True)
                        break
                    msg = json.loads(raw)
                    if msg.get("method") != "channelMessage":
                        continue
                    adapter.handle(msg["params"]["channel"], msg["params"]["message"])
                    now = time.time()
                    if now - last_stat >= 5:
                        print(f"[stat] {adapter.status()}", flush=True)
                        last_stat = now
                adapter.force_reset = True
        except EngineDown as e:
            print(f"[fatal] engine connection lost: {e}", flush=True)
            print("        restart live.exe, then restart this adapter", flush=True)
            sys.exit(1)
        except Exception as e:
            print(f"[ws] error ({type(e).__name__}: {e}); retry in {backoff}s", flush=True)
            adapter.force_reset = True
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


def main():
    ap = argparse.ArgumentParser(description="bitFlyer public WS -> hft-engine bridge")
    ap.add_argument("--product", default="BTC_JPY")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9401)
    args = ap.parse_args()

    adapter = Adapter(args.product, args.host, args.port)
    try:
        adapter.connect_engine()
    except OSError as e:
        print(f"[fatal] live.exe not reachable at {args.host}:{args.port} ({e})")
        print("        start live.exe first: hft-engine> live.exe")
        sys.exit(1)
    print(f"[tcp] engine connected: {args.host}:{args.port}", flush=True)
    try:
        asyncio.run(run(adapter))
    except KeyboardInterrupt:
        print("\n[exit] stopped by user")


if __name__ == "__main__":
    main()
