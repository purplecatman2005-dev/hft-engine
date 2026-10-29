# -*- coding: utf-8 -*-
"""
evolve.py — hft-engine 自動モデル進化ループ(予測vs実測照合 → Opus 4.8改善)

1サイクルの流れ:
  1. 収録: adapter + live.exe を起動し、現在のbitFlyer BTC_JPY板を N分間収録
  2. 照合: eval_ml_vNNN.exe で現行モデルの予測と実測(5秒先mid)を照合
     → 整合性(corr/方向的中)・発火シグナルのペーパー収支・再学習用特徴量を出力
  3. 記録: history/ledger.csv と history/history.md に軌跡を追記
  4. 改善: 評価サマリ+履歴を Opus 4.8 (claude CLI) に渡し、次バージョンの
     train_config.json を生成させる(旧バージョンは残る。新規作成のみ)
  5. 再学習+ビルド: train_version.py → ml_model.hpp → eval_ml_v(N+1).exe

Claude の利用制限を検知したらメッセージを出して自動停止する。
状態は state.json に保存され、再実行で続きから再開できる。

前提: `claude` CLI がログイン済みでPATHにあること(Claude Codeと同じ認証)。

使い方(このディレクトリまたはどこからでも):
  python evolve\\evolve.py                     # 3サイクル、収録10分/サイクル
  python evolve\\evolve.py --cycles 5 --capture-min 15
  python evolve\\evolve.py --reuse-capture ..\\captures\\capture_XXXX.bin  # 収録スキップ(検証用)
  python evolve\\evolve.py --no-improve       # 照合と記録のみ(AI改善なし)
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from glob import glob

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(HERE)
HORSE = os.path.dirname(ENGINE)

sys.path.insert(0, HERE)
from train_version import validate_config, DEFAULT_CONFIG  # noqa: E402

STATE_PATH = os.path.join(HERE, "state.json")
LEDGER_PATH = os.path.join(HERE, "history", "ledger.csv")
HISTORY_MD = os.path.join(HERE, "history", "history.md")
CLAUDE_MODEL = "claude-opus-4-8"
CLAUDE_TIMEOUT = 900
TRAIN_TIMEOUT = 1800
BUILD_TIMEOUT = 600
HISTORY_ROWS_IN_PROMPT = 10

LEDGER_COLS = ("cycle,timestamp,version,capture,duration_s,n_samples,corr,dir_acc,"
               "n_signals,n_scored,hit_rate,avg_net_bps,sum_net_bps,cum_net_bps,"
               "threshold,direction")

# 利用制限らしきメッセージの検知パターン(CLIの文言は変わりうるので広めに)
LIMIT_PATTERNS = re.compile(
    r"(usage limit|rate limit|limit (?:reached|will reset|resets)|hit your|"
    r"out of (?:extra )?usage|5.?hour|quota|credit balance|too many requests|"
    r"overloaded|利用制限|上限に達し)", re.IGNORECASE)


class UsageLimitReached(Exception):
    pass


def _python_exe():
    cand = os.path.join(HORSE, "Scripts", "python.exe")
    return cand if os.path.exists(cand) else sys.executable


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return {"current_version": None, "cycle": 0, "cum_net_bps": 0.0}


def save_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)


def ensure_dirs():
    for d in ("versions", "data", "logs", "build", "history",
              os.path.join("history", "decisions")):
        os.makedirs(os.path.join(HERE, d), exist_ok=True)


def init_v001(state):
    """v001 = 現行 ml_model.hpp のコピー(原本は無変更)。"""
    vdir = os.path.join(HERE, "versions", "v001")
    if not os.path.exists(os.path.join(vdir, "ml_model.hpp")):
        os.makedirs(vdir, exist_ok=True)
        src = os.path.join(ENGINE, "ml_model.hpp")
        if not os.path.exists(src):
            raise SystemExit(f"ベースモデルがありません: {src}(先に build_ml.bat を実行)")
        shutil.copyfile(src, os.path.join(vdir, "ml_model.hpp"))
        cfg = dict(DEFAULT_CONFIG)
        cfg["note"] = "v001: 現行ml_model.hpp(合成seed42-46学習)のコピー。この設定は起点の参考値"
        with open(os.path.join(vdir, "train_config.json"), "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, ensure_ascii=False, indent=2)
        print("初期化: versions/v001 を現行モデルから作成しました")
    if not state["current_version"]:
        state["current_version"] = "v001"
        save_state(state)


def build_eval(version):
    exe = os.path.join(HERE, "build", f"eval_ml_{version}.exe")
    # 壊れた(空・部分書き込みの)exeを「ビルド済み」と誤認しない
    if os.path.exists(exe) and os.path.getsize(exe) > 100_000:
        return exe
    print(f"🔨 eval_ml_{version}.exe をビルド中...")
    proc = subprocess.run(
        ["cmd", "/c", os.path.join(HERE, "build_eval.bat"), version],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=BUILD_TIMEOUT, cwd=ENGINE)
    if proc.returncode != 0 or not os.path.exists(exe):
        raise RuntimeError("ビルド失敗:\n" +
                           ((proc.stdout or "") + (proc.stderr or ""))[-2000:])
    return exe


def capture_live(minutes, port, stamp):
    """live.exe + adapter を起動して現在の板をminutes分収録し、captureパスを返す。"""
    live_exe = os.path.join(ENGINE, "live.exe")
    if not os.path.exists(live_exe):
        raise SystemExit(f"live.exe がありません: {live_exe}(build.bat でビルド)")
    t0 = time.time()
    live_log = open(os.path.join(HERE, "logs", f"{stamp}_live.log"), "w",
                    encoding="utf-8", errors="replace")
    ad_log = open(os.path.join(HERE, "logs", f"{stamp}_adapter.log"), "w",
                  encoding="utf-8", errors="replace")
    print(f"📡 ライブ収録開始({minutes}分, port {port})...")
    live = subprocess.Popen([live_exe, str(port)], cwd=ENGINE,
                            stdout=live_log, stderr=subprocess.STDOUT)
    adapter = None
    try:
        time.sleep(3)
        adapter = subprocess.Popen(
            [_python_exe(), os.path.join(ENGINE, "adapters", "bitflyer_adapter.py"),
             "--port", str(port)],
            cwd=os.path.join(ENGINE, "adapters"),
            stdout=ad_log, stderr=subprocess.STDOUT)
        deadline = t0 + minutes * 60
        while time.time() < deadline:
            time.sleep(5)
            if live.poll() is not None:
                raise RuntimeError("live.exe が収録中に終了しました(ログ参照: "
                                   f"logs/{stamp}_live.log)")
            if adapter.poll() is not None:
                raise RuntimeError("adapter が収録中に終了しました(ネットワーク?ログ参照: "
                                   f"logs/{stamp}_adapter.log)")
            remain = int(deadline - time.time())
            print(f"\r   収録中... 残り {remain // 60}:{remain % 60:02d} ",
                  end="", flush=True)
        print()
    finally:
        for p in (adapter, live):
            if p and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
        live_log.close()
        ad_log.close()

    caps = [p for p in glob(os.path.join(ENGINE, "captures", "capture_*.bin"))
            if os.path.getmtime(p) >= t0 - 5]
    if not caps:
        raise RuntimeError("収録ファイルが見つかりません(接続失敗の可能性。"
                           f"logs/{stamp}_adapter.log を確認)")
    cap = max(caps, key=os.path.getmtime)
    size = os.path.getsize(cap)
    if size < 50_000:
        raise RuntimeError(f"収録データが小さすぎます({size}B)。接続や流動性を確認: {cap}")
    print(f"   収録完了: {os.path.basename(cap)} ({size / 1e6:.1f} MB)")
    return cap


def run_eval(version, capture, stamp, args):
    exe = build_eval(version)
    out = os.path.join(HERE, "logs", f"{stamp}_{version}_signals.csv")
    feats = os.path.join(HERE, "data", f"live_features_{stamp}.csv")
    summ = os.path.join(HERE, "logs", f"{stamp}_{version}_summary.json")
    print(f"🔍 照合中: {version} × {os.path.basename(capture)}")
    proc = subprocess.run(
        [exe, capture, "--horizon-ms", str(args.horizon_ms),
         "--cooldown", str(args.cooldown), "--fee-bps", str(args.fee_bps),
         "--tag", version, "--out", out, "--features-out", feats,
         "--summary", summ],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=600, cwd=ENGINE)
    sys.stdout.write(proc.stdout or "")
    if proc.returncode != 0:
        raise RuntimeError("eval_ml 失敗:\n" + (proc.stderr or proc.stdout or "")[-2000:])
    with open(summ, encoding="utf-8") as fh:
        summary = json.load(fh)
    # 再学習に使えない空の特徴量ファイルは残さない
    try:
        with open(feats, encoding="utf-8") as fh:
            n_rows = sum(1 for _ in fh) - 1
        if n_rows <= 0:
            os.remove(feats)
    except OSError:
        pass
    return summary


def record(state, cycle, version, summary, direction):
    state["cum_net_bps"] = state.get("cum_net_bps", 0.0) + summary["sum_net_bps"]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    if not os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, "w", encoding="utf-8-sig") as fh:
            fh.write(LEDGER_COLS + "\n")
    with open(LEDGER_PATH, "a", encoding="utf-8-sig") as fh:
        fh.write(",".join(str(x) for x in [
            cycle, ts, version, os.path.basename(summary["capture"]),
            summary["duration_s"], summary["n_samples"],
            f"{summary['corr_pred_dmid']:.4f}", f"{summary['dir_acc']:.4f}",
            summary["n_signals"], summary["n_scored"],
            f"{summary['hit_rate']:.4f}", f"{summary['avg_net_bps']:.3f}",
            f"{summary['sum_net_bps']:.3f}", f"{state['cum_net_bps']:.3f}",
            f"{summary['threshold']:.6g}",
            '"' + direction.replace('"', "'") + '"',
        ]) + "\n")
    if not os.path.exists(HISTORY_MD):
        with open(HISTORY_MD, "w", encoding="utf-8") as fh:
            fh.write("# evolve 履歴 — 予測vs実測の照合とモデル修正の軌跡\n\n"
                     "収支(net_bps)はペーパートレード: 発火時テイカー往復の "
                     "方向×mid変動 − スプレッド − 手数料。\n\n"
                     "| cycle | version | 日時 | サンプル | corr | 方向的中 | 発火 | 採点 "
                     "| 的中率 | 平均net_bps | 累積net_bps | AIの修正方向 |\n"
                     "|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    with open(HISTORY_MD, "a", encoding="utf-8") as fh:
        fh.write(f"| {cycle} | {version} | {ts} | {summary['n_samples']} "
                 f"| {summary['corr_pred_dmid']:+.4f} | {summary['dir_acc']:.1%} "
                 f"| {summary['n_signals']} | {summary['n_scored']} "
                 f"| {summary['hit_rate']:.1%} | {summary['avg_net_bps']:+.2f} "
                 f"| {state['cum_net_bps']:+.1f} | {direction.replace('|', '/')} |\n")
    save_state(state)


def call_claude(prompt):
    """claude -p で Opus 4.8 にプロンプトを投げ、応答テキストを返す。"""
    cmd = ["claude", "-p", "--model", CLAUDE_MODEL, "--output-format", "json"]
    try:
        proc = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=CLAUDE_TIMEOUT, cwd=HERE)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Claude CLI が {CLAUDE_TIMEOUT} 秒でタイムアウトしました")
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    result_text, is_error = None, False
    try:
        payload = json.loads(proc.stdout)
        result_text = payload.get("result")
        is_error = bool(payload.get("is_error"))
        combined += "\n" + str(result_text or "")
    except (json.JSONDecodeError, TypeError):
        result_text = proc.stdout
    if LIMIT_PATTERNS.search(combined):
        raise UsageLimitReached(combined.strip()[-500:])
    if proc.returncode != 0 or is_error or not (result_text or "").strip():
        raise RuntimeError("Claude CLI がエラーを返しました:\n" + combined.strip()[-1000:])
    return result_text


def parse_ai_response(text):
    """応答から {...} JSON を取り出し (analysis, direction, expected, config) を返す。"""
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        raise ValueError("応答にJSONが見つかりません:\n" + text[:500])
    obj = json.loads(text[s:e + 1])
    for k in ("analysis", "direction", "config"):
        if k not in obj:
            raise ValueError(f"応答JSONに '{k}' がありません")
    cfg = validate_config(obj["config"])
    return (str(obj["analysis"]), str(obj["direction"]),
            str(obj.get("expected_effect", "")), cfg)


def build_prompt(version, next_version, summary, args):
    with open(os.path.join(HERE, "prompts", "improve.md"), encoding="utf-8") as fh:
        tpl = fh.read()
    vdir = os.path.join(HERE, "versions", version)
    report = {}
    rp = os.path.join(vdir, "train_report.json")
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as fh:
            report = json.load(fh)
    with open(os.path.join(vdir, "train_config.json"), encoding="utf-8") as fh:
        cur_cfg = json.load(fh)
    hist = "(履歴なし: 初回サイクル)"
    if os.path.exists(HISTORY_MD):
        lines = [ln for ln in open(HISTORY_MD, encoding="utf-8")
                 if ln.startswith("|")]
        hist = "".join(lines[:2] + lines[-HISTORY_ROWS_IN_PROMPT:])
    live_rows = 0
    for p in glob(os.path.join(HERE, "data", "live_features_*.csv")):
        try:
            with open(p, encoding="utf-8") as fh:
                live_rows += max(0, sum(1 for _ in fh) - 1)
        except OSError:
            pass
    knobs = ("trees(10-2000) leaves(3-127) lr(0.001-0.3) min_data_in_leaf(10-2000) "
             "bagging_fraction(0.3-1.0) feature_fraction(0.3-1.0) fire_pct(80-99.5) "
             "use_synthetic(bool) use_live(bool) live_weight(0.1-20) "
             "feature_subset(null|list) note(任意メモ)")
    for token, val in [
        ("<<VERSION>>", version), ("<<NEXT_VERSION>>", next_version),
        ("<<SUMMARY_JSON>>", json.dumps(summary, ensure_ascii=False, indent=2)),
        ("<<TRAIN_REPORT_JSON>>", json.dumps(report, ensure_ascii=False, indent=2)),
        ("<<CURRENT_CONFIG_JSON>>", json.dumps(cur_cfg, ensure_ascii=False, indent=2)),
        ("<<HISTORY_TABLE>>", hist), ("<<LIVE_ROWS_TOTAL>>", str(live_rows)),
        ("<<KNOB_SPEC>>", knobs),
    ]:
        tpl = tpl.replace(token, val)
    return tpl


def train_new_version(next_version):
    print(f"🎓 {next_version} を再学習中...")
    proc = subprocess.run(
        [_python_exe(), os.path.join(HERE, "train_version.py"), next_version],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=TRAIN_TIMEOUT, cwd=ENGINE)
    sys.stdout.write(proc.stdout or "")
    if proc.returncode != 0:
        raise RuntimeError("再学習失敗:\n" +
                           ((proc.stdout or "") + (proc.stderr or ""))[-2000:])


def next_version_name(version):
    return f"v{int(version[1:]) + 1:03d}"


def print_limit_message(detail):
    print("\n" + "=" * 60)
    print("⛔ Claude(Opus 4.8)の利用制限に達したため自動停止しました。")
    print("   評価結果と履歴は保存済みです。制限が解除された後に")
    print("   同じコマンドを再実行すれば続きから再開できます。")
    print("=" * 60)
    print("検知メッセージ(末尾):")
    print("  " + detail.strip().replace("\n", "\n  ")[-400:])


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="hft-engine 自動モデル進化ループ")
    ap.add_argument("--cycles", type=int, default=3, help="実行サイクル数(既定3)")
    ap.add_argument("--capture-min", type=float, default=10.0,
                    help="1サイクルの収録分数(既定10)")
    ap.add_argument("--port", type=int, default=9402,
                    help="live.exe のポート(既定9402: 手動運用の9401と分離)")
    ap.add_argument("--horizon-ms", type=int, default=5000)
    ap.add_argument("--cooldown", type=int, default=64)
    ap.add_argument("--fee-bps", type=float, default=15.0,
                    help="片道テイカー手数料bps(既定15=0.15%%)")
    ap.add_argument("--reuse-capture", default=None,
                    help="収録せず既存captureで照合(動作検証用)")
    ap.add_argument("--no-improve", action="store_true",
                    help="照合と記録のみ(AI改善・再学習をしない)")
    args = ap.parse_args()

    ensure_dirs()
    state = load_state()
    init_v001(state)

    for i in range(args.cycles):
        cycle = state["cycle"] + 1
        version = state["current_version"]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n===== cycle {cycle} (model {version}) =====")

        build_eval(version)
        if args.reuse_capture:
            cap = os.path.abspath(args.reuse_capture)
            if not os.path.exists(cap):
                raise SystemExit(f"capture がありません: {cap}")
        else:
            cap = capture_live(args.capture_min, args.port, stamp)

        summary = run_eval(version, cap, stamp, args)

        if args.no_improve:
            record(state, cycle, version, summary, "(照合のみ・AI改善なし)")
            state["cycle"] = cycle
            save_state(state)
            continue

        nxt = next_version_name(version)
        prompt = build_prompt(version, nxt, summary, args)
        print(f"🤖 Opus 4.8 に {nxt} の設計を依頼中(最大{CLAUDE_TIMEOUT // 60}分)...")
        try:
            answer = call_claude(prompt)
            analysis, direction, expected, cfg = parse_ai_response(answer)
        except UsageLimitReached as e:
            record(state, cycle, version, summary, "(利用制限で停止: 改善は未実施)")
            state["cycle"] = cycle
            save_state(state)
            print_limit_message(str(e))
            return 0

        record(state, cycle, version, summary, direction)
        vdir = os.path.join(HERE, "versions", nxt)
        os.makedirs(vdir, exist_ok=True)
        with open(os.path.join(vdir, "train_config.json"), "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, ensure_ascii=False, indent=2)
        decision = {
            "cycle": cycle, "from_version": version, "to_version": nxt,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "analysis": analysis, "direction": direction,
            "expected_effect": expected, "config": cfg,
            "eval_summary": summary,
        }
        with open(os.path.join(HERE, "history", "decisions",
                               f"cycle_{cycle:03d}_{nxt}.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(decision, fh, ensure_ascii=False, indent=2)
        print(f"📝 修正方向: {direction}")

        train_new_version(nxt)
        build_eval(nxt)
        state["current_version"] = nxt
        state["cycle"] = cycle
        save_state(state)
        print(f"✅ cycle {cycle} 完了: {version} → {nxt}")

    print(f"\n全 {args.cycles} サイクル完了。履歴: evolve/history/history.md")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n中断しました(状態は state.json に保存済み。再実行で続きから)")
        sys.exit(130)
