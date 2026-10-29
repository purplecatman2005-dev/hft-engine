# -*- coding: utf-8 -*-
"""
GBDTシグナルモデルの学習と C++ ヘッダへのコンパイル

ml_dump_features.exe が出力した特徴量CSVで LightGBM 回帰を学習し、
全決定木を if-else 直書きの ml_model.hpp に変換する。生成コードは
分岐のみ・ヒープ確保なしで、backtest_ml.cpp のホットパスに埋め込まれる。

検証用に ml_verify.csv (特徴量 + Python側予測値) も出力し、
backtest_ml.exe --verify で C++ 生成コードとの完全一致を確認する。

使い方:
  python ml_train_export.py [--features ml_features.csv] [--trees 80]

--features はカンマ区切りで複数CSVを受け取り、その順に連結する
(複数seedのダンプを時系列ブロックとして扱う。前80%が学習、後20%が検証)。
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def emit_node(node: dict, lines: list[str], indent: str) -> None:
    if "leaf_value" in node:
        lines.append(f"{indent}return {node['leaf_value']!r};")
        return
    if node.get("decision_type") != "<=":
        raise SystemExit(f"未対応の分岐型: {node.get('decision_type')}")
    thr = float(node["threshold"])
    lines.append(f"{indent}if (f[{node['split_feature']}] <= {thr!r}) {{")
    emit_node(node["left_child"], lines, indent + "    ")
    lines.append(f"{indent}}} else {{")
    emit_node(node["right_child"], lines, indent + "    ")
    lines.append(f"{indent}}}")


def codegen(booster: lgb.Booster, threshold: float, n_feat: int, names) -> str:
    dump = booster.dump_model()
    trees = dump["tree_info"]
    lines = [
        "#pragma once",
        "// ml_model.hpp — ml_train_export.py が生成した GBDT 推論コード。手編集しない。",
        f"// trees={len(trees)}  features={list(names)}",
        "",
        "namespace mlmodel {",
        "",
        f"inline constexpr int N_FEATURES = {n_feat};",
        f"inline constexpr double THRESHOLD = {threshold!r};   // |pred| がこれを超えたら発火",
        "",
    ]
    for k, t in enumerate(trees):
        lines.append(f"inline double tree_{k}(const double* f) {{")
        emit_node(t["tree_structure"], lines, "    ")
        lines.append("}")
        lines.append("")
    lines.append("inline double predict(const double* f) {")
    lines.append("    double s = 0.0;")
    for k in range(len(trees)):
        lines.append(f"    s += tree_{k}(f);")
    lines.append("    return s;")
    lines.append("}")
    lines.append("")
    lines.append("}  // namespace mlmodel")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="GBDTシグナル学習 + C++コード生成")
    p.add_argument("--features", default=str(HERE / "ml_features.csv"))
    p.add_argument("--trees", type=int, default=80)
    p.add_argument("--leaves", type=int, default=31)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--fire-pct", type=float, default=90.0,
                   help="発火閾値 = 検証データ|pred|のこのパーセンタイル")
    p.add_argument("--out", default=str(HERE / "ml_model.hpp"))
    p.add_argument("--verify-out", default=str(HERE / "ml_verify.csv"))
    args = p.parse_args()

    df = pd.concat([pd.read_csv(f.strip()) for f in args.features.split(",")],
                   ignore_index=True)
    feat_cols = [c for c in df.columns if c not in ("event_idx", "dmid")]
    X = df[feat_cols].to_numpy(dtype=np.float64)
    y = df["dmid"].to_numpy(dtype=np.float64)
    split = int(len(df) * 0.8)          # 時系列順に前80%で学習
    print(f"rows: train={split} valid={len(df) - split}  features={feat_cols}")

    train = lgb.Dataset(X[:split], label=y[:split])
    booster = lgb.train(
        {
            "objective": "regression",
            "learning_rate": args.lr,
            "num_leaves": args.leaves,
            "min_data_in_leaf": 200,
            "feature_fraction": 1.0,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbosity": -1,
            "seed": 7,
        },
        train,
        num_boost_round=args.trees,
    )

    pred_v = booster.predict(X[split:])
    yv = y[split:]
    corr = float(np.corrcoef(pred_v, yv)[0, 1]) if yv.std() > 0 else float("nan")
    nz = yv != 0
    dir_acc = float((np.sign(pred_v[nz]) == np.sign(yv[nz])).mean()) \
        if nz.any() else float("nan")
    thr = float(np.percentile(np.abs(pred_v), args.fire_pct))
    fired = np.abs(pred_v) > thr
    fired_nz = fired & nz
    fired_acc = float((np.sign(pred_v[fired_nz]) == np.sign(yv[fired_nz])).mean()) \
        if fired_nz.any() else float("nan")
    print(f"valid corr(pred, dmid)      : {corr:+.4f}")
    print(f"valid direction accuracy    : {dir_acc:.1%} (全サンプル, mid不変除く)")
    print(f"fire threshold (p{args.fire_pct:.0f})    : {thr:.5f}")
    print(f"direction accuracy if fired : {fired_acc:.1%}")

    imp = sorted(zip(feat_cols, booster.feature_importance("gain")),
                 key=lambda t: -t[1])
    print("feature importance (gain)   : " +
          ", ".join(f"{n}={g:.0f}" for n, g in imp))

    Path(args.out).write_text(
        codegen(booster, thr, len(feat_cols), feat_cols), encoding="utf-8")
    print(f"wrote {args.out}")

    vi = np.linspace(split, len(df) - 1, min(500, len(df) - split), dtype=int)
    vdf = pd.DataFrame(X[vi], columns=[f"f{i}" for i in range(len(feat_cols))])
    vdf["pred"] = booster.predict(X[vi])
    vdf.to_csv(args.verify_out, index=False, float_format="%.17g")
    print(f"wrote {args.verify_out} ({len(vdf)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
