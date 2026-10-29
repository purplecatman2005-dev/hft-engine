# -*- coding: utf-8 -*-
"""
train_version.py — evolve用バージョン別GBDT学習 + C++ヘッダ生成

evolve/versions/<vNNN>/train_config.json を読み、LightGBMを学習して
同ディレクトリに ml_model.hpp と train_report.json を書く。
既存の ml_train_export.py / ml_model.hpp は一切変更しない(独立実装)。

使い方: python train_version.py v002
"""
import json
import math
import os
import sys
from glob import glob

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(HERE)

# ml_features.hpp の ML_FEAT_NAMES と同一順序であること(C++側 f[] のインデックス)
FEAT_NAMES = [
    "ofi_z64", "ofi_z256", "micro_prem", "spread", "imbalance", "ret32", "ret128",
    "ofi_z16", "spread_dev64", "rvol64", "ret8", "micro_prem_rel",
]

SYNTHETIC_FILES = [
    "ml_features.csv", "ml_features_s43.csv", "ml_features_s44.csv",
    "ml_features_s45.csv", "ml_features_s46.csv",
]

# AIが動かせるノブと許容範囲。範囲外はクランプ、未知キーは拒否。
CONFIG_SPEC = {
    "trees":            (int,   10,    2000),
    "leaves":           (int,   3,     127),
    "lr":               (float, 0.001, 0.3),
    "min_data_in_leaf": (int,   10,    2000),
    "bagging_fraction": (float, 0.3,   1.0),
    "feature_fraction": (float, 0.3,   1.0),
    "fire_pct":         (float, 80.0,  99.5),
    "use_synthetic":    (bool,  None,  None),
    "use_live":         (bool,  None,  None),
    "live_weight":      (float, 0.1,   20.0),
    "feature_subset":   (list,  None,  None),   # None または FEAT_NAMES の部分集合
}

DEFAULT_CONFIG = {
    "trees": 400, "leaves": 15, "lr": 0.03, "min_data_in_leaf": 200,
    "bagging_fraction": 0.8, "feature_fraction": 1.0, "fire_pct": 97.0,
    "use_synthetic": True, "use_live": False, "live_weight": 3.0,
    "feature_subset": None,
}


def validate_config(cfg):
    """設定を検証・クランプして返す。不正なら ValueError。evolve.pyからも使用。"""
    if not isinstance(cfg, dict):
        raise ValueError("config はオブジェクトである必要があります")
    unknown = set(cfg) - set(CONFIG_SPEC) - {"note"}
    if unknown:
        raise ValueError(f"未知の設定キー: {sorted(unknown)}")
    out = dict(DEFAULT_CONFIG)
    if "note" in cfg:
        out["note"] = str(cfg["note"])[:500]
    for k, (typ, lo, hi) in CONFIG_SPEC.items():
        if k not in cfg or cfg[k] is None:
            if k in cfg:
                out[k] = None if k == "feature_subset" else out[k]
            continue
        v = cfg[k]
        if typ is bool:
            out[k] = bool(v)
        elif typ is list:
            if not isinstance(v, list):
                raise ValueError(f"{k} はリストか null")
            bad = [x for x in v if x not in FEAT_NAMES]
            if bad:
                raise ValueError(f"feature_subset に未知の特徴量: {bad}")
            if len(v) < 3:
                raise ValueError("feature_subset は3特徴量以上")
            out[k] = list(dict.fromkeys(v))  # 順序維持で重複除去
        else:
            v = typ(v)
            out[k] = min(max(v, lo), hi)
    if not out["use_synthetic"] and not out["use_live"]:
        raise ValueError("use_synthetic / use_live の少なくとも一方は true")
    return out


def load_data(cfg):
    """設定に従い学習データを (df, weights, sources) で返す。時系列順ブロック連結。"""
    import pandas as pd
    frames, weights, sources = [], [], []
    if cfg["use_synthetic"]:
        for name in SYNTHETIC_FILES:
            p = os.path.join(ENGINE, name)
            if not os.path.exists(p):
                raise SystemExit(f"合成特徴量が見つかりません: {p}")
            df = pd.read_csv(p)
            frames.append(df)
            weights += [1.0] * len(df)
            sources.append(f"synthetic:{name}({len(df)})")
    if cfg["use_live"]:
        live_files = sorted(glob(os.path.join(HERE, "data", "live_features_*.csv")))
        if not live_files and not cfg["use_synthetic"]:
            raise SystemExit("live特徴量が evolve/data/ にありません(use_live のみは不可)")
        for p in live_files:
            df = pd.read_csv(p)
            if len(df) == 0:
                continue
            frames.append(df)
            weights += [cfg["live_weight"]] * len(df)
            sources.append(f"live:{os.path.basename(p)}({len(df)})")
    if not frames:
        raise SystemExit("学習データが空です")
    full = pd.concat(frames, ignore_index=True)
    if len(full) < 500:
        raise SystemExit(f"学習データが少なすぎます({len(full)}行 < 500)")
    return full, weights, sources


def emit_node(node, lines, indent, remap):
    if "leaf_value" in node:
        lines.append(f"{indent}return {node['leaf_value']!r};")
        return
    if node.get("decision_type") != "<=":
        raise SystemExit(f"未対応の分岐型: {node.get('decision_type')}")
    thr = float(node["threshold"])
    orig = remap[node["split_feature"]]   # 学習列 -> C++の f[] インデックス
    lines.append(f"{indent}if (f[{orig}] <= {thr!r}) {{")
    emit_node(node["left_child"], lines, indent + "    ", remap)
    lines.append(f"{indent}}} else {{")
    emit_node(node["right_child"], lines, indent + "    ", remap)
    lines.append(f"{indent}}}")


def codegen(booster, threshold, used_names, version):
    remap = [FEAT_NAMES.index(n) for n in used_names]
    dump = booster.dump_model()
    trees = dump["tree_info"]
    lines = [
        "#pragma once",
        f"// ml_model.hpp ({version}) — evolve/train_version.py が生成したGBDT推論コード。手編集しない。",
        f"// trees={len(trees)}  features={used_names} (f[]は12特徴量の元インデックス)",
        "",
        "namespace mlmodel {",
        "",
        f"inline constexpr int N_FEATURES = {len(FEAT_NAMES)};",
        f"inline constexpr double THRESHOLD = {threshold!r};   // |pred| がこれを超えたら発火",
        "",
    ]
    for k, t in enumerate(trees):
        lines.append(f"inline double tree_{k}(const double* f) {{")
        emit_node(t["tree_structure"], lines, "    ", remap)
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


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if len(sys.argv) != 2:
        print("usage: python train_version.py vNNN")
        return 1
    version = sys.argv[1]
    vdir = os.path.join(HERE, "versions", version)
    cfg_path = os.path.join(vdir, "train_config.json")
    if not os.path.exists(cfg_path):
        print(f"config not found: {cfg_path}")
        return 1
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = validate_config(json.load(fh))

    import numpy as np
    import lightgbm as lgb

    df, weights, sources = load_data(cfg)
    used = cfg["feature_subset"] or FEAT_NAMES
    missing = [c for c in used if c not in df.columns]
    if missing:
        raise SystemExit(f"データに列がありません: {missing}")
    X = df[used].to_numpy(dtype=np.float64)
    y = df["dmid"].to_numpy(dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    split = int(len(df) * 0.8)            # 時系列順に前80%で学習
    print(f"[{version}] rows: train={split} valid={len(df) - split}  "
          f"features={len(used)}  sources={len(sources)}")

    train = lgb.Dataset(X[:split], label=y[:split], weight=w[:split])
    booster = lgb.train(
        {
            "objective": "regression",
            "learning_rate": cfg["lr"],
            "num_leaves": cfg["leaves"],
            "min_data_in_leaf": cfg["min_data_in_leaf"],
            "feature_fraction": cfg["feature_fraction"],
            "bagging_fraction": cfg["bagging_fraction"],
            "bagging_freq": 1,
            "verbosity": -1,
            "seed": 7,
        },
        train,
        num_boost_round=cfg["trees"],
    )

    pred_v = booster.predict(X[split:])
    yv = y[split:]
    corr = float(np.corrcoef(pred_v, yv)[0, 1]) if yv.std() > 0 else float("nan")
    nz = yv != 0
    dir_acc = float((np.sign(pred_v[nz]) == np.sign(yv[nz])).mean()) if nz.any() else float("nan")
    thr = float(np.percentile(np.abs(pred_v), cfg["fire_pct"]))
    fired = np.abs(pred_v) > thr
    fired_nz = fired & nz
    fired_acc = float((np.sign(pred_v[fired_nz]) == np.sign(yv[fired_nz])).mean()) \
        if fired_nz.any() else float("nan")
    imp = sorted(zip(used, booster.feature_importance("gain").tolist()),
                 key=lambda t: -t[1])

    out_hpp = os.path.join(vdir, "ml_model.hpp")
    with open(out_hpp, "w", encoding="utf-8") as fh:
        fh.write(codegen(booster, thr, used, version))

    def _clean(x):
        return None if isinstance(x, float) and not math.isfinite(x) else x
    report = {
        "version": version,
        "rows_train": split, "rows_valid": len(df) - split,
        "sources": sources,
        "config": {k: v for k, v in cfg.items() if k != "note"},
        "valid_corr": _clean(corr),
        "valid_dir_acc": _clean(dir_acc),
        "fire_threshold": thr,
        "fired_dir_acc": _clean(fired_acc),
        "feature_importance_gain": [[n, round(g, 1)] for n, g in imp],
    }
    with open(os.path.join(vdir, "train_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"[{version}] valid corr={corr:+.4f} dir_acc={dir_acc:.1%} "
          f"fired_acc={fired_acc if math.isfinite(fired_acc) else float('nan'):.1%} "
          f"th(p{cfg['fire_pct']:g})={thr:.5f}")
    print(f"[{version}] wrote {out_hpp}")
    print("TRAIN_REPORT_JSON:" + json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
