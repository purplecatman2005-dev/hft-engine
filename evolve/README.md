# evolve — 予測vs実測の照合と Opus 4.8 自動モデル進化

現在の bitFlyer BTC_JPY 板をライブ収録し、GBDTモデルの予測と実測(5秒先mid変動)を
照合。結果を Opus 4.8 (claude CLI) に渡して**新バージョンのモデルを自動生成**する
反復ループ。既存ファイル(`ml_model.hpp` / `ml_train_export.py` 等)は一切変更しない。

## 使い方

```bat
cd hft-engine
python evolve\evolve.py                      :: 3サイクル、収録10分/サイクル
python evolve\evolve.py --cycles 5 --capture-min 15
python evolve\evolve.py --no-improve         :: 照合と記録のみ(AIなし)
python evolve\evolve.py --reuse-capture captures\capture_XXXX.bin  :: 収録スキップ(検証用)
```

前提: `claude` CLI ログイン済み / `live.exe` ビルド済み(build.bat) /
Python に websockets・lightgbm。

## 1サイクルの流れ

```
adapter+live.exe ──10分収録──► captures/capture_*.bin (現在の市場)
        │
eval_ml_vNNN.exe ─► 整合性(corr・方向的中) + 発火シグナルのペーパー収支
        │            + 再学習用ラベル付き実データ特徴量 (data/ に蓄積)
        │
history/ ledger.csv・history.md に軌跡を追記
        │
claude -p (Opus 4.8) ─► versions/v(N+1)/train_config.json (修正方向つき)
        │
train_version.py ─► versions/v(N+1)/ml_model.hpp (新規生成、旧版は残る)
        │
build_eval.bat ─► 次サイクルは新バージョンを新しいライブデータで評価
```

## ファイル

| パス | 内容 |
|---|---|
| `../eval_ml.cpp` | capture照合エンジン(FeatureEngine+バージョン指定モデル) |
| `build_eval.bat vNNN` | バージョン指定で eval_ml_vNNN.exe をビルド |
| `train_version.py vNNN` | train_config.json から学習→ml_model.hpp生成 |
| `evolve.py` | オーケストレータ(収録→照合→記録→AI改善→再学習) |
| `prompts/improve.md` | Opus 4.8 への指示テンプレート |
| `versions/vNNN/` | 各版の ml_model.hpp + train_config.json + train_report.json |
| `data/live_features_*.csv` | 実データ特徴量+ラベル(サイクル毎に蓄積、再学習に使用) |
| `history/ledger.csv` | 機械可読の全サイクル記録(累積収支つき) |
| `history/history.md` | 軌跡の一覧表(照合結果・収支・AIの修正方向) |
| `history/decisions/` | AIの分析全文・期待効果・採用設定(サイクル毎JSON) |
| `state.json` | 現在バージョン・サイクル数・累積収支(再開用) |

## 停止と再開

- **利用制限**: Claude の利用制限を検知すると日本語メッセージを表示して自動停止。
  評価結果・履歴は保存済みで、制限解除後に同じコマンドで続きから再開。
- **Ctrl+C**: 収録プロセスを終了して停止。state.json から再開可能。

## 注意

- 収支は**ペーパートレード**(発火時テイカー往復、スプレッド+手数料 `--fee-bps`×2 控除)。
  実発注は一切しない。
- ラベルのスケール: 合成=500イベント先ティック / ライブ=5秒先JPY。混合学習の扱いは
  AIが `use_synthetic` / `use_live` / `live_weight` で判断する。
- 収録ポートは既定 9402(手動運用の 9401 と分離)。
- AIが動かせるのは学習設定のみ(ノブと範囲は `train_version.py` の CONFIG_SPEC で強制)。
  特徴量エンジンやC++本体は変更されない。
