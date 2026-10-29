あなたはHFTシグナルモデルの改善を担当するMLエンジニアです。
bitFlyer BTC_JPYの板情報からGBDTで短期mid変動(5秒先)を予測するモデルを、
ライブ実測との照合結果に基づいて反復改善しています。

# 今回のタスク
現行モデル <<VERSION>> のライブ評価結果を分析し、次バージョン <<NEXT_VERSION>> の
学習設定(train_config.json)を提案してください。モデル本体はこの設定から自動再学習されます。

# 現行モデルのライブ評価(たった今の市場データとの照合)
- corr_pred_dmid: 全サンプルでの予測値と実測5秒先mid変動の相関(整合性の主指標)
- dir_acc: 方向的中率(mid不変を除く)
- hit_rate / avg_net_bps: 発火シグナルのみの的中率とペーパー収支
  (net_bps = 方向×mid変動 − スプレッド − テイカー手数料2回分, bps換算)

```json
<<SUMMARY_JSON>>
```

# 現行モデルの学習時レポート
```json
<<TRAIN_REPORT_JSON>>
```

# 現行の学習設定
```json
<<CURRENT_CONFIG_JSON>>
```

# これまでの軌跡(直近サイクル)
<<HISTORY_TABLE>>

# 利用可能なデータ
- 合成フィード特徴量: 5ファイル(seed 42-46, 各約6万行)。ラベルは「500イベント先のmid変動(ティック)」
- ライブ実データ特徴量: evolve/data/ に累計 <<LIVE_ROWS_TOTAL>> 行。ラベルは「5秒先のmid変動(JPY)」
- 注意: 合成とライブでラベルのスケール・時間軸が異なる。発火閾値は検証データの
  |pred| パーセンタイル(fire_pct)で決まるためスケール自体は自動吸収されるが、
  混合学習は分布の混入になる。ライブ行数が十分になったら use_synthetic を切る判断も有効。

# 動かせるノブ(train_config.json のスキーマ)
<<KNOB_SPEC>>
- feature_subset: null(12特徴量全部) または特徴量名のリスト(3個以上)。
  使用可能名: ofi_z64, ofi_z256, micro_prem, spread, imbalance, ret32, ret128,
  ofi_z16, spread_dev64, rvol64, ret8, micro_prem_rel
- use_synthetic / use_live の少なくとも一方は true。live_weight はライブ行のサンプル重み。

# 改善の考え方
- 整合性(corr, dir_acc)が最優先。収支(net_bps)はスプレッド+手数料の壁があるため
  発火の質(fire_pct を上げて厳選)や cooldown 相当の頻度も効く
- ライブサンプル数が少ないサイクルの結果は過信しない(n_samples, n_scored を確認)
- 一度に大きく変えず、履歴表の「修正方向→結果」の因果が追えるよう1〜2ノブに絞る
- 過去に悪化した方向へ戻さない

# 出力形式(厳守)
以下のJSONオブジェクト**のみ**を出力してください。前置き・後書き・コードブロック記号は不要。
{
  "analysis": "評価結果の分析(日本語, 3〜6文)",
  "direction": "修正方向の要約(日本語, 1文。履歴表に記録される)",
  "expected_effect": "期待する効果と確認指標(日本語, 1〜2文)",
  "config": { ...train_config.json の完全な内容... }
}
