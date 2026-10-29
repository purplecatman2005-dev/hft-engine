@echo off
rem build_ml.bat - GBDT->C++ pipeline: dump features, train+codegen, build backtest, verify
rem train=seed 42-45 (first 80%) / threshold=seed 46 (last 20%). Hyperparams are the best of the 2026-07 sweep.
setlocal
cd /d "%~dp0"
set FLAGS=-O2 -std=c++23 -march=native -Wall -Wextra -static
echo [1/5] ml_dump_features.exe
g++ %FLAGS% ml_dump_features.cpp -o ml_dump_features.exe || exit /b 1
rem keep to 1M events per seed: the synthetic feed's book saturates and mid freezes beyond ~1.4M
echo [2/5] dump features (1M events x seeds 42-46)
.\ml_dump_features.exe 1000000 42 ml_features.csv || exit /b 1
for %%s in (43 44 45 46) do (
    .\ml_dump_features.exe 1000000 %%s ml_features_s%%s.csv || exit /b 1
)
echo [3/5] train LightGBM + generate ml_model.hpp
python ml_train_export.py --features "ml_features.csv,ml_features_s43.csv,ml_features_s44.csv,ml_features_s45.csv,ml_features_s46.csv" --trees 400 --leaves 15 --lr 0.03 --fire-pct 97 || exit /b 1
echo [4/5] backtest_ml.exe
g++ %FLAGS% backtest_ml.cpp -o backtest_ml.exe || exit /b 1
echo [5/5] verify python vs c++ predictions
.\backtest_ml.exe --verify ml_verify.csv || exit /b 1
echo done.
