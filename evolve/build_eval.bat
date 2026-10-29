@echo off
rem build_eval.bat <version> - build eval_ml.exe against a versioned model header
rem usage: build_eval.bat v001  ->  evolve\build\eval_ml_v001.exe
setlocal
cd /d "%~dp0.."
set V=%1
if "%V%"=="" (
    echo usage: build_eval.bat vNNN
    exit /b 1
)
if not exist "evolve\versions\%V%\ml_model.hpp" (
    echo model not found: evolve\versions\%V%\ml_model.hpp
    exit /b 1
)
if not exist evolve\build mkdir evolve\build
copy /Y "evolve\versions\%V%\ml_model.hpp" "evolve\build\model_current.hpp" >nul
g++ -O2 -std=c++23 -march=native -Wall -Wextra -static ^
    -DHFT_PX_WINDOW=1048576 -DML_MODEL_HEADER=\"evolve/build/model_current.hpp\" ^
    eval_ml.cpp -o "evolve\build\eval_ml_%V%.exe" || exit /b 1
echo built: evolve\build\eval_ml_%V%.exe
