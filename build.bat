@echo off
rem build.bat - build with g++ (MinGW-w64)
setlocal
cd /d "%~dp0"
set FLAGS=-O2 -std=c++23 -march=native -Wall -Wextra -static
rem live/replay: real-data build with a wider price window (BTC_JPY, 1 JPY tick)
set LIVE_WINDOW=-DHFT_PX_WINDOW=1048576
echo [1/4] bench.exe
g++ %FLAGS% bench.cpp -o bench.exe || exit /b 1
echo [2/4] backtest.exe
g++ %FLAGS% backtest.cpp -o backtest.exe || exit /b 1
echo [3/4] live.exe
g++ %FLAGS% %LIVE_WINDOW% live.cpp -o live.exe -lws2_32 || exit /b 1
echo [4/4] replay.exe
g++ %FLAGS% %LIVE_WINDOW% replay.cpp -o replay.exe || exit /b 1
echo done.
