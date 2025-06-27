@echo off
TITLE Meta-IFD Runner

echo ==================================================
echo       Starting Meta-IFD Execution...
echo ==================================================
echo.

REM 進入 Meta-IFD 資料夾
cd Meta-IFD

REM 執行一系列帶有不同參數的 Python 腳本
python run.py  --dataset Phish --hidden 4 --lr 0.01 --loss_train 0.5 --concat 1 --batch_size 128 --epochs 1 
python run.py  --dataset Phish --hidden 8 --lr 0.01 --loss_train 0.5 --concat 1 --batch_size 128 --epochs 1 
python run.py  --dataset Phish --hidden 4 --lr 0.02 --loss_train 0.5 --concat 1 --batch_size 128 --epochs 1 
python run.py  --dataset Phish --hidden 4 --lr 0.01 --loss_train 0.5 --concat 2 --batch_size 128 --epochs 1 
python run.py  --dataset Phish --hidden 4 --lr 0.01 --loss_train 0.5 --concat 2 --batch_size 256 --epochs 1 

python run.py  --dataset Phish --hidden 4 --lr 0.01 --loss_train 0.5 --concat 1 --batch_size 128 --epochs 10 
python run.py  --dataset Phish --hidden 8 --lr 0.01 --loss_train 0.5 --concat 1 --batch_size 128 --epochs 10 
python run.py  --dataset Phish --hidden 4 --lr 0.02 --loss_train 0.5 --concat 1 --batch_size 128 --epochs 10 
python run.py  --dataset Phish --hidden 4 --lr 0.01 --loss_train 0.5 --concat 2 --batch_size 128 --epochs 10
python run.py  --dataset Phish --hidden 4 --lr 0.01 --loss_train 0.5 --concat 2 --batch_size 256 --epochs 10

python run.py  --dataset Phish --hidden 4 --lr 0.01 --loss_train 0.5 --concat 1 --batch_size 128 --epochs 5

REM 返回上一層資料夾
cd ..

echo.
echo ==================================================
echo       Meta-IFD Execution Complete!
echo ==================================================
echo.
pause