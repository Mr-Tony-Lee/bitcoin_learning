echo "=================================================="
echo "      Starting Meta-IFD Execution..."
echo "=================================================="
echo

cd Meta-IFD
python3 run.py  --dataset Phish --hidden 4 --lr 0.01 --loss_train 0.5 --concat 1 --batch_size 128 --epochs 1 
python3 run.py  --dataset Phish --hidden 8 --lr 0.01 --loss_train 0.5 --concat 1 --batch_size 128 --epochs 1 
python3 run.py  --dataset Phish --hidden 4 --lr 0.02 --loss_train 0.5 --concat 1 --batch_size 128 --epochs 1 
python3 run.py  --dataset Phish --hidden 4 --lr 0.01 --loss_train 0.5 --concat 2 --batch_size 128 --epochs 1 
python3 run.py  --dataset Phish --hidden 4 --lr 0.01 --loss_train 0.5 --concat 2 --batch_size 256 --epochs 1 

python3 run.py  --dataset Phish --hidden 4 --lr 0.01 --loss_train 0.5 --concat 1 --batch_size 128 --epochs 10 
python3 run.py  --dataset Phish --hidden 8 --lr 0.01 --loss_train 0.5 --concat 1 --batch_size 128 --epochs 10 
python3 run.py  --dataset Phish --hidden 4 --lr 0.02 --loss_train 0.5 --concat 1 --batch_size 128 --epochs 10 
python3 run.py  --dataset Phish --hidden 4 --lr 0.01 --loss_train 0.5 --concat 2 --batch_size 128 --epochs 10
python3 run.py  --dataset Phish --hidden 4 --lr 0.01 --loss_train 0.5 --concat 2 --batch_size 256 --epochs 10

python3 run.py  --dataset Phish --hidden 4 --lr 0.01 --loss_train 0.5 --concat 1 --batch_size 128 --epochs 5

cd ..
echo "=================================================="
echo "      Meta-IFD Execution Complete!"
echo "=================================================="
echo