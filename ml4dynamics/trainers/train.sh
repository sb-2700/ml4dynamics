mkdir log
#rm ../models/NS/*
#rm ../models/RD/*
python train.py --type NS --n 128 --Re 100 --GPU 0 &
python train.py --type NS --n 128 --Re 200 --GPU 0 > log/NS128-200.txt 2>&1 &
python train.py --type NS --n 128 --Re 300 --GPU 3 > log/NS128-300.txt 2>&1 &
python train.py --type NS --n 128 --Re 400 --GPU 1 > log/NS128-400.txt 2>&1 &
python train.py --type NS --n 128 --Re 500 --GPU 1 > log/NS128-500.txt 2>&1 &
python train.py --type NS --n 256 --Re 100 --GPU 2 > log/NS256-100.txt 2>&1 &
python train.py --type NS --n 256 --Re 200 --GPU 2 > log/NS256-200.txt 2>&1 &
python train.py --type NS --n 256 --Re 300 --GPU 2 > log/NS256-300.txt 2>&1 &
python train.py --type NS --n 256 --Re 400 --GPU 3 > log/NS256-400.txt 2>&1 &
python train.py --type NS --n 256 --Re 500 --GPU 3 > log/NS256-500.txt 2>&1 &

python train.py --type RD --n 128 --beta 0.2 --GPU 0 &
python train.py --type RD --n 128 --beta 0.4 --GPU 0 > log/RD128-4.txt 2>&1 &
python train.py --type RD --n 128 --beta 0.6 --GPU 0 > log/RD128-6.txt 2>&1 &
python train.py --type RD --n 128 --beta 0.8 --GPU 0 > log/RD128-8.txt 2>&1 &
python train.py --type RD --n 128 --beta 1.0 --GPU 0 > log/RD128-10.txt 2>&1 &
python train.py --type RD --n 64 --beta 0.2 --GPU 0 > log/RD64-2.txt 2>&1 &
python train.py --type RD --n 64 --beta 0.4 --GPU 0 > log/RD64-4.txt 2>&1 &
python train.py --type RD --n 64 --beta 0.6 --GPU 0 > log/RD64-6.txt 2>&1 &
python train.py --type RD --n 64 --beta 0.8 --GPU 0 > log/RD64-8.txt 2>&1 &
python train.py --type RD --n 64 --beta 1.0 --GPU 0 > log/RD64-10.txt 2>&1 &