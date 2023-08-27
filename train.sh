#mkdir log
#rm ../models/NS/*
#rm ../models/RD/*
python train.py --type NS --n 128 --Re 100 --GPU 0 &
python train.py --type NS --n 128 --Re 200 --GPU 0 > log/NS128-200.txt 2>&1 &
python train.py --type NS --n 128 --Re 300 --GPU 3 > log/NS128-300.txt 2>&1 &
python train.py --type NS --n 128 --Re 400 --GPU 1 > log/NS128-400.txt 2>&1 &
python train.py --type NS --n 128 --Re 500 --GPU 1 > log/NS128-500.txt 2>&1 &
#python train.py --type NS --n 256 --Re 100 --GPU 2 > log/NS256-100.txt 2>&1 &
python train.py --type NS --n 256 --Re 200 --GPU 2 > log/NS256-200.txt 2>&1 &
python train.py --type NS --n 256 --Re 300 --GPU 2 > log/NS256-300.txt 2>&1 &
#python train.py --type NS --n 256 --Re 400 --GPU 3 > log/NS256-400.txt 2>&1 &
#python train.py --type NS --n 256 --Re 500 --GPU 3 > log/NS256-500.txt 2>&1 &
