mkdir log
rm ../models/NS/*
rm ../models/RD/*
python training.py --type NS --n 128 --Re 100 --GPU 0 > log/NS128-100.txt 2>&1 &
python training.py --type NS --n 128 --Re 200 --GPU 0 > log/NS128-200.txt 2>&1 &
python training.py --type NS --n 128 --Re 300 --GPU 0 > log/NS128-300.txt 2>&1 &
python training.py --type NS --n 128 --Re 400 --GPU 1 > log/NS128-400.txt 2>&1 &
python training.py --type NS --n 128 --Re 500 --GPU 1 > log/NS128-500.txt 2>&1 &