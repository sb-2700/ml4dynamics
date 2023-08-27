###################################################
#                   finished                      #
###################################################
python3 generate_NSdata.py  --Re 100 --nx 128 &
python3 generate_NSdata.py  --Re 200 --nx 128 &
python3 generate_NSdata.py  --Re 300 --nx 128 &
python3 generate_NSdata.py  --Re 400 --nx 128 & 
python3 generate_NSdata.py  --Re 500 --nx 128 & 
#python generate_RDdata.py  --beta 1.0 & 
#python generate_RDdata.py  --beta 0.8 &
#python generate_RDdata.py  --beta 0.6 &
#python generate_RDdata.py  --beta 0.4 & 
#python generate_RDdata.py  --beta 0.2 &
#python generate_NSdata.py  --Re 100 --nx 256 &
python3 generate_NSdata.py  --Re 200 --nx 256 &
python3 generate_NSdata.py  --Re 300 --nx 256 &
#python generate_NSdata.py  --Re 400 --nx 256 &
#python generate_NSdata.py  --Re 500 --nx 256 &