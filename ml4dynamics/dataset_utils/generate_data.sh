###################################################
#                   finished                      #
###################################################
python generate_NSdata.py  --Re 100 --nx 128 &
python generate_NSdata.py  --Re 200 --nx 128 &
python generate_NSdata.py  --Re 300 --nx 128 &
python generate_NSdata.py  --Re 400 --nx 128 & 
python generate_NSdata.py  --Re 500 --nx 128 & 
python generate_RDdata.py  --gamma 0.05 & 
python generate_RDdata.py  --gamma 0.10 & 
python generate_RDdata.py  --gamma 0.15 & 
python generate_RDdata.py  --gamma 0.20 & 
python generate_RDdata.py  --gamma 0.25 & 
python generate_NSdata.py  --Re 100 --nx 256 &
python generate_NSdata.py  --Re 200 --nx 256 &
python generate_NSdata.py  --Re 300 --nx 256 &
python generate_NSdata.py  --Re 400 --nx 256 &
python generate_NSdata.py  --Re 500 --nx 256 &