BSIZE=16
PARAMS=-O3
all:
	nvcc -DBSIZE=${BSIZE} ${PARAMS} -Xcompiler -fopenmp main.cu -o prog

