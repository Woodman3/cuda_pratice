CC=nvcc
CFLAGS= -arch=compute_86 -code=sm_86 -g -G

all:gemm.o

run:gemm.o
	./$^

dbg:gemm.o
	cuda-gdb ./$^

gemm.o:gemm.cu
	$(CC) $(CFLAGS) $^ -o $@
	
