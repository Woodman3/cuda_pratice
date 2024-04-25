CC=nvcc
CFLAGS= -arch=compute_86 -code=sm_86 -g -G

all:demo.o gemm.o

run:gemm.o
	./$^

dbg:gemm.o
	cuda-gdb ./$^

demo.o:demo.cu
	$(CC) $(CFLAGS) $^ -o $@

gemm.o:gemm.cu
	$(CC) $(CFLAGS) $^ -o $@
	
