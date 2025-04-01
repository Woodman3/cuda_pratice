CC=nvcc
CFLAGS= -arch=compute_86 -code=sm_86 -g -G

all:demo.o

run:gemm.o
	./$^

dbg:gemm.o
	cuda-gdb ./$^

demo.o:demo.cu
	$(CC) $(CFLAGS) $^ -o $@

	
