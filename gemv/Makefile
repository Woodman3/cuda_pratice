CC=nvcc
CFLAGS= -arch=compute_86 -code=sm_86 -g -G -lcublas -lcurand
TARGET=gemv

all:$(TARGET).o

run:$(TARGET).o
	./$^

dbg:$(TARGET).o
	cuda-gdb ./$^

$(TARGET).o:$(TARGET).cu
	$(CC) $(CFLAGS) $^ -o $@
	
