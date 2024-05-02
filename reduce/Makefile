CC=nvcc
CFLAGS= -arch=compute_86 -code=sm_86 -g -G
TARGET=reduce

all:$(TARGET).o

run:$(TARGET).o
	./$^

dbg:$(TARGET).o
	cuda-gdb ./$^

$(TARGET).o:$(TARGET).cu
	$(CC) $(CFLAGS) $^ -o $@
	
