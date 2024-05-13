CC=nvcc
CFLAGS= -arch=compute_86 -code=sm_86 -g -G -lcublas -lcurand
TARGET=dot

all:$(TARGET).o

run:$(TARGET).o
	./$^

dbg:$(TARGET).o
	cuda-gdb ./$^

$(TARGET).o:$(TARGET).cu
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -rf $(TARGET).o
	
