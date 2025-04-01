CC=nvcc
CFLAGS= -g -G -lcublas -lcurand -std=c++11
TARGET=scan

all:$(TARGET).o

run:$(TARGET).o
	srun -p xhhgnormal -N 1 --gres=gpu:1 ./$^

dbg:$(TARGET).o
	cuda-gdb ./$^

$(TARGET).o:$(TARGET).cu
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -rf $(TARGET).o