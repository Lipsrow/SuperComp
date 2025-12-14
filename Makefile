HOST_COMP ?= mpicxx
ARCH ?= sm_60
NVCC = nvcc

TARGET = gpu
SRC = cuda.cu

TARGET_DEBUG = gpu_debug
SRC_DEBUG = cuda_debug.cu 

all:
	$(NVCC) -arch=$(ARCH) -O2 -std=c++11 -Xcompiler "-fopenmp" -ccbin=$(HOST_COMP) $(SRC) -o $(TARGET) -lcudart -lstdc++

clean:
	rm -f $(TARGET) *.o
	rm *.err

test:
	mpisubmit.pl -p 1 -g 1 --stdout 256x1.out $(TARGET)
test2:
	mpisubmit.pl -p 2 -g 2 --stdout 256x2.out $(TARGET)
test_debug:
	mpisubmit.pl -p 1 -g 1 --stdout 512x1.debug $(TARGET_DEBUG)
test_debug2:
	mpisubmit.pl -p 2 -g 2 --stdout 512x2.debug $(TARGET_DEBUG)
debug:
	$(NVCC) -arch=$(ARCH) -O2 -Xcompiler "-fopenmp" -ccbin=$(HOST_COMP) $(SRC_DEBUG) -o $(TARGET_DEBUG) -lcudart -lstdc++ -std=c++11