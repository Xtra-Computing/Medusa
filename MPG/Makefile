CUDA_SDK_PATH :=/home/jianlong/cuda4.2/NVIDIA_GPU_Computing_SDK
CUDPP_PATH := /home/jianlong/Projects/cudpp2.0
OPT=-O3
CFLAGS=$(OPT) -c -w -fopenmp -lgomp
INCLUDES :=  -I$(CUDA_SDK_PATH)/C/common/inc -I/home/jianlong/Projects/cudpp2.0/include
CULIBS := -L$(CUDA_SDK_PATH)/C/lib -lcutil_x86_64 -L/lib -lcudpp
NVFLAGS=$(OPT) -arch=sm_20 $(INCLUDES)
LDFLAGS=$(OPT) $(CULIBS)
CC=g++
NVCC=/home/jianlong/cuda4.2/toolkit/bin/nvcc -Xcompiler -fopenmp
LD=/home/jianlong/cuda4.2/toolkit/bin/nvcc  -lgomp -lpthread
OBJS=main.o CPUFunctorHoster.o MessageDataType.o VertexDataType.o EdgeDataType.o\
../MedusaRT/Combiner.o  ../MedusaRT/GraphConverter.o  ../MedusaRT/GraphRenderingAPI.o    ../MedusaRT/OpenglSettings.o  ../MedusaRT/Utilities.o \
../MedusaRT/GraphGenerator.o  ../MedusaRT/GraphStorage.o         ../MedusaRT/PublicAPI.o \
 ../MedusaRT/CUDAOpenglInterop.o  ../MedusaRT/GraphReader.o    ../MedusaRT/MessageArrayManager.o  ../MedusaRT/SystemLibCPU.o ../MultipleGPU/PartitionManager.o ../MultipleGPU/MultiGraphStorage.o \
../MultipleGPU/WEQueue.o ../Tools/ReplicaNumberAnalysis.o ../Tools/RegularityAnalysis.o 

.PHONY: test
test: $(OBJS)
	$(LD) $(LDFLAGS) -o test $(OBJS)
%.o : %.cxx
	$(CC) $(CFLAGS) -c $< -o $*.o
.cpp.o :
	$(NVCC) $(NVFLAGS) -c $< -o $*.o
%.o : %.cu
	$(NVCC) $(NVFLAGS) -c $< -o $*.o 
clean:
	-rm test $(OBJS)

