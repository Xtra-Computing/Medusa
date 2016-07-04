cmake_minimum_required(VERSION 2.8)
project(MBFS)

set(CUDA_SDK_UTIL_PTH /home/cheyulin/NVIDIA_CUDA-7.5_Samples/common)
set(CUDPP_PATH /home/cheyulin/Tools/cudpp-2.2-build) #Static Link

set(MBFS_APP_FILES Configuration.h CPUFunctorHoster.h CPUFunctorHoster.cpp DeviceDataStructure.h 
EdgeDataType.h EdgeDataType.cu InitMessage.h main.cu MessageDataType.h MessageDataType.cu MsgCombiner.h 
UserOperators.h VertexDataType.h VertexDataType.cu Vertex.h Edge.h Message.h)

set(MEDUSA_FILES ../MedusaRT/APIKernelWrappers.h ../MedusaRT/Combiner.cu ../MedusaRT/Combiner.h 
../MedusaRT/CUDAOpenglInterop.h ../MedusaRT/CUDAOpenglInterop.cu
../MedusaRT/GraphConverter.cu ../MedusaRT/GraphConverter.h ../MedusaRT/GraphGenerator.cu 
../MedusaRT/GraphGenerator.h ../MedusaRT/GraphReader.cpp ../MedusaRT/GraphReader.h
../MedusaRT/GraphRenderingAPI.cu ../MedusaRT/GraphRenderingAPI.h ../MedusaRT/GraphStorage.cu ../MedusaRT/GraphStorage.h 
../MedusaRT/HMMSMA.h ../MedusaRT/HMMUtilGPU.h ../MedusaRT/MessageArrayManager.cu 
../MedusaRT/MessageArrayManager.h ../MedusaRT/OpenglSettings.cu ../MedusaRT/OpenglSettings.h ../MedusaRT/PublicAPI.cu 
../MedusaRT/PublicAPI.h ../MedusaRT/SystemLibCPU.cu ../MedusaRT/SystemLibCPU.h 
../MedusaRT/SystemLibGPU.h ../MedusaRT/Utilities.cu ../MedusaRT/Utilities.h)

set(MGPU_FILES ../MultipleGPU/Gather.cu ../MultipleGPU/GPUDef.h ../MultipleGPU/MultiAPIHoster.h 
../MultipleGPU/MultiGraphStorage.cu ../MultipleGPU/MultiGraphStorage.h ../MultipleGPU/MultiPublicAPI.h
 ../MultipleGPU/MultiUtilities.h ../MultipleGPU/PartitionManager.cu ../MultipleGPU/PartitionManager.h
  ../MultipleGPU/WEQueue.cpp ../MultipleGPU/WEQueue.h)

include_directories(${CUDA_SDK_UTIL_PTH}/inc ${CUDPP_PATH}/include)
link_directories(${CUDA_SDK_UTIL_PTH}/lib ${CUDPP_PATH}/lib)
    
find_package(CUDA REQUIRED)

find_package(Threads)
    message("Attention:"${CMAKE_THREAD_LIBS_INIT})
    # Add To Non-NVCC Compiler Flag
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler ${CMAKE_THREAD_LIBS_INIT}")

find_package(OpenMP REQUIRED)
    # Add To Non-NVCC Compiler Flag
    message("Attention:"${OpenMP_CXX_FLAGS})
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler ${OpenMP_CXX_FLAGS}")
    
# CC Version
set(CUDA_DEVICE_VERSION "20" CACHE STRING "CUDA Device Version")

list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fopenmp")

 # Macro for Compute Capability
 macro(set_compute_capability cc)
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${cc},code=sm_${cc}")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${cc},code=compute_${cc}")
 endmacro(set_compute_capability)
 
set_compute_capability(${CUDA_DEVICE_VERSION})

cuda_add_executable(MBFS ${MBFS_APP_FILES} ${MEDUSA_FILES} ${MGPU_FILES} ../Compatibility/Compatability.h)

    