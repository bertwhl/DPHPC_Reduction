#include "device_launch_parameters.h"
#include <iostream>

int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for(int i=0;i<deviceCount;i++)
    {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        std::cout << "======================================================" << std::endl;  
        std::cout << "GPU Device No." << i << ": " << devProp.name << std::endl;
        std::cout << "Total Global Memory Size: " << devProp.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
        std::cout << "Number of Streaming Multiprocessor:" << devProp.multiProcessorCount << std::endl;
        std::cout << "Shared Memory per Block:" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "Maximum Number of Threads per Block:" << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "Number of Registers per Block: " << devProp.regsPerBlock << std::endl;
        std::cout << "Maximum Number of Threads per SM:" << devProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Maximum Number of Warps per SM:" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
        std::cout << "======================================================" << std::endl;     
        
    }
    return 0;
}
