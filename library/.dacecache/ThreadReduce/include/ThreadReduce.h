#include <dace/dace.h>
typedef void * ThreadReduceHandle_t;
extern "C" ThreadReduceHandle_t __dace_init_ThreadReduce(int H, int W, long long blockDim_x, long long gridDim_x, long long loopNum, long long rowNum);
extern "C" void __dace_exit_ThreadReduce(ThreadReduceHandle_t handle);
extern "C" void __program_ThreadReduce(ThreadReduceHandle_t handle, double * __restrict__ __return, double * __restrict__ inputs, int H, int W, long long blockDim_x, long long gridDim_x, long long loopNum, long long rowNum);
