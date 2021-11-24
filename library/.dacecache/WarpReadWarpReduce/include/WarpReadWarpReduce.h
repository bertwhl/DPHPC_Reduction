#include <dace/dace.h>
typedef void * WarpReadWarpReduceHandle_t;
extern "C" WarpReadWarpReduceHandle_t __dace_init_WarpReadWarpReduce(int H, int W, long long gridDim_x, long long gridDim_y, long long loopNum);
extern "C" void __dace_exit_WarpReadWarpReduce(WarpReadWarpReduceHandle_t handle);
extern "C" void __program_WarpReadWarpReduce(WarpReadWarpReduceHandle_t handle, double * __restrict__ __return, double * __restrict__ inputs, int H, int W, long long gridDim_x, long long gridDim_y, long long loopNum);
