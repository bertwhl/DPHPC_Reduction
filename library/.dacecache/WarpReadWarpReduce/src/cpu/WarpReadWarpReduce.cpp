/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct WarpReadWarpReduce_t {
    dace::cuda::Context *gpu_context;
};

DACE_EXPORTED void __dace_runkernel_assign_13_4_map_0_1_2(WarpReadWarpReduce_t *__state, double * __restrict__ gpu___return, int W);
DACE_EXPORTED void __dace_runkernel_WarpReadWarpReduce_15_0_0_0(WarpReadWarpReduce_t *__state, double * __restrict__ gpu___return, const double * __restrict__ gpu_inputs, int H, int W, int gridDim_x, const int gridDim_y, long long loopNum);
void __program_WarpReadWarpReduce_internal(WarpReadWarpReduce_t *__state, double * __restrict__ __return, double * __restrict__ inputs, int H, int W, long long gridDim_x, long long gridDim_y, long long loopNum)
{
    double * gpu_inputs;
    cudaMalloc((void**)&gpu_inputs, (H * W) * sizeof(double));
    double * gpu___return;
    cudaMalloc((void**)&gpu___return, W * sizeof(double));

    {

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                cudaMemcpyAsync(gpu_inputs, inputs, (H * W) * sizeof(double), cudaMemcpyHostToDevice, __state->gpu_context->streams[1]);
            } // End omp section
            #pragma omp section
            {
                __dace_runkernel_assign_13_4_map_0_1_2(__state, gpu___return, W);
            } // End omp section
        } // End omp sections
        cudaStreamSynchronize(__state->gpu_context->streams[0]);
        cudaStreamSynchronize(__state->gpu_context->streams[1]);


    }
    {

        __dace_runkernel_WarpReadWarpReduce_15_0_0_0(__state, gpu___return, gpu_inputs, H, W, gridDim_x, gridDim_y, loopNum);
        cudaMemcpyAsync(__return, gpu___return, W * sizeof(double), cudaMemcpyDeviceToHost, __state->gpu_context->streams[0]);
        cudaStreamSynchronize(__state->gpu_context->streams[0]);


    }
    cudaFree(gpu_inputs);
    cudaFree(gpu___return);
}

DACE_EXPORTED void __program_WarpReadWarpReduce(WarpReadWarpReduce_t *__state, double * __restrict__ __return, double * __restrict__ inputs, int H, int W, long long gridDim_x, long long gridDim_y, long long loopNum)
{
    __program_WarpReadWarpReduce_internal(__state, __return, inputs, H, W, gridDim_x, gridDim_y, loopNum);
}
DACE_EXPORTED int __dace_init_cuda(WarpReadWarpReduce_t *__state, int H, int W, long long gridDim_x, long long gridDim_y, long long loopNum);
DACE_EXPORTED int __dace_exit_cuda(WarpReadWarpReduce_t *__state);

DACE_EXPORTED WarpReadWarpReduce_t *__dace_init_WarpReadWarpReduce(int H, int W, long long gridDim_x, long long gridDim_y, long long loopNum)
{
    int __result = 0;
    WarpReadWarpReduce_t *__state = new WarpReadWarpReduce_t;


    __result |= __dace_init_cuda(__state, H, W, gridDim_x, gridDim_y, loopNum);

    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_WarpReadWarpReduce(WarpReadWarpReduce_t *__state)
{
    __dace_exit_cuda(__state);
    delete __state;
}

