/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct ThreadReduce_t {
    dace::cuda::Context *gpu_context;
};

DACE_EXPORTED void __dace_runkernel_assign_45_4_map_0_1_2(ThreadReduce_t *__state, double * __restrict__ gpu___return, int W);
DACE_EXPORTED void __dace_runkernel_ThreadReduce_47_0_0_0(ThreadReduce_t *__state, double * __restrict__ gpu___return, const double * __restrict__ gpu_inputs, int H, int W, const long long blockDim_x, int gridDim_x, long long loopNum, const long long rowNum);
void __program_ThreadReduce_internal(ThreadReduce_t *__state, double * __restrict__ __return, double * __restrict__ inputs, int H, int W, long long blockDim_x, long long gridDim_x, long long loopNum, long long rowNum)
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
                __dace_runkernel_assign_45_4_map_0_1_2(__state, gpu___return, W);
            } // End omp section
        } // End omp sections
        cudaStreamSynchronize(__state->gpu_context->streams[0]);
        cudaStreamSynchronize(__state->gpu_context->streams[1]);


    }
    {

        __dace_runkernel_ThreadReduce_47_0_0_0(__state, gpu___return, gpu_inputs, H, W, blockDim_x, gridDim_x, loopNum, rowNum);
        cudaMemcpyAsync(__return, gpu___return, W * sizeof(double), cudaMemcpyDeviceToHost, __state->gpu_context->streams[0]);
        cudaStreamSynchronize(__state->gpu_context->streams[0]);


    }
    cudaFree(gpu_inputs);
    cudaFree(gpu___return);
}

DACE_EXPORTED void __program_ThreadReduce(ThreadReduce_t *__state, double * __restrict__ __return, double * __restrict__ inputs, int H, int W, long long blockDim_x, long long gridDim_x, long long loopNum, long long rowNum)
{
    __program_ThreadReduce_internal(__state, __return, inputs, H, W, blockDim_x, gridDim_x, loopNum, rowNum);
}
DACE_EXPORTED int __dace_init_cuda(ThreadReduce_t *__state, int H, int W, long long blockDim_x, long long gridDim_x, long long loopNum, long long rowNum);
DACE_EXPORTED int __dace_exit_cuda(ThreadReduce_t *__state);

DACE_EXPORTED ThreadReduce_t *__dace_init_ThreadReduce(int H, int W, long long blockDim_x, long long gridDim_x, long long loopNum, long long rowNum)
{
    int __result = 0;
    ThreadReduce_t *__state = new ThreadReduce_t;


    __result |= __dace_init_cuda(__state, H, W, blockDim_x, gridDim_x, loopNum, rowNum);

    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_ThreadReduce(ThreadReduce_t *__state)
{
    __dace_exit_cuda(__state);
    delete __state;
}

