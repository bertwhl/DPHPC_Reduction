/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct redc_1d_t {
    dace::cuda::Context *gpu_context;
};

DACE_EXPORTED void __dace_runkernel_redc_1d_12_0_0_2(redc_1d_t *__state, double * __restrict__ gpu_a);
DACE_EXPORTED void __dace_runkernel_assign_21_4_gmap_0_0_11(redc_1d_t *__state, double * __restrict__ gpu___return, const double * __restrict__ __tmp0);
void __program_redc_1d_internal(redc_1d_t *__state, double * __restrict__ __return, double * __restrict__ a)
{
    double * __tmp0;
    cudaMalloc((void**)&__tmp0, 1 * sizeof(double));

    {
        double * gpu_a;
        cudaMalloc((void**)&gpu_a, 512 * sizeof(double));
        double * gpu___return;
        cudaMalloc((void**)&gpu___return, 1 * sizeof(double));

        cudaMemcpyAsync(gpu_a, a, 512 * sizeof(double), cudaMemcpyHostToDevice, __state->gpu_context->streams[0]);
        __dace_runkernel_redc_1d_12_0_0_2(__state, gpu_a);
        cudaMemcpyAsync(__tmp0, gpu_a, 1 * sizeof(double), cudaMemcpyDeviceToDevice, __state->gpu_context->streams[0]);
        cudaMemcpyAsync(a, gpu_a, 512 * sizeof(double), cudaMemcpyDeviceToHost, __state->gpu_context->streams[0]);

        cudaEventRecord(__state->gpu_context->events[0], __state->gpu_context->streams[0]);
        cudaStreamWaitEvent(__state->gpu_context->streams[1], __state->gpu_context->events[0], 0);

        __dace_runkernel_assign_21_4_gmap_0_0_11(__state, gpu___return, __tmp0);
        cudaMemcpyAsync(__return, gpu___return, 1 * sizeof(double), cudaMemcpyDeviceToHost, __state->gpu_context->streams[0]);
        cudaStreamSynchronize(__state->gpu_context->streams[0]);
        cudaStreamSynchronize(__state->gpu_context->streams[1]);

        cudaFree(gpu_a);
        cudaFree(gpu___return);

    }
    cudaFree(__tmp0);
}

DACE_EXPORTED void __program_redc_1d(redc_1d_t *__state, double * __restrict__ __return, double * __restrict__ a)
{
    __program_redc_1d_internal(__state, __return, a);
}
DACE_EXPORTED int __dace_init_cuda(redc_1d_t *__state);
DACE_EXPORTED int __dace_exit_cuda(redc_1d_t *__state);

DACE_EXPORTED redc_1d_t *__dace_init_redc_1d()
{
    int __result = 0;
    redc_1d_t *__state = new redc_1d_t;


    __result |= __dace_init_cuda(__state);

    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_redc_1d(redc_1d_t *__state)
{
    __dace_exit_cuda(__state);
    delete __state;
}

