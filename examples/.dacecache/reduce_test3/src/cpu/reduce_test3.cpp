/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct reduce_test3_t {
    dace::cuda::Context *gpu_context;
};

DACE_EXPORTED void __dace_runkernel_assign_4_4_map_0_1_2(reduce_test3_t *__state, double * __restrict__ gpu___return);
DACE_EXPORTED void __dace_runkernel_reduce_test3_5_0_0_0(reduce_test3_t *__state, double * __restrict__ gpu___return, const double * __restrict__ gpu_inputs);
void __program_reduce_test3_internal(reduce_test3_t *__state, double * __restrict__ __return, double * __restrict__ inputs)
{
    double * gpu_inputs;
    cudaMalloc((void**)&gpu_inputs, 32768 * sizeof(double));
    double * gpu___return;
    cudaMalloc((void**)&gpu___return, 4096 * sizeof(double));

    {

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                cudaMemcpyAsync(gpu_inputs, inputs, 32768 * sizeof(double), cudaMemcpyHostToDevice, __state->gpu_context->streams[1]);
            } // End omp section
            #pragma omp section
            {
                __dace_runkernel_assign_4_4_map_0_1_2(__state, gpu___return);
            } // End omp section
        } // End omp sections
        cudaStreamSynchronize(__state->gpu_context->streams[0]);
        cudaStreamSynchronize(__state->gpu_context->streams[1]);


    }
    {

        __dace_runkernel_reduce_test3_5_0_0_0(__state, gpu___return, gpu_inputs);
        cudaMemcpyAsync(__return, gpu___return, 4096 * sizeof(double), cudaMemcpyDeviceToHost, __state->gpu_context->streams[0]);
        cudaStreamSynchronize(__state->gpu_context->streams[0]);


    }
    cudaFree(gpu_inputs);
    cudaFree(gpu___return);
}

DACE_EXPORTED void __program_reduce_test3(reduce_test3_t *__state, double * __restrict__ __return, double * __restrict__ inputs)
{
    __program_reduce_test3_internal(__state, __return, inputs);
}
DACE_EXPORTED int __dace_init_cuda(reduce_test3_t *__state);
DACE_EXPORTED int __dace_exit_cuda(reduce_test3_t *__state);

DACE_EXPORTED reduce_test3_t *__dace_init_reduce_test3()
{
    int __result = 0;
    reduce_test3_t *__state = new reduce_test3_t;


    __result |= __dace_init_cuda(__state);

    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_reduce_test3(reduce_test3_t *__state)
{
    __dace_exit_cuda(__state);
    delete __state;
}

