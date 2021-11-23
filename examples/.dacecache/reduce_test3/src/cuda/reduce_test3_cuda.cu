
#include <cuda_runtime.h>
#include <dace/dace.h>


struct reduce_test3_t {
    dace::cuda::Context *gpu_context;
};



DACE_EXPORTED int __dace_init_cuda(reduce_test3_t *__state);
DACE_EXPORTED void __dace_exit_cuda(reduce_test3_t *__state);

DACE_DFI void reduce_test3_5_4_6_8_0_0_6(const double * __tmp_9_25_r_in_from_2_0, double * __tmp_10_12_w) {
    double value;

    {

        {
            double __out;

            ///////////////////
            // Tasklet code (_convert_to_float64_)
            __out = dace::float64(0);
            ///////////////////

            value = __out;
        }

    }
    {

        {
            for (auto i = 0; i < 4; i += 1) {
                {
                    double __inp = __tmp_9_25_r_in_from_2_0[(8192 * i)];
                    double __out;

                    ///////////////////
                    // Tasklet code (assign_9_16)
                    __out = __inp;
                    ///////////////////

                    dace::wcr_fixed<dace::ReductionType::Sum, double>::reduce(&value, __out);
                }
            }
        }
        {
            double __inp = value;
            double __out;

            ///////////////////
            // Tasklet code (assign_10_12)
            __out = __inp;
            ///////////////////

            dace::wcr_fixed<dace::ReductionType::Sum, double>::reduce_atomic(__tmp_10_12_w, __out);
        }

    }
    
}



int __dace_init_cuda(reduce_test3_t *__state) {
    int count;

    // Check that we are able to run cuda code
    if (cudaGetDeviceCount(&count) != cudaSuccess)
    {
        printf("ERROR: GPU drivers are not configured or cuda-capable device "
               "not found\n");
        return 1;
    }
    if (count == 0)
    {
        printf("ERROR: No cuda-capable devices found\n");
        return 2;
    }

    // Initialize cuda before we run the application
    float *dev_X;
    cudaMalloc((void **) &dev_X, 1);
    cudaFree(dev_X);

    __state->gpu_context = new dace::cuda::Context(2, 1);

    // Create cuda streams and events
    for(int i = 0; i < 2; ++i) {
        cudaStreamCreateWithFlags(&__state->gpu_context->streams[i], cudaStreamNonBlocking);
    }
    for(int i = 0; i < 1; ++i) {
        cudaEventCreateWithFlags(&__state->gpu_context->events[i], cudaEventDisableTiming);
    }

    

    return 0;
}

void __dace_exit_cuda(reduce_test3_t *__state) {
    

    // Destroy cuda streams and events
    for(int i = 0; i < 2; ++i) {
        cudaStreamDestroy(__state->gpu_context->streams[i]);
    }
    for(int i = 0; i < 1; ++i) {
        cudaEventDestroy(__state->gpu_context->events[i]);
    }

    delete __state->gpu_context;
}

__global__ void assign_4_4_map_0_1_2(double * __restrict__ gpu___return) {
    {
        int __i0 = (blockIdx.x * 32 + threadIdx.x);
        if (__i0 < 4096) {
            {
                double __out;

                ///////////////////
                // Tasklet code (assign_4_4)
                __out = 0;
                ///////////////////

                gpu___return[__i0] = __out;
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_assign_4_4_map_0_1_2(reduce_test3_t *__state, double * __restrict__ gpu___return);
void __dace_runkernel_assign_4_4_map_0_1_2(reduce_test3_t *__state, double * __restrict__ gpu___return)
{

    void  *assign_4_4_map_0_1_2_args[] = { (void *)&gpu___return };
    cudaLaunchKernel((void*)assign_4_4_map_0_1_2, dim3(int_ceil(int_ceil(4096, 1), 32), int_ceil(1, 1), int_ceil(1, 1)), dim3(32, 1, 1), assign_4_4_map_0_1_2_args, 0, __state->gpu_context->streams[0]);
}
__global__ void reduce_test3_5_0_0_0(double * __restrict__ gpu___return, const double * __restrict__ gpu_inputs) {
    {
        {
            int blockIdx_x = blockIdx.x;
            int blockIdx_y = blockIdx.y;
            {
                {
                    {
                        int threadIdx_x = threadIdx.x;
                        int threadIdx_y = threadIdx.y;
                        if (threadIdx_x < 32) {
                            if (threadIdx_y < 32) {
                                reduce_test3_5_4_6_8_0_0_6(&gpu_inputs[((((1024 * blockIdx_x) + (4096 * blockIdx_y)) + threadIdx_x) + (32 * threadIdx_y))], &gpu___return[(((1024 * blockIdx_x) + threadIdx_x) + (32 * threadIdx_y))]);
                            }
                        }
                    }
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_reduce_test3_5_0_0_0(reduce_test3_t *__state, double * __restrict__ gpu___return, const double * __restrict__ gpu_inputs);
void __dace_runkernel_reduce_test3_5_0_0_0(reduce_test3_t *__state, double * __restrict__ gpu___return, const double * __restrict__ gpu_inputs)
{

    void  *reduce_test3_5_0_0_0_args[] = { (void *)&gpu___return, (void *)&gpu_inputs };
    cudaLaunchKernel((void*)reduce_test3_5_0_0_0, dim3(int_ceil(4, 1), int_ceil(2, 1), 1), dim3(32, 32, 1), reduce_test3_5_0_0_0_args, 0, __state->gpu_context->streams[0]);
}

