
#include <cuda_runtime.h>
#include <dace/dace.h>


struct WarpReadWarpReduce_t {
    dace::cuda::Context *gpu_context;
};



DACE_EXPORTED int __dace_init_cuda(WarpReadWarpReduce_t *__state, int H, int W, long long gridDim_x, long long gridDim_y, long long loopNum);
DACE_EXPORTED void __dace_exit_cuda(WarpReadWarpReduce_t *__state);

DACE_DFI void WarpReadWarpReduce_15_4_17_8_24_12_2_1_2(const double * __tmp_27_29_r, long long& __tmp_27_36_r, long long& __tmp_27_44_r, long long& __tmp_28_26_r, double& __tmp_27_20_w, long long& __tmp_28_16_w, int H, int W, long long colIdx, long long rowIdx) {
    long long __sym___tmp_27_36_r;
    long long __sym___tmp_27_44_r;

    if (((rowIdx < H) && (colIdx < W))) {
        __sym___tmp_27_36_r = __tmp_27_36_r;
        __sym___tmp_27_44_r = __tmp_27_44_r;
        {

            {
                double __inp = __tmp_27_29_r[((W * __sym___tmp_27_36_r) + __sym___tmp_27_44_r)];
                double __out;

                ///////////////////
                // Tasklet code (assign_27_20)
                __out = __inp;
                ///////////////////

                dace::wcr_fixed<dace::ReductionType::Sum, double>::reduce(&__tmp_27_20_w, __out);
            }

        }
        goto __state_3_assign_28_16;

    }
    __state_3_assign_28_16:;
    {

        {
            long long __inp = __tmp_28_26_r;
            long long __out;

            ///////////////////
            // Tasklet code (assign_28_16)
            __out = __inp;
            ///////////////////

            dace::wcr_fixed<dace::ReductionType::Sum, long long>::reduce(&__tmp_28_16_w, __out);
        }

    }
    
}

DACE_DFI void WarpReadWarpReduce_15_4_17_8_0_0_7(const long long& __tmp_23_23_r, const double * __tmp_27_29_r_in_from_9_0, double& __tmp_30_12_w, int H, int W, long long blockIdx_x, long long blockIdx_y, int loopNum, long long threadIdx_x, long long threadIdx_y) {
    double value;
    long long rowIdx;
    long long colIdx;
    long long delta;

    {

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                {
                    double __out;

                    ///////////////////
                    // Tasklet code (_convert_to_float64_)
                    __out = dace::float64(0);
                    ///////////////////

                    value = __out;
                }
            } // End omp section
            #pragma omp section
            {
                {
                    long long __out;

                    ///////////////////
                    // Tasklet code (assign_21_12)
                    __out = ((32 * blockIdx_y) + threadIdx_y);
                    ///////////////////

                    rowIdx = __out;
                }
            } // End omp section
            #pragma omp section
            {
                {
                    long long __out;

                    ///////////////////
                    // Tasklet code (assign_22_12)
                    __out = ((32 * blockIdx_x) + threadIdx_x);
                    ///////////////////

                    colIdx = __out;
                }
            } // End omp section
            #pragma omp section
            {
                {
                    long long __in2 = __tmp_23_23_r;
                    long long __out;

                    ///////////////////
                    // Tasklet code (_Mult_)
                    __out = (32 * __in2);
                    ///////////////////

                    delta = __out;
                }
            } // End omp section
        } // End omp sections

    }
    {

        {
            for (auto loopIdx = 0; loopIdx < loopNum; loopIdx += 1) {
                WarpReadWarpReduce_15_4_17_8_24_12_2_1_2(&__tmp_27_29_r_in_from_9_0[0], rowIdx, colIdx, delta, value, rowIdx, H, W, colIdx, rowIdx);
            }
        }
        {
            double __inp = value;
            double __out;

            ///////////////////
            // Tasklet code (assign_30_12)
            __out = __inp;
            ///////////////////

            __tmp_30_12_w = __out;
        }

    }
    
}

DACE_DFI void WarpReadWarpReduce_15_4_32_8_0_0_11(double& __tmp_34_37_r, double * __tmp_38_16_w, int W, long long blockIdx_x, long long threadIdx_x, long long threadIdx_y) {
    double reduced[1]  DACE_ALIGN(64);
    long long colIdx;
    long long __sym___tmp5;

    {

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                {
                    double __a = __tmp_34_37_r;
                    double __out;

                    ///////////////////

                    __out = dace::warpReduce<dace::ReductionType::Sum, double>::reduce(__a);

                    ///////////////////

                    reduced[0] = __out;
                }
            } // End omp section
            #pragma omp section
            {
                {
                    long long __out;

                    ///////////////////
                    // Tasklet code (assign_36_12)
                    __out = ((32 * blockIdx_x) + threadIdx_y);
                    ///////////////////

                    colIdx = __out;
                }
            } // End omp section
        } // End omp sections

    }
    if (((threadIdx_x == 0) && (colIdx < W))) {
        __sym___tmp5 = colIdx;
        {

            {
                double __inp = reduced[0];
                double __out;

                ///////////////////
                // Tasklet code (assign_38_16)
                __out = __inp;
                ///////////////////

                dace::wcr_fixed<dace::ReductionType::Sum, double>::reduce_atomic(__tmp_38_16_w + __sym___tmp5, __out);
            }

        }
        goto __state_4_endif_37;

    }
    __state_4_endif_37:;
    
}



int __dace_init_cuda(WarpReadWarpReduce_t *__state, int H, int W, long long gridDim_x, long long gridDim_y, long long loopNum) {
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

void __dace_exit_cuda(WarpReadWarpReduce_t *__state) {
    

    // Destroy cuda streams and events
    for(int i = 0; i < 2; ++i) {
        cudaStreamDestroy(__state->gpu_context->streams[i]);
    }
    for(int i = 0; i < 1; ++i) {
        cudaEventDestroy(__state->gpu_context->events[i]);
    }

    delete __state->gpu_context;
}

__global__ void assign_13_4_map_0_1_2(double * __restrict__ gpu___return, int W) {
    {
        int __i0 = (blockIdx.x * 32 + threadIdx.x);
        if (__i0 < W) {
            {
                double __out;

                ///////////////////
                // Tasklet code (assign_13_4)
                __out = 0;
                ///////////////////

                gpu___return[__i0] = __out;
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_assign_13_4_map_0_1_2(WarpReadWarpReduce_t *__state, double * __restrict__ gpu___return, int W);
void __dace_runkernel_assign_13_4_map_0_1_2(WarpReadWarpReduce_t *__state, double * __restrict__ gpu___return, int W)
{

    void  *assign_13_4_map_0_1_2_args[] = { (void *)&gpu___return, (void *)&W };
    cudaLaunchKernel((void*)assign_13_4_map_0_1_2, dim3(int_ceil(int_ceil(W, 1), 32), int_ceil(1, 1), int_ceil(1, 1)), dim3(32, 1, 1), assign_13_4_map_0_1_2_args, 0, __state->gpu_context->streams[0]);
}
__global__ void WarpReadWarpReduce_15_0_0_0(double * __restrict__ gpu___return, const double * __restrict__ gpu_inputs, int H, int W, int gridDim_x, const int gridDim_y, long long loopNum) {
    {
        {
            int blockIdx_x = blockIdx.x;
            int blockIdx_y = blockIdx.y;
            __shared__ double shared[1024];
            {
                {
                    {
                        int threadIdx_x = threadIdx.x;
                        int threadIdx_y = threadIdx.y;
                        if (threadIdx_x < 32) {
                            if (threadIdx_y < 32) {
                                WarpReadWarpReduce_15_4_17_8_0_0_7(gridDim_y, &gpu_inputs[0], shared[((32 * threadIdx_x) + threadIdx_y)], H, W, blockIdx_x, blockIdx_y, loopNum, threadIdx_x, threadIdx_y);
                            }
                        }
                    }
                }
            }
            __syncthreads();
            {
                {
                    {
                        int threadIdx_x = threadIdx.x;
                        int threadIdx_y = threadIdx.y;
                        if (threadIdx_x < 32) {
                            if (threadIdx_y < 32) {
                                WarpReadWarpReduce_15_4_32_8_0_0_11(shared[(threadIdx_x + (32 * threadIdx_y))], &gpu___return[0], W, blockIdx_x, threadIdx_x, threadIdx_y);
                            }
                        }
                    }
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_WarpReadWarpReduce_15_0_0_0(WarpReadWarpReduce_t *__state, double * __restrict__ gpu___return, const double * __restrict__ gpu_inputs, int H, int W, int gridDim_x, const int gridDim_y, long long loopNum);
void __dace_runkernel_WarpReadWarpReduce_15_0_0_0(WarpReadWarpReduce_t *__state, double * __restrict__ gpu___return, const double * __restrict__ gpu_inputs, int H, int W, int gridDim_x, const int gridDim_y, long long loopNum)
{

    void  *WarpReadWarpReduce_15_0_0_0_args[] = { (void *)&gpu___return, (void *)&gpu_inputs, (void *)&H, (void *)&W, (void *)&gridDim_x, (void *)&gridDim_y, (void *)&loopNum };
    cudaLaunchKernel((void*)WarpReadWarpReduce_15_0_0_0, dim3(int_ceil(gridDim_x, 1), int_ceil(gridDim_y, 1), 1), dim3(32, 32, 1), WarpReadWarpReduce_15_0_0_0_args, 0, __state->gpu_context->streams[0]);
}

