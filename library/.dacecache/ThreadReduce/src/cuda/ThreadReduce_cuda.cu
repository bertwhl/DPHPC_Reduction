
#include <cuda_runtime.h>
#include <dace/dace.h>


struct ThreadReduce_t {
    dace::cuda::Context *gpu_context;
};



DACE_EXPORTED int __dace_init_cuda(ThreadReduce_t *__state, int H, int W, long long blockDim_x, long long gridDim_x, long long loopNum, long long rowNum);
DACE_EXPORTED void __dace_exit_cuda(ThreadReduce_t *__state);

DACE_DFI void ThreadReduce_47_4_49_8_59_16_2_1_5(const double * __tmp_61_33_r, long long& __tmp_61_40_r, long long& __tmp_61_48_r, long long& __tmp_62_30_r, double& __tmp_61_24_w, long long& __tmp_62_20_w, int H, int W, long long rowIdx) {
    long long __sym___tmp_61_40_r;
    long long __sym___tmp_61_48_r;

    if ((rowIdx < H)) {
        __sym___tmp_61_40_r = __tmp_61_40_r;
        __sym___tmp_61_48_r = __tmp_61_48_r;
        {

            {
                double __inp = __tmp_61_33_r[((W * __sym___tmp_61_40_r) + __sym___tmp_61_48_r)];
                double __out;

                ///////////////////
                // Tasklet code (assign_61_24)
                __out = __inp;
                ///////////////////

                dace::wcr_fixed<dace::ReductionType::Sum, double>::reduce(&__tmp_61_24_w, __out);
            }

        }
        goto __state_3_assign_62_20;

    }
    __state_3_assign_62_20:;
    {

        {
            long long __inp = __tmp_62_30_r;
            long long __out;

            ///////////////////
            // Tasklet code (assign_62_20)
            __out = __inp;
            ///////////////////

            dace::wcr_fixed<dace::ReductionType::Sum, long long>::reduce(&__tmp_62_20_w, __out);
        }

    }
    
}

DACE_DFI void ThreadReduce_47_4_49_8_0_0_8(const long long& __tmp_53_18_r, const long long& __tmp_58_24_r, const double * __tmp_61_33_r_in_from_10_0, double * __tmp_64_16_w, int H, int W, long long blockIdx_x, int loopNum, long long rowNum, long long threadIdx_x) {
    double value;
    long long rowIdx;
    long long colIdx;
    long long __sym___tmp8;

    {
        long long __tmp4;
        double __tmp6;
        long long Idx;

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
                    long long __in1 = __tmp_53_18_r;
                    long long __out;

                    ///////////////////
                    // Tasklet code (_Mult_)
                    __out = (__in1 * blockIdx_x);
                    ///////////////////

                    __tmp4 = __out;
                }
                {
                    long long __in1 = __tmp4;
                    long long __out;

                    ///////////////////
                    // Tasklet code (_Add_)
                    __out = (__in1 + threadIdx_x);
                    ///////////////////

                    Idx = __out;
                }
                {
                    long long __in1 = Idx;
                    double __out;

                    ///////////////////
                    // Tasklet code (_Div_)
                    __out = (dace::float64(__in1) / dace::float64(W));
                    ///////////////////

                    __tmp6 = __out;
                }
                {
                    double __inp = __tmp6;
                    long long __out;

                    ///////////////////
                    // Tasklet code (_convert_to_int64_)
                    __out = dace::int64(__inp);
                    ///////////////////

                    rowIdx = __out;
                }
                {
                    long long __in1 = Idx;
                    long long __out;

                    ///////////////////
                    // Tasklet code (_Mod_)
                    __out = (__in1 % dace::int64(W));
                    ///////////////////

                    colIdx = __out;
                }
            } // End omp section
        } // End omp sections

    }
    if ((rowIdx < rowNum)) {
        __sym___tmp8 = colIdx;
        {
            long long delta;

            {
                long long __inp = __tmp_58_24_r;
                long long __out;

                ///////////////////
                // Tasklet code (assign_58_16)
                __out = __inp;
                ///////////////////

                delta = __out;
            }
            {
                for (auto loopIdx = 0; loopIdx < loopNum; loopIdx += 1) {
                    ThreadReduce_47_4_49_8_59_16_2_1_5(&__tmp_61_33_r_in_from_10_0[0], rowIdx, colIdx, delta, value, rowIdx, H, W, rowIdx);
                }
            }
            {
                double __inp = value;
                double __out;

                ///////////////////
                // Tasklet code (assign_64_16)
                __out = __inp;
                ///////////////////

                dace::wcr_fixed<dace::ReductionType::Sum, double>::reduce_atomic(__tmp_64_16_w + __sym___tmp8, __out);
            }

        }
        goto __state_2_endif_57;

    }
    __state_2_endif_57:;
    
}



int __dace_init_cuda(ThreadReduce_t *__state, int H, int W, long long blockDim_x, long long gridDim_x, long long loopNum, long long rowNum) {
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

void __dace_exit_cuda(ThreadReduce_t *__state) {
    

    // Destroy cuda streams and events
    for(int i = 0; i < 2; ++i) {
        cudaStreamDestroy(__state->gpu_context->streams[i]);
    }
    for(int i = 0; i < 1; ++i) {
        cudaEventDestroy(__state->gpu_context->events[i]);
    }

    delete __state->gpu_context;
}

__global__ void assign_45_4_map_0_1_2(double * __restrict__ gpu___return, int W) {
    {
        int __i0 = (blockIdx.x * 32 + threadIdx.x);
        if (__i0 < W) {
            {
                double __out;

                ///////////////////
                // Tasklet code (assign_45_4)
                __out = 0;
                ///////////////////

                gpu___return[__i0] = __out;
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_assign_45_4_map_0_1_2(ThreadReduce_t *__state, double * __restrict__ gpu___return, int W);
void __dace_runkernel_assign_45_4_map_0_1_2(ThreadReduce_t *__state, double * __restrict__ gpu___return, int W)
{

    void  *assign_45_4_map_0_1_2_args[] = { (void *)&gpu___return, (void *)&W };
    cudaLaunchKernel((void*)assign_45_4_map_0_1_2, dim3(int_ceil(int_ceil(W, 1), 32), int_ceil(1, 1), int_ceil(1, 1)), dim3(32, 1, 1), assign_45_4_map_0_1_2_args, 0, __state->gpu_context->streams[0]);
}
__global__ void ThreadReduce_47_0_0_0(double * __restrict__ gpu___return, const double * __restrict__ gpu_inputs, int H, int W, const long long blockDim_x, int gridDim_x, long long loopNum, const long long rowNum) {
    {
        int blockIdx_x = blockIdx.x;
        {
            {
                int threadIdx_x = threadIdx.x;
                if (threadIdx_x < blockDim_x) {
                    ThreadReduce_47_4_49_8_0_0_8(blockDim_x, rowNum, &gpu_inputs[0], &gpu___return[0], H, W, blockIdx_x, loopNum, rowNum, threadIdx_x);
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_ThreadReduce_47_0_0_0(ThreadReduce_t *__state, double * __restrict__ gpu___return, const double * __restrict__ gpu_inputs, int H, int W, const long long blockDim_x, int gridDim_x, long long loopNum, const long long rowNum);
void __dace_runkernel_ThreadReduce_47_0_0_0(ThreadReduce_t *__state, double * __restrict__ gpu___return, const double * __restrict__ gpu_inputs, int H, int W, const long long blockDim_x, int gridDim_x, long long loopNum, const long long rowNum)
{

    void  *ThreadReduce_47_0_0_0_args[] = { (void *)&gpu___return, (void *)&gpu_inputs, (void *)&H, (void *)&W, (void *)&blockDim_x, (void *)&gridDim_x, (void *)&loopNum, (void *)&rowNum };
    cudaLaunchKernel((void*)ThreadReduce_47_0_0_0, dim3(int_ceil(gridDim_x, 1), 1, 1), dim3(max(1, blockDim_x), 1, 1), ThreadReduce_47_0_0_0_args, 0, __state->gpu_context->streams[0]);
}

