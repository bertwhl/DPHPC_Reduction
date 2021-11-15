
#include <cuda_runtime.h>
#include <dace/dace.h>


struct redc_1d_t {
    dace::cuda::Context *gpu_context;
};



DACE_EXPORTED int __dace_init_cuda(redc_1d_t *__state);
DACE_EXPORTED void __dace_exit_cuda(redc_1d_t *__state);

DACE_DFI void redc_1d_12_4_13_8_0_0_10(double * __tmp_18_27_r, double * __tmp_18_34_r, double * __tmp_18_20_w, long long j, long long k) {
    double __tmp4[1]  DACE_ALIGN(64);
    long long __tmp5;
    long long stride;
    long long t;
    long long __sym___tmp3;
    long long __sym___tmp5;

    {

        {
            long long __out;

            ///////////////////
            // Tasklet code (assign_14_12)
            __out = 512;
            ///////////////////

            stride = __out;
        }

    }
    while ((stride > 0)) {
        {

            {
                long long __out;

                ///////////////////
                // Tasklet code (assign_16_16)
                __out = ((32 * j) + k);
                ///////////////////

                t = __out;
            }

        }
        if ((t < stride)) {
            __sym___tmp3 = t;
            {

                #pragma omp parallel sections
                {
                    #pragma omp section
                    {

                        dace::CopyND<double, 1, false, 1>::template ConstDst<1>::Copy(
                        __tmp_18_27_r + __sym___tmp3, __tmp4, 1);
                    } // End omp section
                    #pragma omp section
                    {
                        {
                            long long __in1 = t;
                            long long __in2 = stride;
                            long long __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            __tmp5 = __out;
                        }
                    } // End omp section
                } // End omp sections

            }
            __sym___tmp5 = __tmp5;
            __sym___tmp3 = t;
            {
                double __tmp7[1]  DACE_ALIGN(64);

                {
                    double __in1 = __tmp4[0];
                    double __in2 = __tmp_18_34_r[__sym___tmp5];
                    double __out;

                    ///////////////////
                    // Tasklet code (_Add_)
                    __out = (__in1 + __in2);
                    ///////////////////

                    __tmp7[0] = __out;
                }
                {
                    double __inp = __tmp7[0];
                    double __out;

                    ///////////////////
                    // Tasklet code (assign_18_20)
                    __out = __inp;
                    ///////////////////

                    __tmp_18_20_w[__sym___tmp3] = __out;
                }

            }
            goto __state_2_BinOp_19;

        }
        __state_2_BinOp_19:;
        {
            double __tmp8;

            #pragma omp parallel sections
            {
                #pragma omp section
                {
                    {
                        long long __in1 = stride;
                        double __out;

                        ///////////////////
                        // Tasklet code (_Div_)
                        __out = (dace::float64(__in1) / dace::float64(2));
                        ///////////////////

                        __tmp8 = __out;
                    }
                    {
                        double __inp = __tmp8;
                        long long __out;

                        ///////////////////
                        // Tasklet code (assign_19_16)
                        __out = __inp;
                        ///////////////////

                        stride = __out;
                    }
                } // End omp section
                #pragma omp section
                {
                    {

                        ///////////////////
                        __syncthreads();
                        ///////////////////

                    }
                } // End omp section
            } // End omp sections

        }

    }
    
}



int __dace_init_cuda(redc_1d_t *__state) {
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

void __dace_exit_cuda(redc_1d_t *__state) {
    

    // Destroy cuda streams and events
    for(int i = 0; i < 2; ++i) {
        cudaStreamDestroy(__state->gpu_context->streams[i]);
    }
    for(int i = 0; i < 1; ++i) {
        cudaEventDestroy(__state->gpu_context->events[i]);
    }

    delete __state->gpu_context;
}

__global__ void redc_1d_12_0_0_2(double * __restrict__ gpu_a) {
    {
        int i = blockIdx.x;
        {
            {
                {
                    int k = threadIdx.x;
                    int j = threadIdx.y;
                    if (k < 32) {
                        if (j < 16) {
                            redc_1d_12_4_13_8_0_0_10(&gpu_a[0], &gpu_a[0], &gpu_a[0], j, k);
                        }
                    }
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_redc_1d_12_0_0_2(redc_1d_t *__state, double * __restrict__ gpu_a);
void __dace_runkernel_redc_1d_12_0_0_2(redc_1d_t *__state, double * __restrict__ gpu_a)
{

    void  *redc_1d_12_0_0_2_args[] = { (void *)&gpu_a };
    cudaLaunchKernel((void*)redc_1d_12_0_0_2, dim3(int_ceil(1, 1), 1, 1), dim3(32, 16, 1), redc_1d_12_0_0_2_args, 0, __state->gpu_context->streams[0]);
}
__global__ void assign_21_4_gmap_0_0_11(double * __restrict__ gpu___return, const double * __restrict__ __tmp0) {
    {
        int assign_21_4__gmapi = (blockIdx.x * 32 + threadIdx.x);
        if (assign_21_4__gmapi < 1) {
            {
                double __inp = __tmp0[0];
                double __out;

                ///////////////////
                // Tasklet code (assign_21_4)
                __out = __inp;
                ///////////////////

                gpu___return[0] = __out;
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_assign_21_4_gmap_0_0_11(redc_1d_t *__state, double * __restrict__ gpu___return, const double * __restrict__ __tmp0);
void __dace_runkernel_assign_21_4_gmap_0_0_11(redc_1d_t *__state, double * __restrict__ gpu___return, const double * __restrict__ __tmp0)
{

    void  *assign_21_4_gmap_0_0_11_args[] = { (void *)&gpu___return, (void *)&__tmp0 };
    cudaLaunchKernel((void*)assign_21_4_gmap_0_0_11, dim3(int_ceil(int_ceil(1, 1), 32), int_ceil(1, 1), int_ceil(1, 1)), dim3(32, 1, 1), assign_21_4_gmap_0_0_11_args, 0, __state->gpu_context->streams[0]);
}

