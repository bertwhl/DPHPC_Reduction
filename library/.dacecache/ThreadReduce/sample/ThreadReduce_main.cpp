#include <cstdlib>
#include "../include/ThreadReduce.h"

int main(int argc, char **argv) {
    ThreadReduceHandle_t handle;
    int H = 42;
    int W = 42;
    long long blockDim_x = 42;
    long long gridDim_x = 42;
    long long loopNum = 42;
    long long rowNum = 42;
    double * __restrict__ __return = (double*) calloc(W, sizeof(double));
    double * __restrict__ inputs = (double*) calloc((H * W), sizeof(double));


    handle = __dace_init_ThreadReduce(H, W, blockDim_x, gridDim_x, loopNum, rowNum);
    __program_ThreadReduce(handle, __return, inputs, H, W, blockDim_x, gridDim_x, loopNum, rowNum);
    __dace_exit_ThreadReduce(handle);

    free(__return);
    free(inputs);


    return 0;
}
