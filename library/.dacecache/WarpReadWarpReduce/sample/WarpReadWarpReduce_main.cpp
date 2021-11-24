#include <cstdlib>
#include "../include/WarpReadWarpReduce.h"

int main(int argc, char **argv) {
    WarpReadWarpReduceHandle_t handle;
    int H = 42;
    int W = 42;
    long long gridDim_x = 42;
    long long gridDim_y = 42;
    long long loopNum = 42;
    double * __restrict__ __return = (double*) calloc(W, sizeof(double));
    double * __restrict__ inputs = (double*) calloc((H * W), sizeof(double));


    handle = __dace_init_WarpReadWarpReduce(H, W, gridDim_x, gridDim_y, loopNum);
    __program_WarpReadWarpReduce(handle, __return, inputs, H, W, gridDim_x, gridDim_y, loopNum);
    __dace_exit_WarpReadWarpReduce(handle);

    free(__return);
    free(inputs);


    return 0;
}
