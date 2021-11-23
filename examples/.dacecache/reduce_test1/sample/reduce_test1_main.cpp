#include <cstdlib>
#include "../include/reduce_test1.h"

int main(int argc, char **argv) {
    reduce_test1Handle_t handle;
    double * __restrict__ __return = (double*) calloc(128, sizeof(double));
    double * __restrict__ inputs = (double*) calloc(16384, sizeof(double));


    handle = __dace_init_reduce_test1();
    __program_reduce_test1(handle, __return, inputs);
    __dace_exit_reduce_test1(handle);

    free(__return);
    free(inputs);


    return 0;
}
