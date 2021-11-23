#include <cstdlib>
#include "../include/reduce_test3.h"

int main(int argc, char **argv) {
    reduce_test3Handle_t handle;
    double * __restrict__ __return = (double*) calloc(4096, sizeof(double));
    double * __restrict__ inputs = (double*) calloc(32768, sizeof(double));


    handle = __dace_init_reduce_test3();
    __program_reduce_test3(handle, __return, inputs);
    __dace_exit_reduce_test3(handle);

    free(__return);
    free(inputs);


    return 0;
}
