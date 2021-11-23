#include <cstdlib>
#include "../include/reduce_test2.h"

int main(int argc, char **argv) {
    reduce_test2Handle_t handle;
    double * __restrict__ __return = (double*) calloc(128, sizeof(double));
    double * __restrict__ inputs = (double*) calloc(16384, sizeof(double));


    handle = __dace_init_reduce_test2();
    __program_reduce_test2(handle, __return, inputs);
    __dace_exit_reduce_test2(handle);

    free(__return);
    free(inputs);


    return 0;
}
