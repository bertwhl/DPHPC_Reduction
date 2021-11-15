#include <cstdlib>
#include "../include/redc_1d.h"

int main(int argc, char **argv) {
    redc_1dHandle_t handle;
    double * __restrict__ __return = (double*) calloc(1, sizeof(double));
    double * __restrict__ a = (double*) calloc(512, sizeof(double));


    handle = __dace_init_redc_1d();
    __program_redc_1d(handle, __return, a);
    __dace_exit_redc_1d(handle);

    free(__return);
    free(a);


    return 0;
}
