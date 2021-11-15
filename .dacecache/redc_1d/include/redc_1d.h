#include <dace/dace.h>
typedef void * redc_1dHandle_t;
extern "C" redc_1dHandle_t __dace_init_redc_1d();
extern "C" void __dace_exit_redc_1d(redc_1dHandle_t handle);
extern "C" void __program_redc_1d(redc_1dHandle_t handle, double * __restrict__ __return, double * __restrict__ a);
