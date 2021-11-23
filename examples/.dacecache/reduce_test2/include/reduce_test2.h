#include <dace/dace.h>
typedef void * reduce_test2Handle_t;
extern "C" reduce_test2Handle_t __dace_init_reduce_test2();
extern "C" void __dace_exit_reduce_test2(reduce_test2Handle_t handle);
extern "C" void __program_reduce_test2(reduce_test2Handle_t handle, double * __restrict__ __return, double * __restrict__ inputs);
