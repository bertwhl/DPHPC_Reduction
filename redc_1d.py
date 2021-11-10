import numpy as np
import pytest

import dace
from dace.transformation.interstate import GPUTransformSDFG
from dace.frontend.common import op_repository as oprepo

from util import find_map_by_param, sync_threads

@dace.program
def myprog(a: dace.float64[512]):
    for i in dace.map[0:1]:
        for j, k in dace.map[0:16, 0:32]:
            stride = 512
            tid = j * 32 + k
            while stride > 32:
                a[tid] = a[tid] + a[tid + stride]
                stride = stride / 2
                sync_threads()
            if j == 0:
                while stride > 0:
                    a[k] = a[k] + a[k + stride]
                    stride = stride / 2
    return a[0]


'''

when trying to unroll everything like this:

###########################################

@dace.program
def myprog(a: dace.float64[512]):
    for i in dace.map[0:1]:
        for j, k in dace.map[0:16, 0:32]:
            tid = j * 32 + k
            a[tid] = a[tid] + a[tid + 512]
            sync_threads()
            a[tid] = a[tid] + a[tid + 256]
            sync_threads()
            a[tid] = a[tid] + a[tid + 128]
            sync_threads()
            a[tid] = a[tid] + a[tid + 64]
            sync_threads()
            a[tid] = a[tid] + a[tid + 32]
            a[tid] = a[tid] + a[tid + 16]
            a[tid] = a[tid] + a[tid + 8]
            a[tid] = a[tid] + a[tid + 4]
            a[tid] = a[tid] + a[tid + 2]
            a[tid] = a[tid] + a[tid + 1]
    return a[0]

###########################################

dace complains:

    "DaceSyntaxError: Incompatible subsets __sym___tmp3 and 0:512" 

not knowing why...

'''

@pytest.mark.gpu
def test_redc_1d():
    print('1-D reduction test')

    sdfg = redc_1d.to_sdfg()
    block_map = find_map_by_param(sdfg, 'i')
    block_map.schedule = dace.ScheduleType.GPU_Device
    thread_map = find_map_by_param(sdfg, 'j')
    thread_map.schedule = dace.ScheduleType.GPU_ThreadBlock
    sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

    # Test
    arr_in = np.random.rand(512)
    ans = np.sum(arr_in, axis=0)
    out = sdfg(arr_in)

    print("input size: 512, expected: %f, got: %f" % (ans, out))
    assert abs(ans - out) <= 1e-5

if __name__ == '__main__':
    test_redc_1d()