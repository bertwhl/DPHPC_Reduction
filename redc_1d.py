import numpy as np
import pytest

import dace
from dace.transformation.interstate import GPUTransformSDFG
from dace.frontend.common import op_repository as oprepo

from util import find_map_by_param, sync_threads

@dace.program
def redc_1d(a: dace.float64[512]):
    for i in dace.map[0:1]:
        for j, k in dace.map[0:16, 0:32]:
            stride = 512
            while stride > 0:
                t = j * 32 + k
                if t < stride:
                    a[t] = a[t] + a[t + stride]
                stride = stride / 2
                sync_threads()
    return a[0]

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