import dace
import sys
import numpy as np
from dace.transformation.interstate import GPUTransformSDFG, StateFusion
from dace.frontend.common import op_repository as oprepo
from util import *

H = dace.symbol('H')
W = dace.symbol('W')

@dace.program
def AB_MWPR_1(inputs: dace.float64[H, W], wn: dace.int64):
    outputs = dace.ndarray([H], dtype=dace.float64)
    outputs[:] = 0
    for block_id in dace.map[0:H]:
        for warp_id, thread_id in dace.map[0:wn, 0:32]:
            col = warp_id*32+thread_id
            reduced = warpReduce_sum(inputs[block_id, col])
            if thread_id == 0:
                outputs[block_id] += reduced
    return outputs

@dace.program
def AB_MWPR_2(inputs: dace.float64[H, W], bn: dace.int64):
    outputs = dace.ndarray([H], dtype=dace.float64)
    outputs[:] = 0
    for block_id in dace.map[0:bn]:
        for warp_id, thread_id in dace.map[0:32, 0:32]:
            index = block_id * 1024 + warp_id * 32 + thread_id
            index_x = dace.int64(index/640)
            index_y = index % 640
            value = warpReduce_sum(inputs[index_x, index_y])
            if thread_id == 0:
                outputs[index_x] += value
    return outputs

@dace.program
def AB_MWPR_3(inputs: dace.float64[H, W], bn: dace.int64, ln: dace.int64, rpb: dace.int64, wpr: dace.int64):
    outputs = dace.ndarray([H], dtype=dace.float64)
    outputs[:] = 0
    for block_id in dace.map[0:bn]:
        _rpb = dace.int32(rpb)
        _wpr = dace.int32(wpr)
        shared = dace.ndarray([32,32], dtype=dace.float64, storage=dace.StorageType.GPU_Shared)
        for warp_id_x, warp_id_y, thread_id in dace.map[0:rpb, 0:wpr, 0:32]:
            value = dace.float64(0)
            for i in dace.map[0:ln]:
                value += inputs[block_id * _rpb + warp_id_x 
                    ,i * _wpr * 32 + warp_id_y * 32 + thread_id]
            shared[warp_id_x*wpr+warp_id_y, thread_id] = value
        reduced = dace.ndarray([32], dtype=dace.float64, storage=dace.StorageType.GPU_Shared) # should be rpb instead of 32, but stupid dace can't work with variables here...
        for warp_id_x, warp_id_y, thread_id in dace.map[0:rpb, 0:wpr, 0:32]:
            reduced[warp_id_x] = warpReduce_sum(shared[warp_id_x * wpr + warp_id_y, thread_id])
            if thread_id == 0:
                outputs[block_id * rpb + warp_id_x] += reduced[warp_id_x]
    return outputs

if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise Exception('only accept 3 argument')
    test_case = sys.argv[1]
    h, w = int(sys.argv[2]), int(sys.argv[3])
    if test_case ==  "1":
        sdfg1 = AB_MWPR_1.to_sdfg()
        sdfg1.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})
        
        warp_num = w // 32 # w should be 1~32
 
        a1 = np.random.rand(h, w)
        b1 = sdfg1(H = h, W = w, inputs = a1, wn = warp_num)
        c1 = np.sum(a1, axis=1)
        assert np.allclose(b1, c1)

    elif test_case == "2":
        sdfg2 = AB_MWPR_2.to_sdfg()
        sdfg2.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

        block_num = h * w // 512

        a2 = np.random.rand(h, w)
        b2 = sdfg2(H = h, W = w, inputs = a2, bn = block_num)
        c2 = np.sum(a2, axis=1)
        assert np.allclose(b2, c2)

    elif test_case == "3":
        sdfg3 = AB_MWPR_3.to_sdfg()
        sdfg3.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

        row_per_block = 16
        warp_per_row = 32 // row_per_block
        block_num = h // row_per_block
        loop_num = w * row_per_block // 1024

        a3 = np.random.rand(h, w)
        b3 = sdfg3(H = h, W = w, inputs = a3, bn = block_num, ln = loop_num, rpb = row_per_block, wpr = warp_per_row)
        c3 = np.sum(a3, axis=1)
        assert np.allclose(b3, c3)
    else:
        raise Exception('invalid case number, only accept 1,2,3')