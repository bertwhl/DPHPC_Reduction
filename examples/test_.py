import numpy as np
import dace
from dace.transformation.interstate import GPUTransformSDFG, StateFusion
from util import *

H = dace.symbol('H')
W = dace.symbol('W')

@dace.program
def AtomicReduceToGlobalMem(inputs: dace.float64[H, W], num_blocks_per_row: dace.int64, loopNum: dace.int64):
    outputs = dace.ndarray([H], dtype=dace.float64)
    outputs[:] = 0
    for blockIdx_y, blockIdx_x in dace.map[0:num_blocks_per_row, 0:H]:
        for warp_id, thread_id in dace.map[0:32,0:32]:
            row_id = blockIdx_x
            col_id = 1024*blockIdx_y + 32*warp_id +thread_id
            delta = 1024*num_blocks_per_row
            value = dace.float64(0)
            for loopIdx in dace.map[0:loopNum]:
                if col_id<W:
                    value += inputs[row_id, col_id]
                col_id += delta
            reduced = warpReduce_sum(value)
            if thread_id == 0:
                outputs[row_id] += reduced
    return outputs

@dace.program
def AtomicReduceToSharedNGlobalMem(inputs: dace.float64[H, W], num_blocks_per_row: dace.int64, loopNum: dace.int64):
    outputs = dace.ndarray([H], dtype=dace.float64)
    outputs[:] = 0
    for blockIdx_y, blockIdx_x in dace.map[0:num_blocks_per_row, 0:H]:
        shared = dace.ndarray([32], dtype=dace.float64, storage=dace.StorageType.GPU_Shared)
        row_id = blockIdx_x
        for warp_id, thread_id in dace.map[0:32,0:32]:
            col_id = 1024*blockIdx_y + 32*warp_id +thread_id
            delta = 1024*num_blocks_per_row
            value = dace.float64(0)
            for loopIdx in dace.map[0:loopNum]:
                if col_id<W:
                    value += inputs[row_id, col_id]
                col_id += delta
            reduced = warpReduce_sum(value)
            if thread_id == 0:
                shared[warp_id] = reduced
        for warp_id, thread_id in dace.map[0:32,0:32]:
            if warp_id == 0:
                value = shared[thread_id]
                reduced = warpReduce_sum(value)
                if thread_id == 0:
                    outputs[row_id] += reduced
    return outputs

@dace.program
def NonAtomicAdd(inputs: dace.float64[H, W], num_blocks_per_row: dace.int64, loopNum: dace.int64):
    outputs = dace.ndarray([H], dtype=dace.float64)
    outputs[:] = 0
    for blockIdx_y, blockIdx_x in dace.map[0:num_blocks_per_row, 0:H]:
        shared = dace.ndarray([1], dtype=dace.float64, storage=dace.StorageType.GPU_Shared)
        shared[0] = 0
        row_id = blockIdx_x
        delta = 1024*num_blocks_per_row
        for warp_id, thread_id in dace.map[0:32,0:32]:
            col_id = 1024*blockIdx_y + 32*warp_id +thread_id
            value = dace.float64(0)
            for loopIdx in dace.map[0:loopNum]:
                if col_id<W:
                    value += inputs[row_id, col_id]
                col_id += delta
            reduced = warpReduce_sum(value)
            if thread_id == 0:
                shared[0] += reduced
        for warp_id, thread_id in dace.map[0:32,0:32]:
            if (warp_id==0) and (thread_id==0):
                outputs[row_id] += shared[0] 
    return outputs



sdfg = NonAtomicAdd.to_sdfg()
sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})
row = 8
col = 1024*6
num_blocks_per_row = 1
loopNum = (col+1024*num_blocks_per_row-1) // (1024*num_blocks_per_row) 
test_input = np.random.rand(row, col)
test_output = sdfg(H=row, W=col, inputs=test_input, num_blocks_per_row=num_blocks_per_row, loopNum=loopNum)
expected_output = np.sum(test_input, axis=1)
print(test_output)
print(expected_output)
assert np.allclose(test_output, expected_output)
