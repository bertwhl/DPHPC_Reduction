import dace
from util import *

W = dace.symbol('W')
H = dace.symbol('H')

@dace.program
def Reduce(inputs: dace.float64[W], gridDim_x: dace.int64, blockDim_y: dace.int64, loopNum: dace.int64):
    # define and initialize the output vector
    outputs = dace.ndarray([1], dtype=dace.float64)
    outputs[0] = 0
    # blocks mapping
    for blockIdx_x in dace.map[0:gridDim_x]:
        # thread mapping
        for threadIdx_y, threadIdx_x in dace.map[0:blockDim_y, 0:32]:
            # initialize value
            value = dace.float64(0)
            # calculate the indexes
            Idx0 = 32*(blockDim_y*blockIdx_x + threadIdx_y) + threadIdx_x
            delta = 32 * gridDim_x * blockDim_y
            # sum up the values
            for loopIdx in dace.map[0:loopNum]:
                Idx = Idx0 + loopIdx * delta
                if Idx<W:
                    value += inputs[Idx]
                # write back to global memory with atomic add
            reduced = warpReduce_sum(value)
            if threadIdx_x == 0:
                outputs[0] += reduced
    return outputs

@dace.program
def Reduce1(inputs: dace.float64[W], gridDim_x: dace.int64, loopNum: dace.int64):
    # define and initialize the output vector
    outputs = dace.ndarray([1], dtype=dace.float64)
    outputs[0] = 0
    # blocks mapping
    for blockIdx_x in dace.map[0:gridDim_x]:
        shared = dace.ndarray([32], dtype=dace.float64, storage=dace.StorageType.GPU_Shared)
        # thread mapping
        for threadIdx_y, threadIdx_x in dace.map[0:32, 0:32]:
            # initialize value
            value = dace.float64(0)
            # calculate the indexes
            Idx = 32*(32*blockIdx_x + threadIdx_y) + threadIdx_x
            delta = 1024 * gridDim_x
            # sum up the values
            for loopIdx in dace.map[0:loopNum]:
                if Idx<W:
                    value += inputs[Idx]
                Idx += delta
                # write back to global memory with atomic add
            reduced = warpReduce_sum(value)
            if threadIdx_x == 0:
                shared[threadIdx_y] = reduced
        for threadIdx_y, threadIdx_x in dace.map[0:32, 0:32]:
            if threadIdx_y == 0:
                value = shared[threadIdx_x]
                reduced = warpReduce_sum(value)
                if threadIdx_x == 0:
                    outputs[0] += reduced
    return outputs

@dace.program
def Reduce1_(inputs: dace.float64[H, W], gridDim_x: dace.int64, loopNum: dace.int64):
    outputs = dace.ndarray([H], dtype=dace.float64)
    outputs[:] = 0
    for blockIdx_y, blockIdx_x in dace.map[0:gridDim_x, 0:H]:
        shared = dace.ndarray([32], dtype=dace.float64, storage=dace.StorageType.GPU_Shared)
        row_id = blockIdx_x
        for warp_id, thread_id in dace.map[0:32,0:32]:
            col_id = 1024*blockIdx_y + 32*warp_id +thread_id
            delta = 1024*gridDim_x
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

if __name__ == '__main__':
    import numpy as np
    from dace.transformation.interstate import GPUTransformSDFG
    # Transform to GPU, keep thread-block map
    sdfg = Reduce.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

    w = 1024*1024*128
    BlockNum = 256
    WarpPerBlock = 16
    # other params
    loopNum = (w+32*BlockNum*WarpPerBlock-1)//(32*BlockNum*WarpPerBlock)

    # Test
    inputs = np.random.rand(w)
    outputs = sdfg(
                # H=1,
                W=w, inputs=inputs, gridDim_x=BlockNum,
                blockDim_y=WarpPerBlock,
                loopNum=loopNum)
    compared = np.sum(inputs, axis=0)
    assert np.allclose(outputs, compared)