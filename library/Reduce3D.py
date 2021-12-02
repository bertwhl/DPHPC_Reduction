import dace
from util import *

H = dace.symbol('H')
W = dace.symbol('W')
X = dace.symbol('X')

#A1-B1-A2 -> A1-A2
#using more blocks
@dace.program
def ColReduce(inputs: dace.float64[X, H, W], gridDim_x: dace.int64, gridDim_y: dace.int64, blockDim_x: dace.int64, loopNum: dace.int64):
    # define and initialize the output vector
    outputs = dace.ndarray([X,W], dtype=dace.float64)
    outputs[:] = 0
    # blocks mapping
    for blockIdx_y, blockIdx_x in dace.map[0:gridDim_y, 0:gridDim_x]:
        # thread mapping
        for threadIdx_x in dace.map[0:blockDim_x]:
            # initialize value
            value = dace.float64(0)
            # calculate the indexes
            levelIdx = blockIdx_y // H
            rowIdx = dace.int64(blockIdx_y % H)
            colIdx = blockDim_x * blockIdx_x + threadIdx_x
            # sum up the values
            if colIdx<W:
                delta = gridDim_y
                for loopIdx in dace.map[0:loopNum]:
                    if rowIdx<H:
                        value += inputs[levelIdx, rowIdx, colIdx]
                    rowIdx += delta
                # write back to global memory with atomic add
                outputs[levelIdx,colIdx] += value
    return outputs


#B1-A1-B2 -> A1
#using more blocks + atomic add
@dace.program
def RowReduceNarrow(inputs: dace.float64[X, H, W], gridDim_y: dace.int64, gridDim_x: dace.int64, blockDim_x: dace.int64):
    outputs = dace.ndarray([H], dtype=dace.float64)
    outputs[:] = 0
    for blockIdx_y, blockIdx_x in dace.map[0:gridDim_y, 0:gridDim_x]:
        for threadIdx_x in dace.map[0:blockDim_x]:
            value = dace.float64(0)
            rowIdx = blockIdx_x*blockDim_x+threadIdx_x
            if rowIdx < H:
                for colIdx in dace.map[0:W]:
                    value += inputs[blockIdx_y,rowIdx,colIdx]
                outputs[rowIdx] += value
    return outputs

if __name__ == '__main__':
    import numpy as np
    from dace.transformation.interstate import GPUTransformSDFG
    test = 'colreduce'


    if test == 'colreduce':
        sdfg = ColReduce.to_sdfg()
        sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

        x = 12
        h = 1000
        w = 12

        # Test
        inputs = np.random.rand(x, h, w)
        outputs = sdfg(X=x, H=h, W=w, inputs=inputs, gridDim_x=1, gridDim_y=12000, blockDim_x=12, loopNum=1)
        compared = np.sum(inputs, axis=1)
        assert np.allclose(outputs, compared)
        print("test 3D column reduce successfully")

    if test == 'rowreduce':
        sdfg = RowReduceNarrow.to_sdfg()
        sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

        x = 12
        h = 1000
        w = 12

        # Test
        inputs = np.random.rand(x, h, w)
        outputs = sdfg(X=x, H=h, W=w, inputs=inputs, gridDim_y=12, gridDim_x=4, blockDim_x=256)
        compared = np.sum(inputs, axis=(0,2))
        assert np.allclose(outputs, compared)

        print("test 3D row reduce successfully")
