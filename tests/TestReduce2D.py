import sys
import numpy as np
import math
from dace.transformation.interstate import GPUTransformSDFG

sys.path.insert(0, "/home/zhoubo")
from DPHPC_Reduction.library.Reduce2D import *
from DPHPC_Reduction.tests.ColumnReduce import WarpReadWarpReduce


# --------- AB Tests ---------


# 1
def test_rrm ():
    sdfg = RowReduceMulti.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

    BlockDefault = 64
    if (h > BlockDefault * 32):
        RowPerBlock = 32
        WarpPerRow = 1
    else:
        RowPerBlock = (h + BlockDefault - 1) // BlockDefault
        WarpPerRow = 32 // RowPerBlock
    BlockNum = (h + RowPerBlock - 1) // RowPerBlock
    loopNum = (w + WarpPerRow * 32 - 1) // (WarpPerRow * 32)

    input = np.random.rand(h, w)
    output = sdfg(H=h, W=w, inputs=input, 
        gridDim_x=BlockNum,loopNum=loopNum, blockDim_z=RowPerBlock, blockDim_y=WarpPerRow)
    ans = np.sum(input, axis=1)
    assert np.allclose(ans, output)

# 2
def test_rrnl():
    sdfg = RowReduceNoLoop.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

    warp_num = math.ceil(w / 32) # w should be 1~32
 
    input = np.random.rand(h, w)
    output = sdfg(H=h, W=w, inputs=input, blockDim_y=warp_num)
    ans = np.sum(input, axis=1)
    assert np.allclose(ans, output)

# 3
def test_rrg():
    sdfg = RowReduceGlobal.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

    BlockDefault = 64
    BlockPerRow = max(1, math.floor(BlockDefault / h))
    loop_num = (w+BlockPerRow*1024-1)//(BlockPerRow*1024)

    input = np.random.rand(h, w)
    output = sdfg(H=h, W=w, inputs=input, gridDim_y=BlockPerRow, loopNum=loop_num)
    ans = np.sum(input, axis=1)
    assert np.allclose(ans, output)

# 4
def test_rrs():
    sdfg = RowReduceShared.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

    BlockDefault = 64
    grid_dim = max(1, math.floor(BlockDefault / h))
    loop_num = math.ceil(w / grid_dim / 1024)

    input = np.random.rand(h, w)
    output = sdfg(H=h, W=w, inputs=input, gridDim_y=grid_dim, loopNum=loop_num)
    ans = np.sum(input, axis=1)
    assert np.allclose(ans, output)

# 5
def test_rrn():
    sdfg = RowReduceNarrow.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

    ThreadPerBlock = 256
    BlockNum = (h+ThreadPerBlock-1)//ThreadPerBlock

    input = np.random.rand(h, w)
    output = sdfg(H=h, W=w, inputs=input, gridDim_x=BlockNum, blockDim_x=ThreadPerBlock)
    ans = np.sum(input, axis=1)
    assert np.allclose(ans, output)


# --------- BA Tests ---------


# 6
def test_cr():
    sdfg = ColReduce.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

    ThreadPerBlock = w
    BlockPerRow = 1
    Default = 256
    if h<Default:
        BlockPerColumn = h
    else:
        BlockPerColumn = Default
    loopNum = (h+BlockPerColumn-1)//BlockPerColumn

    inputs = np.random.rand(h, w)
    outputs = sdfg(H=h, W=w, inputs=inputs, gridDim_x=BlockPerRow, gridDim_y=BlockPerColumn, blockDim_x=ThreadPerBlock, loopNum=loopNum)
    compared = np.sum(inputs, axis=0)
    assert np.allclose(outputs, compared)

# 7
def test_crn():
    sdfg = ColReduceNarrow.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

    BlockDefault = 64
    BlockMax = (h+32-1)//32
    if BlockDefault > BlockMax:
        BlockPerColumn = BlockMax
        loopNum = 1
    else:
        BlockPerColumn = BlockDefault
        loopNum = (h+32*BlockPerColumn-1)//(32*BlockPerColumn)

    inputs = np.random.rand(h, w)
    outputs = sdfg(H=h, W=w, inputs=inputs, gridDim_x=BlockPerColumn, loopNum=loopNum)
    compared = np.sum(inputs, axis=0)
    assert np.allclose(outputs, compared)    

# 8
def test_wrwr():
    sdfg = WarpReadWarpReduce.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

    NumSubcolumn = (w+31)//32
    BlockDefault = 64
    Default = BlockDefault//NumSubcolumn
    BlockMax = (h+32-1)//32
    if Default<BlockMax:
        BlockPerColumn = Default
    else:
        BlockPerColumn = BlockMax
    loopNum = (h+32*BlockPerColumn-1)//(32*BlockPerColumn)

    inputs = np.random.rand(h, w)
    outputs = sdfg(H=h, W=w, inputs=inputs, gridDim_x=NumSubcolumn, gridDim_y=BlockPerColumn, loopNum=loopNum)
    compared = np.sum(inputs, axis=0)
    assert np.allclose(outputs, compared) 


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

# 9
def test_non_atomic():
    sdfg = NonAtomicAdd.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

    BlockDefault = 64
    grid_dim = max(1, math.floor(BlockDefault / h))
    loop_num = math.ceil(w / grid_dim / 1024)

    input = np.random.rand(h, w)
    output = sdfg(H=h, W=w, inputs=input, num_blocks_per_row=grid_dim, loopNum=loop_num)
    ans = np.sum(input, axis=1)
    assert np.allclose(ans, output)


# --------- Run Tests ---------


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise Exception('only accept 3 arguments')
    test_case = sys.argv[1]
    h, w = int(sys.argv[2]), int(sys.argv[3])
    
    for i in range(1):
        if test_case ==  "1":
            test_rrm()
        elif test_case == "2":
            test_rrnl()
        elif test_case == "3":
            test_rrg()
        elif test_case == "4":
            test_rrs()
        elif test_case == "5":
            test_rrn()
        elif test_case == "6":
            test_cr()
        elif test_case == "7":
            test_crn()
        elif test_case == "8":
            test_wrwr()
        elif test_case == "9": 
            test_non_atomic()
        else:
            raise Exception('invalid case number, only accept 1, 2, 3, 4, 5, 6, 7, 8, 9')