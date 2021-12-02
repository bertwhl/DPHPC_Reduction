import numpy as np
from dace.transformation.interstate import GPUTransformSDFG

import Reduce1D, Reduce2D

# squeeze the dimensions
def squeeze(a, axis):
    # input parameters
    shape_in = a.shape
    if type(axis) == tuple:
        axis_in = list(axis)
    else:
        axis_in = [axis]
    axis_in.sort()
    axis_in.append(-1)
    dim_in = len(shape_in)
    # output parameters
    shape_out = []
    dim_out = 0
    # temps
    shape_temp = 1
    flag = -1
    flag_prev = -1
    itr = iter(axis_in)
    axis_red = next(itr)
    for idx in range(dim_in):
        flag = 1 if (axis_red==idx) else 0
        if flag:
            axis_red = next(itr)
        if(flag!=flag_prev):
            flag_prev = flag
            dim_out += 1
            shape_out.append(shape_temp)
            shape_temp = shape_in[idx]
        else:
            shape_temp *= shape_in[idx]
    shape_out.append(shape_temp)
    shape = tuple(shape_out[1:])
    a.resize(shape,refcheck=False)
    return flag, dim_out, shape

def Reduce(inputs, axis):
    # squeeze the dimensions
    RedType, Dim, Shape = squeeze(a=inputs, axis=axis)

    # default block number(number of SMs)
    BlockDefault = 64

    # The Scheduler
    # 1-D case
    if Dim == 1:
        w, = Shape

        # calculate parameters
        BlockMax = (w+512-1)//512
        if (BlockMax<4*BlockDefault):
            BlockNum = BlockMax
            loopNum = 1
        else:
            BlockNum = 4*BlockDefault
            loopNum = (w+512*BlockNum-1)//(512*BlockNum)

        # create sdfg
        sdfg = Reduce1D.Reduce.to_sdfg()
        sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

        # reduce
        outputs = sdfg(W=w, inputs=inputs, gridDim_x=BlockNum, loopNum=loopNum)
        return outputs

    # 2-D case
    if Dim == 2:
        h, w = Shape

        # Row Reduction
        if RedType == 1:

            # Narrow Case
            if (w<32):
                # calculate parameters
                ThreadPerBlock = 256
                BlockNum = (h+ThreadPerBlock-1)//ThreadPerBlock

                # create sdfg
                sdfg = Reduce2D.RowReduceNarrow.to_sdfg()
                sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

                # reduce
                outputs = sdfg(H=h, W=w, inputs=inputs, gridDim_x=BlockNum, blockDim_x=ThreadPerBlock)
                return outputs

            # *** Case
            else:
                '''
                TODO: WRITE AND TEST 2D ROW REDUCTION FUNCTION AND FINISH THE SCHEDULER HERE
                '''
                return
                
        # Column Reduction
        if RedType == 0:

            # Narrow Case
            if (w<16 and h>1024*8):
                # calculate parameters
                BlockMax = (h+32-1)//32
                if BlockDefault > BlockMax:
                    BlockPerColumn = BlockMax
                    loopNum = 1
                else:
                    BlockPerColumn = BlockDefault
                    loopNum = (h+32*BlockPerColumn-1)//(32*BlockPerColumn)

                # create sdfg
                sdfg = Reduce2D.ColReduceNarrow.to_sdfg()
                sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

                # reduce
                outputs = sdfg(H=h, W=w, inputs=inputs, gridDim_x=BlockPerColumn, loopNum=loopNum)
                return outputs

            # Normal Case
            else:
                # calculate parameters
                if w>256:
                    ThreadPerBlock = 256
                    BlockPerRow = (w+ThreadPerBlock-1)//ThreadPerBlock
                    if BlockPerRow >= BlockDefault:
                        BlockPerColumn = 1
                    else:
                        Default = BlockDefault//BlockPerRow
                        if h<Default:
                            BlockPerColumn = h
                        else:
                            BlockPerColumn = Default
                    loopNum = (h+BlockPerColumn-1)//BlockPerColumn
                else:
                    ThreadPerBlock = w
                    BlockPerRow = 1
                    Default = BlockDefault * 256 // ThreadPerBlock
                    if h<Default:
                        BlockPerColumn = h
                    else:
                        BlockPerColumn = Default
                    loopNum = (h+BlockPerColumn-1)//BlockPerColumn

                # create sdfg
                sdfg = Reduce2D.ColReduce.to_sdfg()
                sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

                # reduce
                outputs = sdfg(H=h, W=w, inputs=inputs, gridDim_x=BlockPerRow, gridDim_y=BlockPerColumn, blockDim_x=ThreadPerBlock, loopNum=loopNum)
                return outputs

if __name__ == '__main__':
    inputs = np.random.rand(1024,1024,128)
    compared = np.sum(inputs, axis=(0,1,2))
    outputs = Reduce(inputs, axis=(0,1,2))
    assert np.allclose(outputs, compared)