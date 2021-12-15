import numpy as np
from dace.transformation.interstate import GPUTransformSDFG

import Reduce1D, Reduce2D, Reduce3D

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

            # multi-line-per-block Case
            elif (h>=64):
                if (h>BlockDefault*32):
                    RowPerBlock = 32
                    WarpPerRow = 1
                else:
                    RowPerBlock = (h+BlockDefault-1)//BlockDefault
                    WarpMax = (w+32-1)//32
                    WarpAva = 32//RowPerBlock
                    if (WarpAva>WarpMax):
                        WarpPerRow = WarpMax
                    else:
                        WarpPerRow = WarpAva
                BlockNum = (h+RowPerBlock-1)//RowPerBlock
                loopNum = (w+WarpPerRow*32-1)//(WarpPerRow*32)

                # create sdfg
                sdfg = Reduce2D.RowReduceMulti.to_sdfg()
                sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

                # reduce
                outputs = sdfg(H=h, W=w, inputs=inputs, gridDim_x=BlockNum, loopNum=loopNum, blockDim_z=RowPerBlock, blockDim_y=WarpPerRow)
                return outputs

            # use no-loop version
            elif (w<=1024):
                # calculate parameters
                WarpPerBlock = (w+31)//32

                # create sdfg
                sdfg = Reduce2D.RowReduceNoLoop.to_sdfg()
                sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

                # reduce
                outputs = sdfg(H=h, W=w, inputs=inputs, blockDim_y=WarpPerBlock)
                return outputs

            # use global memory version
            elif (h>32):
                # calculate parameters
                if (h==64):
                    BlockPerRow = 1
                else:
                    BlockPerRow = 2
                loopNum = (w+BlockPerRow*1024-1)//(BlockPerRow*1024)

                # create sdfg
                sdfg = Reduce2D.RowReduceGlobal.to_sdfg()
                sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

                # reduce
                outputs = sdfg(H=h, W=w, inputs=inputs, gridDim_y=BlockPerRow, loopNum=loopNum)
                return outputs
            
            # use shared memory
            else:
                # calculate parameters
                BlockPerRow = BlockDefault//h
                loopNum = (w+BlockPerRow*1024-1)//(BlockPerRow*1024)

                # create sdfg
                sdfg = Reduce2D.RowReduceShared.to_sdfg()
                sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

                # reduce
                outputs = sdfg(H=h, W=w, inputs=inputs, gridDim_y=BlockPerRow, loopNum=loopNum)
                return outputs

                

                
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
    if Dim == 3:
        x, h, w = Shape
        if RedType == 0:
            if w < 32:
                gridDim_x = max(1,int(64//x))
                loopNum = int(np.ceil(h/gridDim_x/32))

                # create sdfg
                sdfg = Reduce3D.ColReduceNarrow.to_sdfg()
                sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

                # reduce
                outputs = sdfg(X=x, H=h, W=w, inputs=inputs, gridDim_x=gridDim_x, loopNum=loopNum)
                return outputs
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
                sdfg = Reduce3D.ColReduce.to_sdfg()
                sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

                # reduce
                outputs = sdfg(X=x, H=h, W=w, inputs=inputs, gridDim_x=BlockPerRow, gridDim_y=BlockPerColumn, blockDim_x=ThreadPerBlock, loopNum=loopNum)
                return outputs

        else:
            raise NotImplementedError
if __name__ == '__main__':
    inputs = np.random.rand(1795,23)
    compared = np.sum(inputs, axis=(1))
    outputs = Reduce(inputs, axis=(1))
    assert np.allclose(outputs, compared)
