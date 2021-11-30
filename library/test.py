from ColumnReduce import *

def test_WarpReadWarpReduce(h, w, BlockPerColumn):
    # Transform to GPU, keep thread-block map
    sdfg = WarpReadWarpReduce.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

    # the number of sub-columns
    NumSubcolumn = int(np.ceil(w/32))
    # calculate the number of values read and sum up first
    loopNum = int(np.ceil(h/(32*BlockPerColumn)))

    # Test
    inputs = np.random.rand(h, w)
    outputs = sdfg(H=h, W=w, inputs=inputs, gridDim_x=NumSubcolumn, gridDim_y=BlockPerColumn, loopNum=loopNum)
    compared = np.sum(inputs, axis=0)
    assert np.allclose(outputs, compared)

def test_WarpReadWarpReduceCorner(h, w, BlockPerColumn):
    # Transform to GPU, keep thread-block map
    sdfg = WarpReadWarpReduceCorner.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

    h = 1024*64
    w = 7
    BlockPerColumn = 64
    # calculate the number of values read and sum up first
    RowPerWarp = 32//w
    RowPerBlock = 32*RowPerWarp
    loopNum = (h+RowPerBlock*BlockPerColumn-1)//(RowPerBlock*BlockPerColumn)

    # Test
    inputs = np.random.rand(h, w)
    outputs = sdfg(H=h, W=w, inputs=inputs, gridDim_x=BlockPerColumn, rowNum_warp=RowPerWarp, rowNum_block=RowPerBlock, loopNum=loopNum)
    compared = np.sum(inputs, axis=0)
    assert np.allclose(outputs, compared)

def test_ThreadReduce(h, w, ThreadPerBlock, BlockNum):
    # Transform to GPU, keep thread-block map
    sdfg = ThreadReduce.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

    # other params
    rowNum = (BlockNum*ThreadPerBlock)//w
    loopNum = (h+rowNum-1)//rowNum

    # Test
    inputs = np.random.rand(h, w)
    outputs = sdfg(H=h, W=w, inputs=inputs, gridDim_x=BlockNum, blockDim_x=ThreadPerBlock, loopNum=loopNum, rowNum=rowNum)
    compared = np.sum(inputs, axis=0)
    assert np.allclose(outputs, compared)

def test_ThreadReduceAlign(h, w, BlockPerColumn, BlockPerRow, ThreadPerBlock):
    # Transform to GPU, keep thread-block map
    sdfg = ThreadReduceAlign.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

    # other params
    loopNum = (h+BlockPerColumn-1)//BlockPerColumn

    # Test
    inputs = np.random.rand(h, w)
    outputs = sdfg(H=h, W=w, inputs=inputs, gridDim_x=BlockPerRow, gridDim_y=BlockPerColumn, blockDim_x=ThreadPerBlock, loopNum=loopNum)
    compared = np.sum(inputs, axis=0)
    assert np.allclose(outputs, compared)

def test_ThreadReduceAlign1(h, w, BlockPerColumn, RowPerBlock):
    # Transform to GPU, keep thread-block map
    sdfg = ThreadReduceAlign1.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

    # other params
    loopNum = (h+BlockPerColumn*RowPerBlock-1)//(BlockPerColumn*RowPerBlock)

    # Test
    inputs = np.random.rand(h, w)
    outputs = sdfg(H=h, W=w, inputs=inputs, gridDim_x=BlockPerColumn, blockDim_x=w, blockDim_y=RowPerBlock, loopNum=loopNum)
    compared = np.sum(inputs, axis=0)
    assert np.allclose(outputs, compared)

def test_BERT(method, size):
    if size == 'small':
        h = 1024
        w = 4096
    elif size == 'large':
        h = 4096
        w = 4096
    elif size == 'test':
        h = 1024*16
        w = 17
    
    if method == 'WRWR':
        BlockPerColumn = 64
        test_WarpReadWarpReduce(h, w, BlockPerColumn)
    elif method == 'WRWRC':
        BlockPerColumn = 64
        test_WarpReadWarpReduceCorner(h, w, BlockPerColumn)
    elif method == 'TR':
        ThreadPerBlock = 256
        BlockNum = 32
        test_ThreadReduce(h ,w, ThreadPerBlock, BlockNum)
    elif method == 'TRA':
        ThreadPerBlock = w
        BlockPerColumn = 256
        BlockPerRow = 1
        test_ThreadReduceAlign(h, w, BlockPerColumn, BlockPerRow, ThreadPerBlock)
    elif method == 'TRA1':
        BlockPerColumn = 32
        RowPerBlock = 8
        test_ThreadReduceAlign1(h, w, BlockPerColumn, RowPerBlock)


        

if __name__ == '__main__':
    test_BERT('WRWR','test')