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

def test_BERT(method, size):
    if size == 'special_test':
        h = 4096*256
        w = 32*4
        if method == 'WRWR':
            BlockPerColumn = 16
            test_WarpReadWarpReduce(h, w, BlockPerColumn)
        if method == 'TR':
            ThreadPerBlock = 512
            BlockNum = 64
            test_ThreadReduce(h ,w, ThreadPerBlock, BlockNum)

    if size == 'small':
        h = 1024
        w = 4096
    elif size == 'large':
        h = 4096
        w = 4096
    if method == 'WRWR':
        BlockPerColumn = 1
        test_WarpReadWarpReduce(h, w, BlockPerColumn)
    if method == 'TR':
        ThreadPerBlock = 256
        BlockNum = 128
        test_ThreadReduce(h ,w, ThreadPerBlock, BlockNum)
        

if __name__ == '__main__':
    test_BERT('TR','special_test')