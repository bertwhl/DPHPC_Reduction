import cupy as cp
import torch
import jax
import numpy as np
import sys
sys.path.insert(0, "/home/zhoubo/DPHPC_Reduction/library")
from Reduce import Reduce

def _test_cupy(inputs, axis):
    cupy_input = cp.array(inputs)
    cupy_output = cp.sum(cupy_input, axis=axis)
    return cupy_output


def _test_torch(inputs, axis):
    torch.set_default_tensor_type(torch.DoubleTensor)
    torch_input = torch.tensor(inputs)
    torch_input = torch_input.cuda()
    torch_output = torch.sum(torch_input,dim=axis).cuda()
    torch_output = torch_output.cpu()
    return torch_output


def _test_jax(inputs, axis):
    jax.config.update("jax_platform_name", "gpu")
    jax.config.update("jax_enable_x64", True)
    jax_output = jax.numpy.sum(inputs, axis=axis)
    return jax_output

def test_all(inputs, axis):

    # Testing competitors
    cupy_output = _test_cupy(inputs, axis)
    torch_output = _test_torch(inputs, axis)
    jax_output = _test_jax(inputs, axis)

    output = Reduce(inputs, axis)

    assert np.allclose(output, cupy_output)
    assert np.allclose(output, jax_output)
    assert np.allclose(output, torch_output)

if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     raise Exception('only accept 2 arguments')
    test_case = sys.argv[1]
    input_case = sys.argv[2]

    if input_case == "1":
        inputs = np.random.rand(102857600)
        axis = (0)
    elif input_case == "2":
        inputs = np.random.rand(8, 512, 1024)
        axis = (0,1)
    elif input_case == "3":
        inputs = np.random.rand(8, 512, 4096)
        axis = (0,1)
    elif input_case == "4":
        inputs = np.random.rand(8, 32, 224, 224)
        axis = (2,3)
    elif input_case == "5":
        inputs = np.random.rand(12, 10000, 12)
        axis = (1)
    elif input_case == "6":
        inputs = np.random.rand(12, 12, 10000)
        axis = (2)
    else:
        inputs = np.random.rand(int(input_case))
        axis = (0)

    test_num = 1
    for i in range(1):
        if test_case ==  "1":
            cupy_output = _test_cupy(inputs, axis)
        elif test_case == "2":
            torch_output = _test_torch(inputs, axis)
        elif test_case == "3":
            jax_output = _test_jax(inputs, axis)
        elif test_case == "4":
            output = Reduce(inputs, axis)
        elif test_case == "5":
            test_all()
        else:
            raise Exception('invalid case number, only accept 1, 2, 3, 4, 5')