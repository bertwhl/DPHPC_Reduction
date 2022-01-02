import cupy as cp
import torch
import jax
import numpy as np
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
    test_num = 1
    for i in range(test_num):
        inputs = np.random.rand(1000)
        axis = (0)
        test_all(inputs, axis)
        print("test {} done".format(i))