import torch
import torch.cuda
import pdb
from ._ext import functions

initialized = False

def lgamma(x):
    if x.__class__ == torch.FloatTensor:
        output = torch.FloatTensor(x.size())
        functions.lgamma_py(x, output)
        return output
    elif x.__class__ == torch.DoubleTensor:
        output = torch.DoubleTensor(x.size())
        functions.lgamma_dbl_py(x, output)
        return output
    elif x.__class__ == torch.cuda.FloatTensor:
        output = torch.cuda.FloatTensor(x.size())
        functions.lgamma_cuda(x, output)
        return output
    elif x.__class__ == torch.cuda.DoubleTensor:
        output = torch.cuda.DoubleTensor(x.size())
        functions.lgamma_cuda_dbl(x, output)
        return output
    else:
        raise ValueError

def polygamma(n, x):
    if x.__class__ == torch.FloatTensor:
        output = torch.FloatTensor(x.size())
        functions.polygamma(n, x, output)
        return output
    elif x.__class__ == torch.DoubleTensor:
        output = torch.DoubleTensor(x.size())
        functions.polygamma_dbl(n, x, output)
        return output
    elif x.__class__ == torch.cuda.FloatTensor:
        output = torch.cuda.FloatTensor(x.size())
        functions.polygamma_cuda(n, x, output)
        return output
    elif x.__class__ == torch.cuda.DoubleTensor:
        output = torch.cuda.DoubleTensor(x.size())
        functions.polygamma_cuda_dbl(n, x, output)
        return output
    else:
        raise ValueError

def gamma_sample(a):
    global initialized
    if not initialized:
        functions.init_random()
        initialized = True
    if a.__class__ == torch.cuda.FloatTensor:
        raise ValueError("haven't implemented this yet")
    elif a.__class__ == torch.cuda.DoubleTensor:
        shape = a.size()
        a = a.view(-1, 1)
        msk = (a < 1)
        a_val = a[msk].clone()
        a[msk] += 1
        output = torch.cuda.DoubleTensor(a.size())
        functions.sample_gamma_dbl(a, output)
        if msk.any():
            uniforms = torch.cuda.DoubleTensor(a_val.size()).uniform_().pow(1./a_val)
            output[msk] *= uniforms
        return output.view(*shape)

def beta_sample(a, b):
    if not ((a.__class__ == torch.cuda.DoubleTensor) and (b.__class__ == torch.cuda.DoubleTensor)):
        pdb.set_trace()
    global initialized
    if not initialized:
        functions.init_random()
        initialized = True
    if a.__class__ == torch.cuda.FloatTensor:
        raise ValueError("haven't implemented this yet")
    output = torch.cuda.DoubleTensor(a.size())
    functions.sample_beta_dbl(a, b, output)
    return output
