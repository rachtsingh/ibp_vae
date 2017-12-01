# utils for hacking around
import torch
from torch.autograd import Variable
import shutil
import pdb
import uuid

def show_memusage(device=1):
    import gpustat
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("{}/{}".format(item["memory.used"], item["memory.total"]))

SMALL = 1e-16
EULER_GAMMA = 0.5772156649015329

try:
    input = raw_input
except NameError:
    pass

def binarize(image):
    return image.bernoulli()

def save_dataset(state, filename):
    name = '{}_data.pth.tar'.format(filename)
    torch.save(state, name)

def save_checkpoint(state, is_best, filename='checkpoint'):
    name = '{}.pth.tar'.format(filename)
    torch.save(state, name)
    if is_best:
        best_name = '{}_best.pth.tar'.format(filename)
        shutil.copyfile(name, best_name)

def print_grad(name):
    def hook(grad):
        if math.isnan(grad.sum().data[0]):
            print("grad failed for: {}".format(name))
            pdb.set_trace()
        # print("dL/d{}: {}".format(name, grad.sum()))
    return hook

def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.97 ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def gen_id():
    return str(uuid.uuid4()).replace("-", "")

if torch.cuda.is_available():
    newTensor = torch.cuda.DoubleTensor
else:
    newTensor = torch.DoubleTensor

def isnan(v):
    if v.__class__ == Variable:
        v = v.data
    return np.isnan(v.cpu().numpy()).any()

def findnan(v):
    if v.__class__ == Variable:
        v = v.data
    return np.isnan(v.cpu().numpy())

# crucial
def logit(x):
    return (x + SMALL).log() - (1. - x + SMALL).log()
