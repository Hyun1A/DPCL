import torch
import torch.nn.functional as F
import numpy as np
import copy
import pdb
from collections import OrderedDict as OD
from collections import defaultdict as DD

torch.random.manual_seed(0)

''' For MIR '''
def overwrite_grad(pp, new_grad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        param.grad=torch.zeros_like(param.data)
        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
        en = sum(grad_dims[:cnt + 1])
        this_grad = new_grad[beg: en].contiguous().view(
            param.data.size())
        param.grad.data.copy_(this_grad)
        cnt += 1

def get_grad_vector(args, pp, grad_dims):
    """
     gather the gradients in one vector
    """
    grads = torch.Tensor(sum(grad_dims))
    if args.cuda: grads = grads.cuda()

    grads.fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1
    return grads

def get_future_step_parameters(this_net, grad_vector, grad_dims, lr=1):
    """
    computes \theta-\delta\theta
    :param this_net:
    :param grad_vector:
    :return:
    """
    new_net=copy.deepcopy(this_net)
    overwrite_grad(new_net.parameters,grad_vector,grad_dims)
    with torch.no_grad():
        for param in new_net.parameters():
            if param.grad is not None:
                param.data=param.data - lr*param.grad.data
    return new_net

def get_grad_dims(self):
    self.grad_dims = []
    for param in self.net.parameters():
        self.grad_dims.append(param.data.numel())

''' Others '''
def onehot(t, num_classes, device='cpu'):
    """
    convert index tensor into onehot tensor
    :param t: index tensor
    :param num_classes: number of classes
    """
    return torch.zeros(t.size()[0], num_classes).to(device).scatter_(1, t.view(-1, 1), 1)

def distillation_KL_loss(y, teacher_scores, T, scale=1, reduction='batchmean'):
    """Computes the distillation loss (cross-entropy).
       xentropy(y, t) = kl_div(y, t) + entropy(t)
       entropy(t) does not contribute to gradient wrt y, so we skip that.
       Thus, loss value is slightly different, but gradients are correct.
       \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
       scale is required as kl_div normalizes by nelements and not batch size.
    """
    return F.kl_div(F.log_softmax(y / T, dim=1), F.softmax(teacher_scores / T, dim=1),
            reduction=reduction) * scale

def naive_cross_entropy_loss(input, target, size_average=True):
    """
    in PyTorch's cross entropy, targets are expected to be labels
    so to predict probabilities this loss is needed
    suppose q is the target and p is the input
    loss(p, q) = -\sum_i q_i \log p_i
    """
    assert input.size() == target.size()
    input = torch.log(F.softmax(input, dim=1).clamp(1e-5, 1))
    # input = input - torch.log(torch.sum(torch.exp(input), dim=1)).view(-1, 1)
    loss = - torch.sum(input * target)
    return loss / input.size()[0] if size_average else loss

def compute_offsets(task, nc_per_task, is_cifar):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2

def out_mask(t, nc_per_task, n_outputs):
    # make sure we predict classes within the current task
    offset1 = int(t * nc_per_task)
    offset2 = int((t + 1) * nc_per_task)
    if offset1 > 0:
        output[:, :offset1].data.fill_(-10e10)
    if offset2 < self.n_outputs:
        output[:, offset2:n_outputs].data.fill_(-10e10)

class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(input.size(0), *self.shape)

