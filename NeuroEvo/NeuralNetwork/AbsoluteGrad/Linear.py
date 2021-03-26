import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MulConstant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, w):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(w)
        return F.linear(tensor, w)

    @staticmethod
    def backward(ctx, grad_output):
        # Here we must handle None grad_output tensor. In this case we
        # can skip unnecessary computations and just return None.
        if grad_output is None:
            return None, None

        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        w = ctx.saved_tensors[0]
        return grad_output * w, grad_output * w

class GrowingLinear(nn.Module):
    def __init__(self, D_in, D_out):
        super(GrowingLinear, self).__init__()
        self.linIn = nn.Linear(D_in, 1)
        self.linHidden = nn.Linear(1,D_out)

    def forward(self, x):
        x = self.linIn(x)

        x = F.relu(x)
        x = self.linHidden(x)

        return x

def AbsGradTest():
    x = torch.tensor([10., 10.])
    w = torch.tensor([[1., 2.],
                      [2., 1.]], requires_grad=True).t()
    w2 = torch.tensor([[1., 2.],
                      [2., 1.]], requires_grad=True).t()

    MulConst = MulConstant.apply
    print("x: " + str(x))
    y = MulConst(x, w)
    print("y: " + str(y))
    y = MulConst(y,w2)

    print("y2: " + str(y))
    w.retain_grad()
    y.sum().backward()

    print(w.grad, w.requires_grad)
    print(w2.grad, w2.requires_grad)
    with torch.no_grad():
        w -= w.grad
        w.grad.data.zero_()
        w2 -= w.grad
        w2.grad.data.zero_()

    y = MulConst(x,w)
    y = MulConst(y,w2)
    y.mean().backward()
    print(y)
    print(w.grad, w.requires_grad)
    print(w2.grad, w2.requires_grad)
