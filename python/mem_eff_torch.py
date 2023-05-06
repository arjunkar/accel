"""
Custom memory-efficient forward and backward passes for linear layer.
"""

import sys
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity


class LinearFunctional(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, w, b):
        ctx.save_for_backward(input, w, b)
        with torch.no_grad():
            # Avoids the creation of matmul(input, w) in the computational graph
            output = torch.matmul(input, w) + b
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, w, b = ctx.saved_tensors
        # Detach saved tensors from old graph
        input, w, b = (input.detach().requires_grad_(True), 
                       w.detach().requires_grad_(True), 
                       b.detach().requires_grad_(True)
        )
        # Now track gradients by rerunning forward pass
        with torch.enable_grad():
            output = torch.matmul(input, w) + b
        # Use torch autograd instead of explicit Jacobian-vector product formula
        torch.autograd.backward((output,), grad_output, 
                                retain_graph=False, 
                                inputs=(input, w, b))
        return input.grad, w.grad, b.grad


class Linear(nn.Module):
    def __init__(self, dim_in, dim_out) -> None:
        super().__init__()
        w = torch.empty(size=(dim_in, dim_out))
        b = torch.empty(size=(dim_out,))
        nn.init.uniform_(w, a=-1/dim_out, b=1/dim_out)
        nn.init.uniform_(b, -1/dim_out)
        self.W = nn.Parameter(w, requires_grad=True)
        self.b = nn.Parameter(b, requires_grad=True)

    def forward(self, input):
        linear_func = LinearFunctional.apply
        return linear_func(input, self.W, self.b)


# Profiling memory efficiency vs standard nn.Linear
class Model(nn.Module):
    def __init__(self, layer, num_layers, dim) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [layer(dim, dim) for _ in range(num_layers)]
        )

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input

num_layers = 50
dim = 5

ref_model = Model(nn.Linear, num_layers, dim)
my_model = Model(Linear, num_layers, dim)
input = torch.randn(size=(dim,), requires_grad=True)

with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
    with record_function("ref_forward"):
        ref_output = ref_model(input)
    with record_function("my_forward"):
        my_output = my_model(input)
    with record_function("ref_backward"):
        ref_output.backward(gradient=torch.randn_like(ref_output))
    with record_function("my_backward"):
        my_output.backward(gradient=torch.randn_like(my_output))

with open('./accel/python/out.txt', 'w') as sys.stdout:
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=20))