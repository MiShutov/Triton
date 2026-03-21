from typing import Optional
import torch
import torch.nn as nn

from matmul_fp8_fp8 import alloc_and_quant, scaled_matmul_fp8_fp8


def matmul_autocast_fp8(a, b, transpose_right=False):
    a_fp8, a_scales = alloc_and_quant(a)
    b_fp8, b_scales = alloc_and_quant(b)
    if transpose_right:
        return scaled_matmul_fp8_fp8(a_fp8, b_fp8.T, a_scales, b_scales.T.contiguous())
    return scaled_matmul_fp8_fp8(a_fp8, b_fp8, a_scales, b_scales)


class Fp8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        ctx.save_for_backward(input, weight, bias)

        # output = input @ weight.T
        output = matmul_autocast_fp8(input, weight, transpose_right=True)
        
        if bias is not None:
            output += bias

        return output
    

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight, bias = ctx.saved_tensors
        
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            # grad_input = grad_output @ weight
            grad_input = matmul_autocast_fp8(grad_output, weight)

        if ctx.needs_input_grad[1]:
            # grad_input = grad_output.T @ input
            grad_weight = matmul_autocast_fp8(grad_output.T.contiguous(), input)
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)

        return grad_input, grad_weight, grad_bias


class Fp8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return Fp8LinearFunction.apply(input, self.weight, self.bias)
    
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'