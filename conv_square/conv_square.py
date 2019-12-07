# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:58:19 2019

@author: lijingxiong


Reference: https://github.com/NVlabs/pacnet/blob/master/pac.py
"""

import math
import torch
        
        
class KernelConvFn(torch.autograd.function.Function):
    """2D convolution with kernel.
    
    Copy from: https://github.com/NVlabs/pacnet/blob/master/pac.py/PacConv2dFn
    """
        
    @staticmethod
    def forward(ctx, inputs, kernel, weight, bias=None, stride=1, padding=0, 
                dilation=1):
        """Forward computation.
        
        Args:
            inputs: A tensor with shape [batch, channels, height, width] 
                representing a batch of images.
            kernel: A tensor with shape [batch, channels, k, k, N, N],
                where k = kernel_size and N = number of slide windows.
            weight: A tensor with shape [out_channels, in_channels, 
                kernel_size, kernel_size].
            bias: None or a tenor with shape [out_channels].
            
        Returns:
            outputs: A tensor with shape [batch, out_channels, height, width].
        """
        (batch_size, channels), input_size = inputs.shape[:2], inputs.shape[2:]
        ctx.in_channels = channels
        ctx.input_size = input_size
        ctx.kernel_size = tuple(weight.shape[-2:])
        ctx.dilation = torch.nn.modules.utils._pair(dilation)
        ctx.padding = torch.nn.modules.utils._pair(padding)
        ctx.stride = torch.nn.modules.utils._pair(stride)
        
        needs_input_grad = ctx.needs_input_grad
        ctx.save_for_backward(
            inputs if (needs_input_grad[1] or needs_input_grad[2]) else None,
            kernel if (needs_input_grad[0] or needs_input_grad[2]) else None,
            weight if (needs_input_grad[0] or needs_input_grad[1]) else None)
        ctx._backend = torch._thnn.type2backend[inputs.type()]
        
        # Slide windows, [batch, channels x kernel_size x kernel_size, N x N],
        # where N is the number of slide windows.
        inputs_wins = torch.nn.functional.unfold(inputs, ctx.kernel_size, 
                                                 ctx.dilation, ctx.padding,
                                                 ctx.stride)

        inputs_mul_kernel = inputs_wins.view(
            batch_size, channels, *kernel.shape[2:]) * kernel
                
        # Matrix multiplication
        outputs = torch.einsum('ijklmn,ojkl->iomn', (inputs_mul_kernel, weight))
        
        if bias is not None:
            outputs += bias.view(1, -1, 1, 1)
        return outputs
        
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_outputs):
        grad_inputs = grad_kernel = grad_weight = grad_bias = None
        batch_size, out_channels = grad_outputs.shape[:2]
        output_size = grad_outputs.shape[2:]
        in_channels = ctx.in_channels
        
        # Compute gradients
        inputs, kernel, weight = ctx.saved_tensors
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_inputs_mul_kernel = torch.einsum('iomn,ojkl->ijklmn',
                                                  (grad_outputs, weight))
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            inputs_wins = torch.nn.functional.unfold(inputs, ctx.kernel_size, 
                                                     ctx.dilation, ctx.padding,
                                                     ctx.stride)
            inputs_wins = inputs_wins.view(batch_size, in_channels,
                                           ctx.kernel_size[0], 
                                           ctx.kernel_size[1],
                                           output_size[0], output_size[1])
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.new()
            grad_inputs_wins = grad_inputs_mul_kernel * kernel
            grad_inputs_wins = grad_inputs_wins.view(
                batch_size, -1, output_size[0] * output_size[1])
            grad_inputs = torch.nn.functional.fold(grad_inputs_wins,
                                                   ctx.input_size,
                                                   ctx.kernel_size,
                                                   ctx.dilation,
                                                   ctx.padding,
                                                   ctx.stride)
        if ctx.needs_input_grad[1]:
            grad_kernel = inputs_wins * grad_inputs_mul_kernel
            grad_kernel = grad_kernel.sum(dim=1, keepdim=True)
        if ctx.needs_input_grad[2]:
            inputs_mul_kernel = inputs_wins * kernel
            grad_weight = torch.einsum('iomn,ijklmn->ojkl',
                                       (grad_outputs, inputs_mul_kernel))
        if ctx.needs_input_grad[3]:
            grad_bias = torch.einsum('iomn->o', (grad_outputs,))
        return (grad_inputs, grad_kernel, grad_weight, grad_bias, None, None, 
                None)
        
        
class SquareKernelFn(torch.autograd.function.Function):
    """Compute kernel for square funcion."""
    
    @staticmethod
    def forward(ctx, alpha, a, b, c, kernel_size, stride, padding, dilation):
        """Forward computation.
        
        Implementation of computation: ax^2 + bx + c.
        
        Args:
            alpha: A tensor with shape [batch, 1, height, width] representing
                a batch of depth maps.
            a, b, c: Scalars.
            
        Returns:
            A tensor with shape [batch, 1, k, k, N, N], where 
            k = kernel_size and N = number of slide windows.
        """
        ctx.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        ctx.stride = torch.nn.modules.utils._pair(stride)
        ctx.padding = torch.nn.modules.utils._pair(padding)
        ctx.dilation = torch.nn.modules.utils._pair(dilation)
        ctx.scalars = (a, b, c)
        
        needs_grad = ctx.needs_input_grad
        needs_grad = needs_grad[0] or needs_grad[1] or needs_grad[2]
        ctx.save_for_backward(alpha if needs_grad else None)
        ctx._backend = torch._thnn.type2backend[alpha.type()]
        
        batch_size, channels, in_height, in_width = alpha.shape
        out_height = (in_height + 2 * ctx.padding[0] - 
                      ctx.dilation[0] * (ctx.kernel_size[0] - 1)
                      -1) // ctx.stride[0] + 1
        out_width = (in_width + 2 * ctx.padding[1] - 
                     ctx.dilation[1] * (ctx.kernel_size[1] - 1)
                     -1) // ctx.stride[1] + 1
        
        alpha_wins = torch.nn.functional.unfold(alpha, ctx.kernel_size,
                                                ctx.dilation, ctx.padding,
                                                ctx.stride)
        alpha_wins = alpha_wins.view(batch_size, channels, ctx.kernel_size[0],
                                     ctx.kernel_size[1], out_height, 
                                     out_width)
        
        square_alpha_wins = (a * alpha_wins + b) * alpha_wins + c
        return square_alpha_wins
    
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_outputs):
        grad_alpha = grad_a = grad_b = grad_c = None
        batch_size, out_channels = grad_outputs.shape[:2]
        output_size = grad_outputs.shape[-2:]
        
        # Compute gradients
        a, b, c = ctx.scalars
        alpha = ctx.saved_tensors[0]
        _, in_channels, in_height, in_width = alpha.shape
        needs_input_grad = ctx.needs_input_grad
        if needs_input_grad[0] or needs_input_grad[1] or needs_input_grad[2]:
            grad_alpha = grad_outputs.new()
            alpha_wins = torch.nn.functional.unfold(alpha, ctx.kernel_size,
                                                ctx.dilation, ctx.padding,
                                                ctx.stride)
            alpha_wins = alpha_wins.view(batch_size, 
                                         in_channels, 
                                         ctx.kernel_size[0],
                                         ctx.kernel_size[1], 
                                         output_size[0], 
                                         output_size[1])
        if needs_input_grad[0]:
            grad_alpha_wins = (2 * a * alpha_wins + b) * grad_outputs
            grad_alpha_wins = grad_alpha_wins.view(
                batch_size, -1, output_size[0] * output_size[1])
            grad_alpha = torch.nn.functional.fold(grad_alpha_wins,
                                                  (in_height, in_width),
                                                  ctx.kernel_size,
                                                  ctx.dilation,
                                                  ctx.padding,
                                                  ctx.stride)
        if ctx.needs_input_grad[1]:
            grad_a = alpha_wins * alpha_wins * grad_outputs
            grad_a = torch.einsum('ijklmn->', (grad_a,))
        if ctx.needs_input_grad[2]:
            grad_b = alpha_wins * grad_outputs
            grad_b = torch.einsum('ijklmn->', (grad_b,))
        if ctx.needs_input_grad[3]:
            grad_c = torch.einsum('ijklmn->', (grad_outputs,))
        
        return grad_alpha, grad_a, grad_b, grad_c, None, None, None, None
    
    
class ConvSquare(torch.nn.Module):
    """Implementation of square weighted convolution."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, a=None, b=None, c=None):
        """Constructor."""
        super(ConvSquare, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)
        self.padding = torch.nn.modules.utils._pair(padding)
        self.dilation = torch.nn.modules.utils._pair(dilation)
        
        # Parameters: weight, bias
        self.weight = torch.nn.parameter.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size,
                         kernel_size))
        if bias:
            self.bias = torch.nn.parameter.Parameter(
                torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Scalars
        self.a = a
        self.b = b
        self.c = c
            
        # Initialization
        self.reset_parameters()
        
    def forward(self, inputs, alpha):
        """Forward computation.
        
        Args:
            inputs: A tensor with shape [batch, in_channels, height, width] 
                representing a batch of images.
            alpha: A tensor with shape [batch, 1, height, width] representing
                    a batch of depth maps.
            
        Returns:
            outputs: A tensor with shape [batch, out_channels, height, width].
        """
        kernel = SquareKernelFn.apply(alpha, self.a, self.b, self.c, 
                                      self.kernel_size, self.stride,
                                      self.padding, self.dilation)
        
        outputs = KernelConvFn.apply(inputs, kernel, self.weight,
                                     self.bias, self.stride,
                                     self.padding, self.dilation)
        return outputs
    
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        return s.format(**self.__dict__)
    
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
            
        if self.a is None:
            self.a = torch.nn.parameter.Parameter(torch.Tensor())
            self.a.data = torch.tensor(-4.)
        if self.b is None:
            self.b = torch.nn.parameter.Parameter(torch.Tensor())
            self.b.data = torch.tensor(4.)
        if self.c is None:
            self.c = torch.nn.parameter.Parameter(torch.Tensor())
            self.c.data = torch.tensor(1.)
                