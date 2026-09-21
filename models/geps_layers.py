import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GEPSConv3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        padding_mode: str = 'constant',
        stride: int = 1,
        code: int = 8,
        factor: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.padding = padding
        self.padding_mode = padding_mode
        self.factor = factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.weight = nn.Parameter(
            torch.empty(
                (out_channels, in_channels, kernel_size, kernel_size, kernel_size),
                **factory_kwargs,
            )
        )
        # A: (in_c, code_c, kT, kH, kW)
        self.A = nn.Parameter(
            torch.empty(in_channels, code, kernel_size, kernel_size, kernel_size)
        )
        # B: (out_c, code_c, kT, kH, kW)
        self.B = nn.Parameter(
            torch.empty(out_channels, code, kernel_size, kernel_size, kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty((out_channels,), **factory_kwargs))
            # bias_context: (code_c, out_c)
            self.bias_context = nn.Parameter(
                torch.empty((code, out_channels), **factory_kwargs)
            )
        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias_context', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
                nn.init.uniform_(self.bias_context, -bound, bound)

    def stride_input(self, inputs, kernel_size, stride):
        batch_size, channels, T, H, W = inputs.shape
        b_stride, c_stride, t_stride, h_stride, w_stride = inputs.stride()

        out_t = int((T - kernel_size) / stride + 1)
        out_h = int((H - kernel_size) / stride + 1)
        out_w = int((W - kernel_size) / stride + 1)

        new_shape = (
            batch_size, channels,
            out_t, out_h, out_w,
            kernel_size, kernel_size, kernel_size,
        )
        new_strides = (
            b_stride, c_stride,
            stride * t_stride, stride * h_stride, stride * w_stride,
            t_stride, h_stride, w_stride,
        )
        return torch.as_strided(inputs, size=new_shape, stride=new_strides)

    def forward(self, input: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        weights = torch.einsum('icpqs, bcr -> birpqs', self.A, codes) # A → (B, in_c, code_c, kT, kH, kW)
        context_weights = torch.einsum('birpqs, orpqs -> boipqs', weights, self.B) # B → (B, out_c, in_c, kT, kH, kW)
        combined_weight = self.weight + self.factor * context_weights
        padded = F.pad(
            input,
            (self.padding, self.padding,   # W
             self.padding, self.padding,   # H
             self.padding, self.padding),  # T
            mode=self.padding_mode,
        )
        B = input.shape[0]
        kT = kH = kW = self.kernel_size
        inputs_reshaped = padded.reshape(
            1, B * self.in_channels, padded.shape[2], padded.shape[3], padded.shape[4]
        )
        weight_reshaped = combined_weight.reshape(
            B * self.out_channels, self.in_channels, kT, kH, kW
        )
        out_flat = F.conv3d(inputs_reshaped, weight_reshaped,
                            bias=None, stride=self.stride, padding=0, groups=B)
        # out_flat: (1, B*out_c, out_t, out_h, out_w)
        out = out_flat.reshape(
            B, self.out_channels, out_flat.shape[2], out_flat.shape[3], out_flat.shape[4]
        )

        if self.bias is not None:
            c_e = torch.diagonal(codes, dim1=-2, dim2=-1)          # (B, code_c)
            context_bias = c_e @ self.bias_context                  # (B, out_c)
            combined_bias = self.bias + self.factor * context_bias  # (B, out_c)
            out = out + combined_bias[:, :, None, None, None]

        return out
