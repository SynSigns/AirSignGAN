'''
https://github.com/thstkdgus35/bicubic_pytorch/blob/master/core.py

A standalone PyTorch implementation for fast and efficient bicubic resampling.
The resulting values are the same to MATLAB function imresize('bicubic').
## Author:      Sanghyun Son
## Email:       sonsang35@gmail.com (primary), thstkdgus35@snu.ac.kr (secondary)
## Version:     1.1.0
## Last update: July 9th, 2020 (KST)
Depencency: torch
Example::
>>> import torch
>>> import core
>>> x = torch.arange(16).float().view(1, 1, 4, 4)
>>> y = core.imresize(x, sides=(3, 3))
>>> print(y)
tensor([[[[ 0.7506,  2.1004,  3.4503],
          [ 6.1505,  7.5000,  8.8499],
          [11.5497, 12.8996, 14.2494]]]])
'''

import math
import typing

import torch
from torch.nn import functional as F

__all__ = ['imresize']

K = typing.TypeVar('K', str, torch.Tensor)

def cubic_contribution(x: torch.Tensor, a: float=-0.5) -> torch.Tensor:
    ax = x.abs()
    ax2 = ax * ax
    ax3 = ax * ax2

    range_01 = (ax <= 1)
    range_12 = (ax > 1) * (ax <= 2)

    cont_01 = (a + 2) * ax3 - (a + 3) * ax2 + 1
    cont_01 = cont_01 * range_01.to(dtype=x.dtype)

    cont_12 = (a * ax3) - (5 * a * ax2) + (8 * a * ax) - (4 * a)
    cont_12 = cont_12 * range_12.to(dtype=x.dtype)

    cont = cont_01 + cont_12
    cont = cont / cont.sum()
    return cont

def gaussian_contribution(x: torch.Tensor, sigma: float=2.0) -> torch.Tensor:
    range_3sigma = (x.abs() <= 3 * sigma + 1)
    # Normalization will be done after
    cont = torch.exp(-x.pow(2) / (2 * sigma**2))
    cont = cont * range_3sigma.to(dtype=x.dtype)
    return cont

def discrete_kernel(
        kernel: str, scale: float, antialiasing: bool=True) -> torch.Tensor:

    '''
    For downsampling with integer scale only.
    '''
    downsampling_factor = int(1 / scale)
    if kernel == 'cubic':
        kernel_size_orig = 4
    else:
        raise ValueError('Pass!')

    if antialiasing:
        kernel_size = kernel_size_orig * downsampling_factor
    else:
        kernel_size = kernel_size_orig

    if downsampling_factor % 2 == 0:
        a = kernel_size_orig * (0.5 - 1 / (2 * kernel_size))
    else:
        kernel_size -= 1
        a = kernel_size_orig * (0.5 - 1 / (kernel_size + 1))

    with torch.no_grad():
        r = torch.linspace(-a, a, steps=kernel_size)
        k = cubic_contribution(r).view(-1, 1)
        k = torch.matmul(k, k.t())
        k /= k.sum()

    return k

def reflect_padding(
        x: torch.Tensor,
        dim: int,
        pad_pre: int,
        pad_post: int) -> torch.Tensor:

    '''
    Apply reflect padding to the given Tensor.
    Note that it is slightly different from the PyTorch functional.pad,
    where boundary elements are used only once.
    Instead, we follow the MATLAB implementation
    which uses boundary elements twice.
    For example,
    [a, b, c, d] would become [b, a, b, c, d, c] with the PyTorch implementation,
    while our implementation yields [a, a, b, c, d, d].
    '''
    b, c, h, w = x.size()
    if dim == 2 or dim == -2:
        padding_buffer = x.new_zeros(b, c, h + pad_pre + pad_post, w)
        padding_buffer[..., pad_pre:(h + pad_pre), :].copy_(x)
        for p in range(pad_pre):
            padding_buffer[..., pad_pre - p - 1, :].copy_(x[..., p, :])
        for p in range(pad_post):
            padding_buffer[..., h + pad_pre + p, :].copy_(x[..., -(p + 1), :])
    else:
        padding_buffer = x.new_zeros(b, c, h, w + pad_pre + pad_post)
        padding_buffer[..., pad_pre:(w + pad_pre)].copy_(x)
        for p in range(pad_pre):
            padding_buffer[..., pad_pre - p - 1].copy_(x[..., p])
        for p in range(pad_post):
            padding_buffer[..., w + pad_pre + p].copy_(x[..., -(p + 1)])

    return padding_buffer

def padding(
        x: torch.Tensor,
        dim: int,
        pad_pre: int,
        pad_post: int,
        padding_type: str='reflect') -> torch.Tensor:

    if padding_type == 'reflect':
        x_pad = reflect_padding(x, dim, pad_pre, pad_post)
    else:
        raise ValueError('{} padding is not supported!'.format(padding_type))

    return x_pad

def get_padding(
        base: torch.Tensor,
        kernel_size: int,
        x_size: int) -> typing.Tuple[int, int, torch.Tensor]:

    base = base.long()
    r_min = base.min()
    r_max = base.max() + kernel_size - 1

    if r_min <= 0:
        pad_pre = -r_min
        pad_pre = pad_pre.item()
        base += pad_pre
    else:
        pad_pre = 0

    if r_max >= x_size:
        pad_post = r_max - x_size + 1
        pad_post = pad_post.item()
    else:
        pad_post = 0

    return pad_pre, pad_post, base

def get_weight(
        dist: torch.Tensor,
        kernel_size: int,
        kernel: str='cubic',
        sigma: float=2.0,
        antialiasing_factor: float=1) -> torch.Tensor:

    buffer_pos = dist.new_zeros(kernel_size, len(dist))
    for idx, buffer_sub in enumerate(buffer_pos):
        buffer_sub.copy_(dist - idx)

    # Expand (downsampling) / Shrink (upsampling) the receptive field.
    buffer_pos *= antialiasing_factor
    if kernel == 'cubic':
        weight = cubic_contribution(buffer_pos)
    elif kernel == 'gaussian':
        weight = gaussian_contribution(buffer_pos, sigma=sigma)
    else:
        raise ValueError('{} kernel is not supported!'.format(kernel))

    weight /= weight.sum(dim=0, keepdim=True)
    return weight

def reshape_tensor(x: torch.Tensor, dim: int, kernel_size: int) -> torch.Tensor:
    # Resize height
    if dim == 2 or dim == -2:
        k = (kernel_size, 1)
        h_out = x.size(-2) - kernel_size + 1
        w_out = x.size(-1)
    # Resize width
    else:
        k = (1, kernel_size)
        h_out = x.size(-2)
        w_out = x.size(-1) - kernel_size + 1

    unfold = F.unfold(x, k)
    unfold = unfold.view(unfold.size(0), -1, h_out, w_out)
    return unfold

def resize_1d(
        x: torch.Tensor,
        dim: int,
        side: int=None,
        kernel: str='cubic',
        sigma: float=2.0,
        padding_type: str='reflect',
        antialiasing: bool=True) -> torch.Tensor:

    '''
    Args:
        x (torch.Tensor): A torch.Tensor of dimension (B x C, 1, H, W).
        dim (int):
        scale (float):
        side (int):
    Return:
    '''
    scale = side / x.size(dim)
    # Identity case
    if scale == 1:
        return x

    # Default bicubic kernel with antialiasing (only when downsampling)
    if kernel == 'cubic':
        kernel_size = 4
    else:
        kernel_size = math.floor(6 * sigma)

    if antialiasing and (scale < 1):
        antialiasing_factor = scale
        kernel_size = math.ceil(kernel_size / antialiasing_factor)
    else:
        antialiasing_factor = 1

    # We allow margin to both sides
    kernel_size += 2

    # Weights only depend on the shape of input and output,
    # so we do not calculate gradients here.
    with torch.no_grad():
        d = 1 / (2 * side)
        pos = torch.linspace(
            start=d,
            end=(1 - d),
            steps=side,
            dtype=x.dtype,
            device=x.device,
        )
        pos = x.size(dim) * pos - 0.5
        base = pos.floor() - (kernel_size // 2) + 1
        dist = pos - base
        weight = get_weight(
            dist,
            kernel_size,
            kernel=kernel,
            sigma=sigma,
            antialiasing_factor=antialiasing_factor,
        )
        pad_pre, pad_post, base = get_padding(base, kernel_size, x.size(dim))

    # To backpropagate through x
    x_pad = padding(x, dim, pad_pre, pad_post, padding_type=padding_type)
    unfold = reshape_tensor(x_pad, dim, kernel_size)
    # Subsampling first
    if dim == 2 or dim == -2:
        sample = unfold[..., base, :]
        weight = weight.view(1, kernel_size, sample.size(2), 1)
    else:
        sample = unfold[..., base]
        weight = weight.view(1, kernel_size, 1, sample.size(3))

    # Apply the kernel
    down = sample * weight
    down = down.sum(dim=1, keepdim=True)
    return down

def downsampling_2d(
        x: torch.Tensor,
        k: torch.Tensor,
        scale: int,
        padding_type: str='reflect') -> torch.Tensor:

    c = x.size(1)
    k_h = k.size(-2)
    k_w = k.size(-1)

    k = k.to(dtype=x.dtype, device=x.device)
    k = k.view(1, 1, k_h, k_w)
    k = k.repeat(c, c, 1, 1)
    e = torch.eye(c, dtype=k.dtype, device=k.device, requires_grad=False)
    e = e.view(c, c, 1, 1)
    k = k * e

    pad_h = (k_h - scale) // 2
    pad_w = (k_w - scale) // 2
    x = padding(x, -2, pad_h, pad_h, padding_type=padding_type)
    x = padding(x, -1, pad_w, pad_w, padding_type=padding_type)
    y = F.conv2d(x, k, padding=0, stride=scale)
    return y


def timeseries_resize(x: torch.Tensor,scale: float = None,length: int = None):
    """
    Resizes a time series tensor along the time axis using linear interpolation.
    
    Args:
        x (torch.Tensor): A tensor of shape [batch, timesteps, features].
        scale (float, optional): Scaling factor to compute the new number of timesteps.
        length (int, optional): Directly specify the new number of timesteps.
        
    Returns:
        torch.Tensor: The resized time series tensor of shape [batch, new_timesteps, features].
    """
    if scale is None and length is None:
        raise ValueError("Either scale or length must be specified!")
    if scale is not None and length is not None:
        raise ValueError("Specify either scale or length to avoid conflict!")
    
    batch, timesteps, features = x.shape
    new_length = length if length is not None else max(1, math.ceil(timesteps * scale))
    
    # Transpose to [batch, features, timesteps] for F.interpolate.
    x_transposed = x.transpose(1, 2)
    # Interpolate along the timesteps dimension.
    x_resized = F.interpolate(x_transposed, size=new_length, mode='linear', align_corners=False)
    # Transpose back to [batch, timesteps, features].
    x_resized = x_resized.transpose(1, 2)
    return x_resized


# def timeseries_resize(time_series, scale_factor):
#     """
#     downsample a time series using linear interpolation.
#     time_series: Tensor of shape [1, features, timesteps]
#     scale_factor: A float between 0 and 1.
#     returns a resized time series with new number of timesteps.
#     """
#     batch, vector, timesteps = time_series.shape
#     # Compute the new number of timesteps (at least 1)
#     new_length = max(1, int(round(timesteps * scale_factor)))
#     # Use torch's interpolate which works for 3D tensors as well (assuming 1D interpolation)
#     # Here, we treat the time dimension as the last dimension.
#     time_series_resized = F.interpolate(time_series, size=new_length, mode='linear', align_corners=False)
#     return time_series_resized


def imresize(
        x: torch.Tensor,
        scale: float=None,
        sides: typing.Tuple[int, int]=None,
        kernel: K='cubic',
        sigma: float=2,
        rotation_degree: float=0,
        padding_type: str='reflect',
        antialiasing: bool=True) -> torch.Tensor:

    '''
    Args:
        x (torch.Tensor):
        scale (float):
        sides (tuple(int, int)):
        kernel (str, default='cubic'):
        sigma (float, default=2):
        rotation_degree (float, default=0):
        padding_type (str, default='reflect'):
        antialiasing (bool, default=True):
    Return:
        torch.Tensor:
    '''

    if scale is None and sides is None:
        raise ValueError('One of scale or sides must be specified!')
    if scale is not None and sides is not None:
        raise ValueError('Please specify scale or sides to avoid conflict!')

    if x.dim() == 4:
        b, c, h, w = x.size()
    elif x.dim() == 3:
        c, h, w = x.size()
        b = None
    elif x.dim() == 2:
        h, w = x.size()
        b = c = None
    else:
        raise ValueError('{}-dim Tensor is not supported!'.format(x.dim()))

    x = x.view(-1, 1, h, w)

    if sides is None:
        # Determine output size
        sides = (math.ceil(h * scale), math.ceil(w * scale))
        scale_inv = 1 / scale
        if isinstance(kernel, str) and scale_inv.is_integer():
            kernel = discrete_kernel(kernel, scale, antialiasing=antialiasing)
        elif isinstance(kernel, torch.Tensor) and not scale_inv.is_integer():
            raise ValueError(
                'An integer downsampling factor '
                'should be used with a predefined kernel!'
            )

    if x.dtype != torch.float32 or x.dtype != torch.float64:
        dtype = x.dtype
        x = x.float()
    else:
        dtype = None

    if isinstance(kernel, str):
        # Shared keyword arguments across dimensions
        kwargs = {
            'kernel': kernel,
            'sigma': sigma,
            'padding_type': padding_type,
            'antialiasing': antialiasing,
        }
        # Core resizing module
        x = resize_1d(x, -2, side=sides[0], **kwargs)
        x = resize_1d(x, -1, side=sides[1], **kwargs)
    elif isinstance(kernel, torch.Tensor):
        x = downsampling_2d(x, kernel, scale=int(1 / scale))

    rh = x.size(-2)
    rw = x.size(-1)
    # Back to the original dimension
    if b is not None:
        x = x.view(b, c, rh, rw)        # 4-dim
    else:
        if c is not None:
            x = x.view(c, rh, rw)       # 3-dim
        else:
            x = x.view(rh, rw)          # 2-dim

    if dtype is not None:
        if not dtype.is_floating_point:
            x = x.round()
        # To prevent over/underflow when converting types
        if dtype is torch.uint8:
            x = x.clamp(0, 255)

        x = x.to(dtype=dtype)

    return x

if __name__ == '__main__':
    # Just for debugging
    torch.set_printoptions(precision=4, sci_mode=False, edgeitems=16, linewidth=200)
    a = torch.arange(64).float().view(1, 1, 8, 8)
    z = imresize(a, 0.5)
    print(z)
    #a = torch.arange(16).float().view(1, 1, 4, 4)
    '''
    a = torch.zeros(1, 1, 4, 4)
    a[..., 0, 0] = 100
    a[..., 1, 0] = 10
    a[..., 0, 1] = 1
    a[..., 0, -1] = 100
    a = torch.zeros(1, 1, 4, 4)
    a[..., -1, -1] = 100
    a[..., -2, -1] = 10
    a[..., -1, -2] = 1
    a[..., -1, 0] = 100
    '''
    #b = imresize(a, sides=(3, 8), antialiasing=False)
    #c = imresize(a, sides=(11, 13), antialiasing=True)
    #c = imresize(a, sides=(4, 4), antialiasing=False, kernel='gaussian', sigma=1)
    #print(a)
    #print(b)
    #print(c)

    #r = discrete_kernel('cubic', 1 / 3)
    #print(r)
    '''
    a = torch.arange(225).float().view(1, 1, 15, 15)
    imresize(a, sides=[5, 5])
    '''