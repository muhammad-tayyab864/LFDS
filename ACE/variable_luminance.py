"""
Source:
https://github.com/kornia/kornia/blob/master/kornia/color/ycbcr.py 
"""

import torch

def rgb_to_ycbcr(image: torch.Tensor, luminance_const: list = [0.299, 0.587, 0.114]) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.

    .. image:: _static/img/rgb_to_ycbcr.png

    Args:
        image: RGB Image to be converted to YCbCr with shape :math:`(*, 3, H, W)`.
        luminance_const: The constant to convert RGB to YCbCr (kr, kg, kb)

    Returns:
        YCbCr version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_ycbcr(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    kr, kg, kb = luminance_const

    delta: float = 0.5
    y: torch.Tensor = kr * r + kg * g + kb * b
    cb: torch.Tensor = (b - y) * (1 / ((1 - kb)*2)) + delta
    cr: torch.Tensor = (r - y) * (1 / ((1 - kr)*2)) + delta
    return torch.stack([y, cb, cr], -3)

def ycbcr_to_rgb(image: torch.Tensor, luminance_const: list = [0.299, 0.587, 0.114]) -> torch.Tensor:
    r"""Convert an YCbCr image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: YCbCr Image to be converted to RGB with shape :math:`(*, 3, H, W)`.
        luminance_const: The constant to convert RGB to YCbCr (kr, kg, kb)

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = ycbcr_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    y: torch.Tensor = image[..., 0, :, :]
    cb: torch.Tensor = image[..., 1, :, :]
    cr: torch.Tensor = image[..., 2, :, :]

    delta: float = 0.5
    cb_shifted: torch.Tensor = cb - delta
    cr_shifted: torch.Tensor = cr - delta

    kr, kg, kb = luminance_const

    r: torch.Tensor = y + ((1 - kr)*2) * cr_shifted
    b: torch.Tensor = y + ((1 - kb)*2) * cb_shifted
    g: torch.Tensor = (y - r * kr - b * kb) * (1 / kg)
    return torch.stack([r, g, b], -3)
