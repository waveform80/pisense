# vim: set et sw=4 sts=4 fileencoding=utf-8:
#
# Alternative API for the Sense HAT
# Copyright (c) 2016-2018 Dave Jones <dave@waveform.org.uk>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the copyright holder nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Defines a set of conversions between buffers, PIL images, and numpy array
formats used in pisense.
"""

from __future__ import (
    unicode_literals,
    absolute_import,
    print_function,
    division,
)
native_str = str
str = type('')


import numpy as np
from PIL import Image


color = np.dtype([
    (native_str('r'), np.float16),
    (native_str('g'), np.float16),
    (native_str('b'), np.float16),
])


def image_to_rgb888(img):
    """
    Convert *img* (a PIL :class:`Image`) to RGB888 format in a numpy
    :class:`ndarray` with shape (8, 8, 3).
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')
    try:
        buf = img.tobytes()
    except AttributeError:
        # Ooooooold PIL
        buf = img.tostring()
    return np.frombuffer(buf, dtype=np.uint8).reshape(
        (img.size[1], img.size[0], 3)
    )


def image_to_rgb565(img):
    """
    Convert *img* (a PIL :class:`Image`) to RGB565 format in a numpy
    :class:`ndarray` with shape (8, 8).
    """
    return rgb888_to_rgb565(image_to_rgb888(img))


def image_to_rgb(img):
    """
    Convert *img* (a PIL:class:`Image`) to a numpy :class:`ndarray` in RGB
    format (structured floating-point type with 3 values each between 0 and 1).
    """
    return rgb888_to_rgb(image_to_rgb888(img))


def rgb888_to_image(arr):
    """
    Convert a numpy :class:`ndarray` in RGB888 format (unsigned 8-bit values in
    3 planes) to a PIL :class:`Image`.
    """
    # XXX Change to exception
    assert arr.dtype == np.uint8 and len(arr.shape) == 3 and arr.shape[2] == 3
    return Image.frombuffer('RGB', (arr.shape[1], arr.shape[0]),
                            arr, 'raw', 'RGB', 0, 1)


def rgb888_to_rgb565(arr, out=None):
    if out is None:
        out = np.empty(arr.shape[:2], np.uint16)
    else:
        # XXX Change to exception
        assert out.shape == arr.shape[:2] and out.dtype == np.uint16
    out[...] = (
        ((arr[..., 0] & 0xF8).astype(np.uint16) << 8) |
        ((arr[..., 1] & 0xFC).astype(np.uint16) << 3) |
        ((arr[..., 2] & 0xF8).astype(np.uint16) >> 3)
    )
    return out


def rgb565_to_rgb888(arr, out=None):
    if out is None:
        out = np.empty(arr.shape + (3,), np.uint8)
    else:
        # XXX Change to exception
        assert out.shape == arr.shape + (3,) and out.dtype == np.uint8
    out[..., 0] = ((arr & 0xF800) >> 8).astype(np.uint8)
    out[..., 1] = ((arr & 0x07E0) >> 3).astype(np.uint8)
    out[..., 2] = ((arr & 0x001F) << 3).astype(np.uint8)
    # Fill the bottom bits
    out[..., 0] |= out[..., 0] >> 5
    out[..., 1] |= out[..., 1] >> 6
    out[..., 2] |= out[..., 2] >> 5
    return out


def rgb565_to_image(arr):
    return rgb888_to_image(rgb565_to_rgb888(arr))


def rgb_to_rgb888(arr, out=None):
    """
    Convert a numpy :class:`ndarray` in RGB format (structured floating-point
    type with 3 values each between 0 and 1) to RGB888 format (unsigned 8-bit
    values in 3 planes).
    """
    if out is None:
        out = np.empty(arr.shape + (3,), np.uint8)
    else:
        # XXX Change to exception
        assert out.shape == arr.shape + (3,) and out.dtype == np.uint8
    out[..., 0] = arr['r'] * 255
    out[..., 1] = arr['g'] * 255
    out[..., 2] = arr['b'] * 255
    return out


def rgb888_to_rgb(arr, out=None):
    """
    Convert a numpy :class:`ndarray` in RGB888 format (unsigned 8-bit values
    in 3 planes) to RGB format (structured floating-point type with 3 values,
    each between 0 and 1).
    1.
    """
    if out is None:
        out = np.empty(arr.shape[:2], color)
    else:
        # XXX Change to exception
        assert out.shape == arr.shape[:2] and out.dtype == color
    arr = arr.astype(np.float16) / 255
    out['r'] = arr[..., 0]
    out['g'] = arr[..., 1]
    out['b'] = arr[..., 2]
    return out


def rgb_to_rgb565(arr, out=None):
    """
    Convert a numpy :class:`ndarray` in RGB format (structured floating-point
    type with 3 values each between 0 and 1) to RGB565 format (unsigned 16-bit
    values with 5 bits for red and blue, and 6 bits for green laid out
    RRRRRGGGGGGBBBBB).
    """
    if out is None:
        out = np.zeros(arr.shape, np.uint16)
    else:
        # XXX Change to exception
        assert out.shape == arr.shape and out.dtype == np.uint16
        out[...] = 0
    out |= (arr['r'] * 0x1F).astype(np.uint16) << 11
    out |= (arr['g'] * 0x3F).astype(np.uint16) << 5
    out |= (arr['b'] * 0x1F).astype(np.uint16)
    return out


def rgb565_to_rgb(arr, out=None):
    """
    Convert a numpy :class:`ndarray` in RGB565 format (unsigned 16-bit values
    with 5 bits for red and blue, and 6 bits for green laid out
    RRRRRGGGGGGBBBBB) to RGB format (structured  floating-point type with 3
    values each between 0 and 1).
    """
    if out is None:
        out = np.empty(arr.shape, color)
    else:
        assert out.shape == arr.shape and out.dtype == color
    out['r'] = ((arr & 0xF800) / 0xF800).astype(np.float16)
    out['g'] = ((arr & 0x07E0) / 0x07E0).astype(np.float16)
    out['b'] = ((arr & 0x001F) / 0x001F).astype(np.float16)
    return out


def buf_to_image(buf):
    if isinstance(buf, Image.Image):
        if buf.mode != 'RGB':
            return buf.convert('RGB')
        else:
            return buf
    else:
        arr = buf_to_rgb888(buf)
        return Image.frombuffer('RGB', (arr.shape[1], arr.shape[0]),
                                arr, 'raw', 'RGB', 0, 1)


def buf_to_rgb888(buf):
    if isinstance(buf, Image.Image):
        arr = image_to_rgb888(buf)
    elif isinstance(buf, np.ndarray) and 2 <= len(buf.shape) <= 3:
        if len(buf.shape) == 2:
            if buf.dtype == color:
                arr = rgb_to_rgb888(buf)
            elif buf.dtype == np.uint8:
                arr = np.dstack((buf, buf, buf))
            else:
                raise ValueError("can't coerce dtype %s to uint8" % buf.dtype)
        else:
            if buf.dtype != np.uint8:
                raise ValueError("can't coerce dtype %s to uint8" % buf.dtype)
            arr = buf
    else:
        arr = np.frombuffer(buf, np.uint8)
        if len(arr) == 192:
            arr = arr.reshape((8, 8, 3))
        elif len(arr) == 64:
            arr = arr.reshape((8, 8))
            arr = np.dstack((arr, arr, arr)) # me hearties!
        else:
            raise ValueError("buffer must be 8x8 pixels in size")
    return arr


def buf_to_rgb(buf):
    if isinstance(buf, np.ndarray) and buf.dtype == color:
        return buf
    else:
        arr = buf_to_rgb888(buf)
        return arr.astype(np.float16) / 255
