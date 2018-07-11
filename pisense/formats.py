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

import numpy as np
from PIL import Image

# Make Py2's str and range equivalent to Py3's
native_str = str  # pylint: disable=invalid-name
str = type('')  # pylint: disable=redefined-builtin,invalid-name

color_dtype = np.dtype([  # pylint: disable=invalid-name
    (native_str('r'), np.float32),
    (native_str('g'), np.float32),
    (native_str('b'), np.float32),
])


def check_rgb888(arr):
    if not (
            isinstance(arr, np.ndarray) and
            arr.dtype == np.uint8 and
            len(arr.shape) == 3
            and arr.shape[2] == 3):
        raise ValueError("arr must be a 3-dimensional numpy array of bytes")


def check_rgb565(arr):
    if not (
            isinstance(arr, np.ndarray) and
            arr.dtype == np.uint16 and
            len(arr.shape) == 2):
        raise ValueError("arr must be a 2-dimensional numpy array of "
                         "16-bit ints")


def check_rgb(arr):
    if not (
            isinstance(arr, np.ndarray) and
            arr.dtype == color_dtype and
            len(arr.shape) == 2):
        raise ValueError("arr must be a 2-dimensional numpy array of "
                         "16-bit ints")


def image_to_rgb888(img):
    """
    Convert *img* (an :class:`~PIL.Image.Image`) to RGB888 format in an
    :class:`~numpy.ndarray` with shape (8, 8, 3).
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')
    buf = img.tobytes()
    return np.frombuffer(buf, dtype=np.uint8).reshape(
        (img.size[1], img.size[0], 3)
    )


def image_to_rgb565(img):
    """
    Convert *img* (an :class:`~PIL.Image.Image`) to RGB565 format in an
    :class:`~numpy.ndarray` with shape (8, 8).
    """
    return rgb888_to_rgb565(image_to_rgb888(img))


def image_to_rgb(img):
    """
    Convert *img* (an :class:`~PIL.Image.Image`) to an :class:`~numpy.ndarray`
    in RGB format (structured floating-point type with 3 values each between 0
    and 1).
    """
    return rgb888_to_rgb(image_to_rgb888(img))


def rgb_to_image(arr):
    """
    Convert *arr* (an :class:`~numpy.ndarray` in RGB format, structured
    floating-point type with 3 values each between 0 and 1) to an
    :class:`~PIL.Image.Image`.
    """
    return rgb888_to_image(rgb_to_rgb888(arr))


def rgb888_to_image(arr):
    """
    Convert an :class:`~numpy.ndarray` in RGB888 format (unsigned 8-bit
    values in 3 planes) to an :class:`~PIL.Image.Image`.
    """
    check_rgb888(arr)
    return Image.frombuffer('RGB', (arr.shape[1], arr.shape[0]),
                            arr, 'raw', 'RGB', 0, 1)


def rgb888_to_rgb565(arr, out=None):
    """
    Convert an :class:`~numpy.ndarray` in RGB888 format (unsigned 8-bit values
    in 3 planes) to an :class:`~numpy.ndarray` in RGB565 format (unsigned
    16-bit values with 5 bits for red and blue, and 6 bits for green laid out
    RRRRRGGGGGGBBBBB).
    """
    check_rgb888(arr)
    if out is None:
        out = np.empty(arr.shape[:2], np.uint16)
    else:
        check_rgb565(out)
        if out.shape != arr.shape[:2]:
            raise ValueError("output array has wrong shape")
    out[...] = (
        ((arr[..., 0] & 0xF8).astype(np.uint16) << 8) |
        ((arr[..., 1] & 0xFC).astype(np.uint16) << 3) |
        ((arr[..., 2] & 0xF8).astype(np.uint16) >> 3)
    )
    return out


def rgb565_to_rgb888(arr, out=None):
    """
    Convert an :class:`~numpy.ndarray` in RGB565 format (unsigned 16-bit values
    with 5 bits for red and blue, and 6 bits for green laid out
    RRRRRGGGGGGBBBBB) to an :class:`~numpy.ndarray` in RGB888 format (unsigned
    8-bit values in 3 planes).
    """
    check_rgb565(arr)
    if out is None:
        out = np.empty(arr.shape + (3,), np.uint8)
    else:
        check_rgb888(out)
        if out.shape != arr.shape + (3,):
            raise ValueError("output array has wrong shape")
    out[..., 0] = ((arr & 0xF800) >> 8).astype(np.uint8)
    out[..., 1] = ((arr & 0x07E0) >> 3).astype(np.uint8)
    out[..., 2] = ((arr & 0x001F) << 3).astype(np.uint8)
    # Fill the bottom bits
    out[..., 0] |= out[..., 0] >> 5
    out[..., 1] |= out[..., 1] >> 6
    out[..., 2] |= out[..., 2] >> 5
    return out


def rgb565_to_image(arr):
    """
    Convert an :class:`~numpy.ndarray` in RGB565 format (unsigned 16-bit values
    with 5 bits for red and blue, and 6 bits for green laid out
    RRRRRGGGGGGBBBBB) to an :class:`~PIL.Image.Image`.
    """
    return rgb888_to_image(rgb565_to_rgb888(arr))


def rgb_to_rgb888(arr, out=None):
    """
    Convert a numpy :class:`~numpy.ndarray` in RGB format (structured
    floating-point type with 3 values each between 0 and 1) to RGB888 format
    (unsigned 8-bit values in 3 planes).
    """
    check_rgb(arr)
    if out is None:
        out = np.empty(arr.shape + (3,), np.uint8)
    else:
        check_rgb888(out)
        if out.shape != arr.shape + (3,):
            raise ValueError("output array has wrong shape")
    out[..., 0] = arr['r'] * 255
    out[..., 1] = arr['g'] * 255
    out[..., 2] = arr['b'] * 255
    return out


def rgb888_to_rgb(arr, out=None):
    """
    Convert a numpy :class:`~numpy.ndarray` in RGB888 format (unsigned 8-bit
    values in 3 planes) to RGB format (structured floating-point type with 3
    values, each between 0 and 1).
    1.
    """
    check_rgb888(arr)
    if out is None:
        out = np.empty(arr.shape[:2], color_dtype)
    else:
        check_rgb(out)
        if out.shape != arr.shape[:2]:
            raise ValueError("output array has wrong shape")
    arr = arr.astype(np.float32) / 255
    out['r'] = arr[..., 0]
    out['g'] = arr[..., 1]
    out['b'] = arr[..., 2]
    return out


def rgb_to_rgb565(arr, out=None):
    """
    Convert a numpy :class:`~numpy.ndarray` in RGB format (structured
    floating-point type with 3 values each between 0 and 1) to RGB565 format
    (unsigned 16-bit values with 5 bits for red and blue, and 6 bits for green
    laid out RRRRRGGGGGGBBBBB).
    """
    check_rgb(arr)
    if out is None:
        out = np.zeros(arr.shape, np.uint16)
    else:
        check_rgb565(out)
        if out.shape != arr.shape:
            raise ValueError("output array has wrong shape")
        out[...] = 0
    out |= (arr['r'] * 0x1F).astype(np.uint16) << 11
    out |= (arr['g'] * 0x3F).astype(np.uint16) << 5
    out |= (arr['b'] * 0x1F).astype(np.uint16)
    return out


def rgb565_to_rgb(arr, out=None):
    """
    Convert a numpy :class:`~numpy.ndarray` in RGB565 format (unsigned 16-bit
    values with 5 bits for red and blue, and 6 bits for green laid out
    RRRRRGGGGGGBBBBB) to RGB format (structured  floating-point type with 3
    values each between 0 and 1).
    """
    check_rgb565(arr)
    if out is None:
        out = np.empty(arr.shape, color_dtype)
    else:
        check_rgb(out)
        if out.shape != arr.shape:
            raise ValueError("output array has wrong shape")
    out['r'] = ((arr & 0xF800) / 0xF800).astype(np.float32)
    out['g'] = ((arr & 0x07E0) / 0x07E0).astype(np.float32)
    out['b'] = ((arr & 0x001F) / 0x001F).astype(np.float32)
    return out


def buf_to_rgb888(buf):
    """
    Converts *buf* to a 3-dimensional numpy :class:`~numpy.ndarray` containing
    bytes (RGB888 format). The *buf* parameter can be any of the following
    types:

    * An PIL :class:`~PIL.Image.Image`.

    * An numpy :class:`~numpy.ndarray` with a compatible data-type (the 3-tuple
      of floats used by :class:`ScreenArray`, or simple bytes).

    * A buffer of 192 bytes; each 3 bytes will be taken as RGB levels for
      pixels, working across then down the display.

    The last format is fixed size as a linear buffer has no shape and that is
    the one size we can reasonably guess a shape for. However, the other
    formats are not size limited.
    """
    if isinstance(buf, Image.Image):
        arr = image_to_rgb888(buf)
    elif isinstance(buf, np.ndarray) and 2 <= len(buf.shape) <= 3:
        if len(buf.shape) == 2:
            if buf.dtype == color_dtype:
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
        try:
            arr = np.frombuffer(buf, np.uint8)
        except AttributeError:
            raise TypeError('buf must implement the buffer protocol')
        if len(arr) == 192:
            arr = arr.reshape((8, 8, 3))
        else:
            raise ValueError("buffer must be 8x8 pixels in size")
    return arr


def buf_to_image(buf):
    """
    Converts *buf* to an RGB PIL :class:`~PIL.Image.Image`. The *buf* parameter
    can be any of the types accepted by :func:`buf_to_rgb888`.
    """
    if isinstance(buf, Image.Image):
        if buf.mode != 'RGB':
            return buf.convert('RGB')
        else:
            return buf
    else:
        arr = buf_to_rgb888(buf)
        return Image.frombuffer('RGB', (arr.shape[1], arr.shape[0]),
                                arr, 'raw', 'RGB', 0, 1)


def buf_to_rgb(buf):
    """
    Converts *buf* to a 2-dimensional numpy :class:`~numpy.ndarray` containing
    3-tuples of floats between 0.0 and 1.0 (in other words, the same format as
    :class:`ScreenArray`). The *buf* parameter can be any of the types accepted
    by :func:`buf_to_rgb888`.
    """
    if isinstance(buf, np.ndarray) and len(buf.shape) == 2 and buf.dtype == color_dtype:
        return buf
    else:
        return rgb888_to_rgb(buf_to_rgb888(buf))


def iter_to_rgb(it, shape=(8, 8)):
    """
    Converts *it* (an iterator containing 3-tuples of floats between 0.0 and
    1.0) to a 2-dimensional numpy :class:`~numpy.ndarray` containing the same
    values with the specified *shape*.
    """
    # pylint: disable=invalid-name
    assert len(shape) == 2
    return np.fromiter(it, color_dtype, shape[0] * shape[1]).reshape(shape)
