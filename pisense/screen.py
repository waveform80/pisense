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
Defines the :class:`SenseArray` and :class:`SenseScreen` classes for
controlling and manipulating the RGB pixel array on the Sense HAT.
"""

from __future__ import (
    unicode_literals,
    absolute_import,
    print_function,
    division,
)
native_str = str
str = type('')


import io
import os
import glob
import mmap
import errno
import time

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from colorzero import Color

from .easings import linear


color_dtype = np.dtype([
    (native_str('r'), np.float16),
    (native_str('g'), np.float16),
    (native_str('b'), np.float16),
])


class SenseArray(np.ndarray):
    # pylint: disable=too-few-public-methods

    def __new__(cls):
        # pylint: disable=protected-access
        result = np.ndarray.__new__(cls, shape=(8, 8), dtype=color_dtype)
        result._screen = None
        return result

    def __array_finalize__(self, obj):
        # pylint: disable=protected-access,attribute-defined-outside-init
        if obj is None:
            return
        self._screen = getattr(obj, '_screen', None)

    def __setitem__(self, index, value):
        # pylint: disable=protected-access
        super(SenseArray, self).__setitem__(index, value)
        if self._screen:
            # If we're a slice of the original pixels value, find the parent
            # that contains the complete array and send that to _set_pixels
            orig = self
            while orig.shape != (8, 8) and orig.base is not None:
                orig = orig.base
            self._screen._set_pixels(orig)

    def __setslice__(self, i, j, sequence):
        # pylint: disable=protected-access
        super(SenseArray, self).__setslice__(i, j, sequence)
        if self._screen:
            orig = self
            while orig.shape != (8, 8) and orig.base is not None:
                orig = orig.base
            self._screen._set_pixels(orig)

    def copy(self, order='C'):
        result = super(SenseArray, self).copy(order)
        result._screen = None
        return result


class SenseScreen(object):
    # pylint: disable=too-many-instance-attributes
    SENSE_HAT_FB_NAME = 'RPi-Sense FB'

    def __init__(self):
        self._fb_file = io.open(self._fb_device(), 'wb+')
        self._fb_mmap = mmap.mmap(self._fb_file.fileno(), 128)
        self._fb_array = np.frombuffer(self._fb_mmap, dtype=np.uint16).reshape((8, 8))
        self._array = SenseArray()
        self._array._screen = self
        self._hflip = False
        self._vflip = False
        self._rotation = 0
        self._font_cache = {}

    def close(self):
        self._fb_array = None
        self._fb_mmap.close()
        self._fb_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

    def _fb_device(self):
        for device in glob.glob('/sys/class/graphics/fb*'):
            try:
                with io.open(os.path.join(device, 'name'), 'r') as f:
                    if f.read().strip() == self.SENSE_HAT_FB_NAME:
                        return os.path.join('/dev', os.path.basename(device))
            except IOError as exc:
                if exc.errno != errno.ENOENT:
                    raise
        raise RuntimeError('unable to locate SenseHAT framebuffer device')

    def _get_raw(self):
        return self._fb_array
    def _set_raw(self, value):
        self._fb_array[:] = value
    raw = property(_get_raw, _set_raw)

    def _get_array(self):
        arr = self._array
        arr._screen = None
        try:
            arr = rgb565_to_rgb(self.raw)
            arr = self._undo_transforms(arr)
        finally:
            arr._screen = self
        return result
    def _set_array(self, value):
        if isinstance(value, np.ndarray):
            value = value.view(color_dtype).reshape((8, 8))
        else:
            value = np.array(value, dtype=color_dtype).reshape((8, 8))
        value = self._apply_transforms(value)
        self.raw = (
                ((value['red']   & 0xF8).astype(np.uint16) << 8) |
                ((value['green'] & 0xFC).astype(np.uint16) << 3) |
                ((value['blue']  & 0xF8).astype(np.uint16) >> 3)
                )
    array = property(_get_array, _set_array)

    def _get_vflip(self):
        return self._vflip
    def _set_vflip(self, value):
        # TODO fix this
        p = self.pixels
        self._vflip = bool(value)
        self.pixels = p
    vflip = property(_get_vflip, _set_vflip)

    def _get_hflip(self):
        return self._hflip
    def _set_hflip(self, value):
        # TODO fix this
        p = self.pixels
        self._hflip = bool(value)
        self.pixels = p
    hflip = property(_get_hflip, _set_hflip)

    def _get_rotation(self):
        return self._rotation
    def _set_rotation(self, value):
        if value % 90:
            raise ValueError('rotation must be a multiple of 90')
        # TODO fix this
        p = self.pixels
        self._rotation = value % 360
        self.pixels = p
    rotation = property(_get_rotation, _set_rotation)

    def _apply_transforms(self, arr):
        if self._vflip:
            arr = np.flipud(arr)
        if self._hflip:
            arr = np.fliplr(arr)
        arr = np.rot90(arr, self._rotation // 90)
        return arr

    def _undo_transforms(self, arr):
        arr = np.rot90(arr, (360 - self._rotation) // 90)
        if self._hflip:
            arr = np.fliplr(arr)
        if self._vflip:
            arr = np.flipud(arr)
        return arr

    def clear(self):
        self.raw = 0

    def image(self):
        arr = self._undo_transforms(self.raw)
        arr = rgb565_to_image(arr)
        return arr

    def draw(self, img):
        if not isinstance(img, Image.Image) or img.size != (8, 8) or img.mode != 'RGB':
            raise ValueError('image must be an 8x8 RGB PIL Image')
        arr = image_to_rgb565(img)
        arr = self._apply_transforms(arr)
        self.raw = arr

    def _play(self, frames, fps=10):
        delay = 1 / fps
        for frame in frames:
            self.raw = self._apply_transforms(frame)
            time.sleep(delay)

    def marquee(
            # XXX Is there a better default on Raspbian?
            self, text, font='Piboto-Regular', size=7, foreground=(1, 1, 1),
            background=(0, 0, 0), letter_space=1, steps=None, fps=10):
        try:
            f = self._font_cache[font]
        except KeyError:
            if font is None:
                f = ImageFont.load_default()
            else:
                f = ImageFont.truetype(font, size)
            self._font_cache
        size = f.getsize(text)
        im = Image.new('RGB', (size[0] + 16, size[1]))
        draw = ImageDraw.Draw(im)
        draw.rectangle(((0, 0), im.size), Color(*background).rgb_bytes)
        draw.text((0, 0), text, Color(*foreground).rgb_bytes, f)
        if steps is None:
            steps = im.size[0]
        arr = image_to_rgb565(img)
        x_inc = im.size[0] / steps
        frames = [
            arr[im.size[1] - 8:im.size[1], x:x + 8]
            for x_step in range(steps)
            for x in (x_step * x_inc,)
        ]
        self._play(frames, fps)

    def fade_to(self, image, steps=10, fps=10, easing=linear):
        base = self.image()
        target = buf_to_image(image)
        mask = np.empty((8, 8), np.uint8)
        mask_img = Image.frombuffer('L', (8, 8), mask, 'raw', 'L', 0, 1)
        frames = []
        for f in easing(steps):
            mask[...] = int(255 * f)
            frame = base.copy()
            frame.paste(target, (0, 0), mask_img)
            frames.append(image_to_rgb565(frame))
        self._play(frames, fps)

    def slide_to(self, image, direction='left', steps=10, fps=10, easing=linear):
        raise NotImplementedError()

    def cover_with(self, image, direction='left', steps=10, fps=10, easing=linear):
        raise NotImplementedError()

    def expand_to(self, image, center=(4, 4), steps=10, fps=10, easing=linear):
        raise NotImplementedError()


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


def rgb888_to_image(arr):
    raise NotImplementedError()


def rgb888_to_rgb565(arr, out=None):
    raise NotImplementedError()


def rgb565_to_rgb888(arr, out=None):
    raise NotImplementedError()


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
    arr = (arr * 255).astype(np.uint8)
    out[..., 0] = arr['r']
    out[..., 1] = arr['g']
    out[..., 2] = arr['b']
    return out


def rgb888_to_rgb(arr, out=None):
    """
    Convert a numpy :class:`ndarray` in RGB888 format (unsigned 8-bit values
    in 3 planes) to RGB format (structured floating-point type with 3 values,
    each between 0 and 1).
    1.
    """
    if out is None:
        out = np.empty(arr.shape[:2], color_dtype)
    else:
        # XXX Change to exception
        assert out.shape == arr.shape[:2] and out.dtype == color_dtype
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
        out = np.empty(arr.shape, color_dtype)
    else:
        assert out.shape == arr.shape and out.dtype == color_dtype
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
    elif isinstance(buf, np.ndarray) and 2 <= buf.shape <= 3:
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
    if isinstance(buf, np.ndarray) and buf.dtype == color_dtype:
        return buf
    else:
        arr = buf_to_rgb888(buf)
        return arr.astype(np.float16) / 255
