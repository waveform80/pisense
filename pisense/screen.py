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
Defines the :class:`ScreenArray` and :class:`SenseScreen` classes for
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
import struct
import fcntl

import numpy as np
from colorzero import Color

from .easings import linear
from .anim import scroll_text, fade_to, slide_to, zoom_to
from .images import (
    color,
    buf_to_image,
    image_to_rgb565,
    rgb565_to_image,
    rgb_to_rgb565,
    rgb565_to_rgb,
)


default_gamma = [0,  0,  0,  0,  0,  0,  1,  1,
                 2,  2,  3,  3,  4,  5,  6,  7,
                 8,  9,  10, 11, 12, 14, 15, 17,
                 18, 20, 21, 23, 25, 27, 29, 31]

low_gamma = [0, 1, 1, 1, 1, 1, 1,  1,
             1, 1, 1, 1, 1, 2, 2,  2,
             3, 3, 3, 4, 4, 5, 5,  6,
             6, 7, 7, 8, 8, 9, 10, 10]


def array(data=None, shape=(8, 8)):
    """
    Use this function to construct a new :class:`ScreenArray` and fill it
    with an initial source of *data*, which can be a single :class:`Color`,
    a list of :class:`Color` values, or another (compatible) array.
    """
    # TODO what about an Image?
    result = ScreenArray(shape)
    if data is None:
        result[...] = 0
    else:
        try:
            result[...] = data
        except ValueError:
            result.ravel()[...] = data
    return result


class ScreenArray(np.ndarray):
    # pylint: disable=too-few-public-methods

    def __new__(cls, shape=(8, 8)):
        # pylint: disable=protected-access
        result = np.ndarray.__new__(cls, shape=shape, dtype=color)
        result._screen = None
        return result

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = [
            v.view(np.float16, np.ndarray).reshape(v.shape + (3,))
            if isinstance(v, np.ndarray) and v.dtype == color else
            v.view(v.dtype, np.ndarray).reshape(v.shape)
            if isinstance(v, np.ndarray) else v
            for v in inputs
        ]
        try:
            v, = kwargs['out']
        except (KeyError, ValueError):
            pass
        else:
            kwargs['out'] = (
                v.view(np.float16, np.ndarray).reshape(v.shape + (3,)),
            )
        result = super(ScreenArray, self).__array_ufunc__(
            ufunc, method, *inputs, **kwargs)
        if (
                isinstance(result, np.ndarray) and
                result.dtype == np.float16 and
                len(result.shape) == 3 and
                result.shape[-1] == 3):
            result = result.view(color, self.__class__).squeeze()
        return result

    def __array_finalize__(self, obj):
        # pylint: disable=protected-access,attribute-defined-outside-init
        if obj is None:
            return
        self._screen = getattr(obj, '_screen', None)

    def __setitem__(self, index, value):
        # pylint: disable=protected-access
        super(ScreenArray, self).__setitem__(index, value)
        if self._screen:
            # If we're a slice of the original pixels value, find the parent
            # that contains the complete array and send that to _set_pixels
            orig = self
            while orig.shape != (8, 8) and orig.base is not None:
                orig = orig.base
            self._screen._set_array(orig)

    def __setslice__(self, i, j, sequence):
        # pylint: disable=protected-access
        super(ScreenArray, self).__setslice__(i, j, sequence)
        if self._screen:
            orig = self
            while orig.shape != (8, 8) and orig.base is not None:
                orig = orig.base
            self._screen._set_array(orig)

    def copy(self, order='C'):
        result = super(ScreenArray, self).copy(order)
        result._screen = None
        return result


class SenseScreen(object):
    # pylint: disable=too-many-instance-attributes
    SENSE_HAT_FB_NAME = 'RPi-Sense FB'

    GET_GAMMA = 61696
    SET_GAMMA = 61697
    RESET_GAMMA = 61698
    GAMMA_DEFAULT = 0
    GAMMA_LOW = 1
    GAMMA_USER = 2

    def __init__(self, fps=15, easing=linear):
        # TODO gamma
        self._fb_file = io.open(self._fb_device(), 'wb+')
        self._fb_mmap = mmap.mmap(self._fb_file.fileno(), 128)
        self._fb_array = np.frombuffer(self._fb_mmap, dtype=np.uint16).reshape((8, 8))
        self._array = ScreenArray()
        self._array._screen = self
        self._hflip = False
        self._vflip = False
        self._rotation = 0
        self._font_cache = {}
        self.fps = fps
        self.easing = easing

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

    def _get_gamma(self):
        buf = bytearray(32)
        fcntl.ioctl(self._fb_file, SenseScreen.GET_GAMMA, buf)
        return list(buf)
    def _set_gamma(self, value):
        if value is None:
            fcntl.ioctl(self._fb_file, SenseScreen.RESET_GAMMA, 0)
        else:
            if len(value) != 32:
                raise ValueError('gamma array must contain 32 entries')
            if not all (0 <= v < 32 for v in value):
                raise ValueError('gamma values must be in the range 0..31')
            buf = struct.pack(native_str('32B'), *value)
            fcntl.ioctl(self._fb_file, SenseScreen.SET_GAMMA, buf)
    gamma = property(_get_gamma, _set_gamma)

    def _get_array(self):
        arr = self._array
        arr._screen = None
        try:
            rgb565_to_rgb(self.raw, arr)
            arr = self._undo_transforms(arr)
        finally:
            arr._screen = self
        return arr
    def _set_array(self, value):
        if isinstance(value, np.ndarray):
            value = value.view(color).reshape((8, 8))
        else:
            value = np.array(value, dtype=color).reshape((8, 8))
        value = self._apply_transforms(value)
        rgb_to_rgb565(value, self.raw)
    array = property(_get_array, _set_array)

    def _get_vflip(self):
        return self._vflip
    def _set_vflip(self, value):
        p = self._undo_transforms(self.raw)
        self._vflip = bool(value)
        self.raw = self._apply_transforms(p)
    vflip = property(_get_vflip, _set_vflip)

    def _get_hflip(self):
        return self._hflip
    def _set_hflip(self, value):
        p = self._undo_transforms(self.raw)
        self._hflip = bool(value)
        self.raw = self._apply_transforms(p)
    hflip = property(_get_hflip, _set_hflip)

    def _get_rotation(self):
        return self._rotation
    def _set_rotation(self, value):
        if value % 90:
            raise ValueError('rotation must be a multiple of 90')
        p = self._undo_transforms(self.raw)
        self._rotation = value % 360
        self.raw = self._apply_transforms(p)
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

    def clear(self, color=Color('black')):
        if not isinstance(color, Color):
            color = Color(*color)
        self.raw = color.rgb565

    def image(self):
        arr = self._undo_transforms(self.raw)
        arr = rgb565_to_image(arr)
        return arr

    def draw(self, buf):
        img = buf_to_image(buf)
        if img.size != (8, 8):
            raise ValueError('image must be an 8x8 RGB PIL Image')
        arr = image_to_rgb565(img)
        arr = self._apply_transforms(arr)
        self.raw = arr

    def play(self, frames):
        delay = 1 / self.fps
        for frame in frames:
            self.raw = self._apply_transforms(frame)
            time.sleep(delay)

    def scroll_text(self, text, font=None, size=8, foreground=Color('white'),
                    background=Color('black'), direction='left',
                    duration=None, fps=None):
        self.play(scroll_text(text, font, size, foreground, background,
                              direction, duration,
                              self.fps if fps is None else fps))

    def fade_to(self, image, duration=1, fps=None, easing=None):
        self.play(fade_to(self.image(), image, duration,
                          self.fps if fps is None else fps,
                          self.easing if easing is None else easing))

    def slide_to(self, image, direction='left', cover=False, duration=1,
                 fps=None, easing=None):
        self.play(slide_to(self.image(), image, direction, cover, duration,
                           self.fps if fps is None else fps,
                           self.easing if easing is None else easing))

    def zoom_to(self, image, center=(4, 4), direction='in', duration=1,
                fps=None, easing=None):
        self.play(zoom_to(self.image(), image, center, direction, duration,
                          self.fps if fps is None else fps,
                          self.easing if easing is None else easing))
