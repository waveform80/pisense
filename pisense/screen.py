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

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from colorzero import Color

from .easings import linear
from .images import (
    color,
    buf_to_image,
    image_to_rgb565,
    rgb565_to_image,
    rgb_to_rgb565,
    rgb565_to_rgb,
)


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
            if isinstance(v, np.ndarray) and v.dtype == color else v
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

    def __init__(self, easing=linear):
        self._fb_file = io.open(self._fb_device(), 'wb+')
        self._fb_mmap = mmap.mmap(self._fb_file.fileno(), 128)
        self._fb_array = np.frombuffer(self._fb_mmap, dtype=np.uint16).reshape((8, 8))
        self._array = ScreenArray()
        self._array._screen = self
        self._hflip = False
        self._vflip = False
        self._rotation = 0
        self._font_cache = {}
        self.fps = 15
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
        value = value.clip(0, 1)
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

    def clear(self):
        self.raw = 0

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

    def _play(self, frames):
        delay = 1 / self.fps
        for frame in frames:
            self.raw = self._apply_transforms(frame)
            time.sleep(delay)

    def _load_font(self, font, size):
        try:
            f = self._font_cache[font]
        except KeyError:
            if font is None:
                f = ImageFont.load_default()
            else:
                f = ImageFont.truetype(font, size)
            self._font_cache[font] = f
        return f

    def scroll_text(
            # XXX Is there a better default on Raspbian?
            self, text, font='DejaVuSans', size=9, foreground=(1, 1, 1),
            background=(0, 0, 0), letter_space=1, direction='left',
            duration=None):
        f = self._load_font(font, size)
        size = f.getsize(text)
        # +16 for blank screens either side (to let the text scroll onto and
        # off of the display) and +2 to compensate for spillage due to anti-
        # aliasing
        img = Image.new('RGB', (size[0] + 16 + 2, 8))
        if duration is None:
            steps = img.size[0] - 8
        else:
            steps = int(duration * self.fps)
        x_inc = (img.size[0] - 8) / steps
        try:
            x_steps = {
                'left': range(steps),
                'right': range(steps, -1, -1),
            }[direction]
        except KeyError:
            raise ValueError('invalid direction')
        draw = ImageDraw.Draw(img)
        draw.rectangle(((0, 0), img.size), Color(*background).rgb_bytes)
        draw.text((9, 8 - size[1]), text, Color(*foreground).rgb_bytes, f)
        arr = image_to_rgb565(img)
        frames = [
            arr[:, x:x + 8]
            for x_step in x_steps
            for x in (int(x_step * x_inc),)
        ]
        # Guarantee the final frame is solid background color
        frames[-1] = np.array(
            (Color(*background).rgb565,) * 64, np.uint16).reshape(8, 8)
        self._play(frames)

    def fade_to(self, image, duration=1, easing=None):
        if easing is None:
            easing = self.easing
        start = self.image()
        finish = buf_to_image(image)
        mask = np.empty((8, 8), np.uint8)
        mask_img = Image.frombuffer('L', (8, 8), mask, 'raw', 'L', 0, 1)
        frames = []
        for f in easing(int(duration * self.fps)):
            mask[...] = int(255 * f)
            frame = start.copy()
            frame.paste(finish, (0, 0), mask_img)
            frames.append(image_to_rgb565(frame))
        self._play(frames)

    def slide_to(self, image, direction='left', cover=False, duration=1,
                 easing=None):
        if easing is None:
            easing = self.easing
        try:
            delta_x, delta_y = {
                'left':  (-1, 0),
                'right': (1, 0),
                'up':    (0, -1),
                'down':  (0, 1),
            }[direction]
        except KeyError:
            raise ValueError('invalid direction: ' % direction)
        start = self.image().resize((64, 64))
        image = buf_to_image(image)
        finish = image.resize((64, 64))
        if not cover:
            canvas = Image.new('RGB', (64, 64))
        frames = []
        for f in easing(int(duration * self.fps)):
            x = int(delta_x * f * 64)
            y = int(delta_y * f * 64)
            if cover:
                canvas = start.copy()
            else:
                canvas.paste(start, (x, y))
            canvas.paste(finish, (64 * -delta_x + x, 64 * -delta_y + y))
            frames.append(image_to_rgb565(canvas.resize((8, 8), Image.BOX)))
        # Ensure the final frame is the finish image (without bicubic blur)
        frames[-1] = image_to_rgb565(image)
        self._play(frames)

    def zoom_to(self, image, center=(4, 4), direction='in', duration=1,
                easing=None):
        if easing is None:
            easing = self.easing
        if direction == 'in':
            base = self.image().resize((64, 64))
            top = buf_to_image(image)
            final = top
        elif direction == 'out':
            final = buf_to_image(image)
            base = final.resize((64, 64))
            top = self.image().copy()
        else:
            raise ValueError('invalid direction: %s' % direction)
        frames = []
        mask = np.empty((8, 8), np.uint8)
        mask_img = Image.frombuffer('L', (8, 8), mask, 'raw', 'L', 0, 1)
        for f in easing(int(duration * self.fps)):
            if direction == 'out':
                f = 1 - f
            mask[...] = int(255 * f)
            frame = base.copy()
            frame.paste(top, (center[0] * 8, center[1] * 8), mask_img)
            frame = frame.crop((
                int(center[0] * f * 8),
                int(center[1] * f * 8),
                int(64 - f * 8 * (8 - (center[0] + 1))),
                int(64 - f * 8 * (8 - (center[1] + 1))),
            ))
            frames.append(image_to_rgb565(frame.resize((8, 8), Image.BOX)))
        # Ensure the final frame is the finish image (without bicubic blur)
        frames[-1] = image_to_rgb565(final)
        self._play(frames)
