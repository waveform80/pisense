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

import RTIMU
import numpy as np


color_dtype = np.dtype([
    (native_str('red'),   np.uint8),
    (native_str('green'), np.uint8),
    (native_str('blue'),  np.uint8),
    ])


class SensePixels(np.ndarray):
    def __new__(cls, screen):
        result = np.ndarray.__new__(cls, shape=(8, 8), dtype=color_dtype)
        result._screen = screen
        return result

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._screen = getattr(obj, '_screen', None)

    def __setitem__(self, index, value):
        super(SensePixels, self).__setitem__(index, value)
        if self._screen:
            self._screen._set_pixels(self)

    def __setslice__(self, i, j, sequence)
        super(SensePixels, self).__setslice__(i, j, sequence)
        if self._screen:
            self._screen._set_pixels(self)


class SenseScreen(object):
    SENSE_HAT_FB_NAME = 'RPi-Sense FB'

    def __init__(self):
        self._fb_file = io.open(self._fb_device(), 'wb+')
        self._fb_mmap = mmap.mmap(self._fb_file.fileno(), 128)
        self._fb_array = np.frombuffer(self._fb_mmap, dtype=np.uint16).reshape((8, 8))
        self._fonts = {}

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
            except IOError as e:
                if e.errno != errno.ENOENT:
                    raise
        raise RuntimeError('unable to locate SenseHAT framebuffer device')

    def _get_raw(self):
        return self._fb_array
    def _set_raw(self, value):
        self._fb_array[:] = value
    raw = property(_get_raw, _set_raw)

    def _get_pixels(self):
        result = SensePixels(self)
        result['red']   = ((self.raw & 0xF800) >> 8).astype(np.uint8)
        result['green'] = ((self.raw & 0x07E0) >> 3).astype(np.uint8)
        result['blue']  = ((self.raw & 0x001F) << 3).astype(np.uint8)
        # Fill the bottom bits
        result['red']   |= result['red']   >> 5
        result['green'] |= result['green'] >> 6
        result['blue']  |= result['blue']  >> 5
        return result
    def _set_pixels(self, value):
        value = value.view(color_dtype).reshape((8, 8))
        self.raw = (
                ((value['red']   & 0xF8).astype(np.uint16) << 8) |
                ((value['green'] & 0xFC).astype(np.uint16) << 3) |
                ((value['blue']  & 0xF8).astype(np.uint16) >> 3)
                )
    pixels = property(_get_pixels, _set_pixels)

    def clear(self):
        self.raw = 0

    def draw(self, image):
        if not isinstance(image, np.ndarray):
            try:
                buf = image.tobytes()
            except AttributeError:
                try:
                    buf = image.tostring()
                except AttributeError:
                    raise ValueError('image must be an 8x8 PIL image or numpy array')
            image = np.frombuffer(buf, dtype=np.uint8)
            if len(image) == 192:
                image = image.reshape((8, 8, 3))
            elif len(image) == 64:
                image = image.reshape((8, 8))
                image = np.dstack((image, image, image))
            else:
                raise ValueError('image must be 8x8 pixels in size')
        self.pixels = image

