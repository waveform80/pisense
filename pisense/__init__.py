from __future__ import (
    unicode_literals,
    absolute_import,
    print_function,
    division,
    )
str = type('')

import io
import os
import glob
import mmap
import errno

import numpy as np

class SenseScreen(object):
    SENSE_HAT_FB_NAME = 'RPi-Sense FB'

    def __init__(self):
        self._fb_file = io.open(self._fb_device(), 'wb+')
        self._fb_mmap = mmap.mmap(self._fb_file.fileno(), 128)
        self._fb_array = np.frombuffer(self._fb_mmap, dtype=np.uint16).reshape((8, 8))

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
                    if f.read() == self.SENSE_HAT_FB_NAME:
                        return os.path.join('/dev', os.path.basename(device))
            except IOError as e:
                if e.errno != errno.ENOENT:
                    raise
        raise RuntimeError('unable to locate SenseHAT framebuffer device')

    def _get_raw(self):
        return self._fb_array
    def _set_raw(self, value):
        self._fb_array[:] = value
    pixels_raw = property(_get_raw, _set_raw)

    def _get_cooked(self):
        result = np.empty((8, 8, 3), dtype=np.uint8)
        result[..., 0] = ((self.raw & 0xF800) >> 8).astype(np.uint8)
        result[..., 1] = ((self.raw & 0x07E0) >> 3).astype(np.uint8)
        result[..., 2] = ((self.raw & 0x001F) << 3).astype(np.uint8)
        return result
    def _set_cooked(self, value):
        r, g, b = (value[..., plane] for plane in range(3))
        self.raw = (
                ((r & 0xF8).astype(np.uint16) << 8) |
                ((g & 0xFC).astype(np.uint16) << 3) |
                ((b & 0xF8).astype(np.uint16) >> 3)
                )

