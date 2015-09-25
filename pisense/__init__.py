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
import time
import errno
from collections import namedtuple
from threading import Thread, Event

import RTIMU
import numpy as np


class SenseFont(object):
    def __init__(self, filename_or_obj):
        if isinstance(filename_or_obj, bytes):
            filename_or_obj = filename_or_obj.decode('utf-8')
        if isinstance(filename_or_obj, str):
            with io.open(filename_or_obj, 'r') as font_file:
                self._parse_font(font_file)
        else:
            self._parse_font(font_file)

    def _parse_font(self, f):
        self._chars = {}
        char = None
        lines = []
        for line in f:
            line = line.rstrip()
            if line.endswith(':'):
                if char is not None:
                    self._chars[char] = self._make_array(char, lines)
                char = line[:-1]
                lines = []
            elif line:
                lines.append(line)

    def _make_array(self, char, lines):
        rows = len(lines)
        cols = [len(line) for line in lines]
        if cols != [cols[0]] * rows:
            raise ValueError(
                'irregular number of columns in definition of char "%s"' % char)
        cols = cols[0]
        return np.fromiter(
            (c == '#' for line in lines for c in line),
            dtype=np.bool).reshape((rows, cols))

    def __getitem__(self, key):
        return self._chars[key]

    def render_line(
            self, text, color=(255, 255, 255), letter_space=1):
        w = 0
        h = 0
        for c in text:
            try:
                w += self[c].shape[1] + letter_space
                h = max(h, self[c].shape[0])
            except KeyError:
                raise ValueError('Character "%s" does not exist in font' % c)
        result = np.zeros((h, w, 3), dtype=np.uint8)
        x = 0
        for c in text:
            c_h, c_w = self._chars[c].shape
            for i, c in enumerate(color):
                result[0:c_h, x:x + c_w, i] = self[c] * c
            x += c_w + letter_space
        return result

    def render_text(
            self, text, color=(255, 255, 255), line_space=2, letter_space=1):
        lines = [
            self.render_line(line, color, letter_space=letter_space)
            for line in text.splitlines()
            ]
        height = sum(line.shape[0] for line in lines) + line_space * len(lines)
        width = max(line.shape[1] for line in lines)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        y = 0
        for line in lines:
            image[y:y + line.shape[0], 0:line.shape[1], :] = line
            y += line.shape[0] + line_space
        return image


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
        result = np.empty((8, 8, 3), dtype=np.uint8)
        result[..., 0] = ((self.raw & 0xF800) >> 8).astype(np.uint8)
        result[..., 1] = ((self.raw & 0x07E0) >> 3).astype(np.uint8)
        result[..., 2] = ((self.raw & 0x001F) << 3).astype(np.uint8)
        return result
    def _set_pixels(self, value):
        r, g, b = (value[..., plane] for plane in range(3))
        self.raw = (
                ((r & 0xF8).astype(np.uint16) << 8) |
                ((g & 0xFC).astype(np.uint16) << 3) |
                ((b & 0xF8).astype(np.uint16) >> 3)
                )
    pixels = property(_get_pixels, _set_pixels)

    def draw(self, image):
        if not isinstance(image, np.ndarray):
            try:
                buf = image.tobytes()
            except AttributeError:
                buf = image.tostring()
            image = np.frombuffer(buf, dtype=np.uint8)
            if len(image) == 192:
                image = image.reshape((8, 8, 3))
            elif len(image) == 64:
                image = image.reshape((8, 8))
                image = np.dstack((image, image, image))
            else:
                raise ValueError('image must be 8x8 pixels in size')
        self.pixels = image


Orientation = namedtuple('Orientation', ('roll', 'pitch', 'yaw'))


class SenseIMU(object):
    def __init__(self, imu_settings='RTIMULib'):
        self._settings = RTIMU.Settings(imu_settings)
        self._imu = RTIMU.RTIMU(self._settings)
        if not self._imu.IMUInit():
            raise RuntimeError('IMU initialization failed')
        self._interval = self._imu.IMUGetPollInterval() / 1000.0 # seconds
        self._compass = None
        self._gyroscope = None
        self._accel = None
        self._fusion = None
        self._stopping = Event()
        self._read_thread = Thread(target=self._read)
        self._read_thread.daemon = True
        self._read_thread.start()

    def close(self):
        self._stopping.set()
        self._read_thread.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

    @property
    def name(self):
        return self._imu.IMUName()

    @property
    def compass(self):
        return self._compass

    @property
    def gyroscope(self):
        return self._gyroscope

    @property
    def accel(self):
        return self._accel

    @property
    def fusion(self):
        return self._fusion

    def _read(self):
        while not self._stopping.wait(self._interval):
            if self._imu.IMURead():
                d = self._imu.getIMUData()
                if d.get('compassValid', False):
                    self._compass = Orientation(*d['compass'])
                if d.get('gyroValid', False):
                    self._gyroscope = Orientation(*d['gyro'])
                if d.get('accelValid', False):
                    self._accel = Orientation(*d['accel'])
                if d.get('fusionValid', False):
                    self._fusion = Orientation(*d['fusion'])

