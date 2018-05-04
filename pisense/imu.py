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

from __future__ import (
    unicode_literals,
    absolute_import,
    print_function,
    division,
    )
str = type('')

import time
from math import degrees
from collections import namedtuple

import RTIMU

from .settings import SenseSettings


IMUState = namedtuple('IMUState', ('compass', 'gyro', 'accel', 'orient'))

class IMUReadings(namedtuple('IMUReadings', ('x', 'y', 'z'))):
    __slots__ = ()
    def __repr__(self):
        return 'IMUReadings(x=%g, y=%g, z=%g)' % self

class IMUOrient(namedtuple('IMUOrient', ('roll', 'pitch', 'yaw'))):
    __slots__ = ()
    def __repr__(self):
        return 'IMUOrient(roll=%g (%.1f°), pitch=%g (%.1f°), yaw=%g (%.1f°))' % (
            self.roll, degrees(self.roll), self.pitch, degrees(self.pitch),
            self.yaw, degrees(self.yaw))


class SenseIMU(object):
    def __init__(self, settings=None):
        # TODO rotation
        if not isinstance(settings, SenseSettings):
            settings = SenseSettings(settings)
        self._settings = settings
        self._imu = RTIMU.RTIMU(self._settings.settings)
        if not self._imu.IMUInit():
            raise RuntimeError('IMU initialization failed')
        self._interval = self._imu.IMUGetPollInterval() / 1000.0 # seconds
        self._imu.setCompassEnable(True)
        self._imu.setGyroEnable(True)
        self._imu.setAccelEnable(True)
        self._sensors = frozenset(('compass', 'gyro', 'accel'))
        self._readings = IMUState(
            IMUReadings(None, None, None),
            IMUReadings(None, None, None),
            IMUReadings(None, None, None),
            IMUOrient(None, None, None)
        )
        self._last_read = None

    def close(self):
        self._imu = None
        self._settings = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

    def __iter__(self):
        while True:
            value = self.read()
            if value.orient is not None:
                yield value

    def read(self):
        self._read(True)
        return self._readings

    def _read(self, wait):
        now = time.time()
        if self._last_read is not None:
            if wait:
                time.sleep(max(0.0, self._interval - (now - self._last_read)))
            elif now - self._last_read < self._interval:
                return
        if self._imu.IMURead():
            d = self._imu.getIMUData()
            self._readings = IMUState(
                IMUReadings(*d['compass']) if d.get('compassValid', False) else None,
                IMUReadings(*d['gyro']) if d.get('gyroValid', False) else None,
                IMUReadings(*d['accel']) if d.get('accelValid', False) else None,
                IMUOrient(*d['fusionPose']) if d.get('fusionPoseValid', False) else None,
            )
            self._last_read = now

    @property
    def name(self):
        return self._imu.IMUName()

    @property
    def compass(self):
        self._read(False)
        return self._readings.compass

    @property
    def gyro(self):
        self._read(False)
        return self._readings.gyro

    @property
    def accel(self):
        self._read(False)
        return self._readings.accel

    @property
    def orient(self):
        self._read(False)
        return self._readings.orient

    def _get_sensors(self):
        return self._sensors
    def _set_sensors(self, value):
        if isinstance(value, bytes):
            value = value.decode('ascii')
        if isinstance(value, str):
            value = {value}
        clean = {'compass', 'gyro', 'accel'} & set(value)
        if clean != value:
            raise ValueError('invalid sensor "%s"' % (value - clean).pop())
        self._sensors = frozenset(clean)
        self._imu.setCompassEnable('compass' in self._sensors)
        self._imu.setGyroEnable('gyro' in self._sensors)
        self._imu.setAccelEnable('accel' in self._sensors)
    orient_sensors = property(_get_sensors, _set_sensors)
