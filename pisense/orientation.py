from __future__ import (
    unicode_literals,
    absolute_import,
    print_function,
    division,
    )
str = type('')

import math
import time
from collections import namedtuple

import RTIMU


Readings = namedtuple('Readings', ('x', 'y', 'z'))
Orientation = namedtuple('Orientation', ('roll', 'pitch', 'yaw'))


class SenseOrientation(object):
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
        self._last_read = None
        self.orientation_sensors = {'compass', 'gyroscope', 'accelerometer'}

    def __iter__(self):
        while True:
            value = self.orientation
            if value:
                yield value
            delay = max(0.0, self._last_read + self._interval - time.time())
            if delay:
                time.sleep(delay)

    @property
    def name(self):
        return self._imu.IMUName()

    @property
    def compass(self):
        self._refresh()
        return self._compass

    @property
    def gyroscope(self):
        self._refresh()
        return self._gyroscope

    @property
    def accelerometer(self):
        self._refresh()
        return self._accelerometer

    @property
    def orientation(self):
        self._refresh()
        return self._fusion

    @property
    def orientation_degrees(self):
        return Orientation(*(math.degrees(e) % 360 for e in self.orientation))

    def _get_sensors(self):
        return self._sensors
    def _set_sensors(self, value):
        self._sensors = frozenset(value)
        self._imu.setCompassEnable('compass' in self._sensors)
        self._imu.setGyroEnable('gyroscope' in self._sensors)
        self._imu.setAccelEnable('accelerometer' in self._sensors)
    orientation_sensors = property(_get_sensors, _set_sensors)

    def _refresh(self):
        now = time.time()
        if self._last_read is None or now - self._last_read > self._interval:
            if self._imu.IMURead():
                d = self._imu.getIMUData()
                if d.get('compassValid', False):
                    self._compass = Readings(*d['compass'])
                if d.get('gyroValid', False):
                    self._gyroscope = Readings(*d['gyro'])
                if d.get('accelValid', False):
                    self._accelerometer = Readings(*d['accel'])
                if d.get('fusionPoseValid', False):
                    self._fusion = Orientation(d['fusionPose'])
                self._last_read = now

