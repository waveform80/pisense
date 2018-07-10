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
Defines the :class:`SenseIMU`, :class:`IMUState`, :class:`IMUVector`, and
:class:`IMUOrient` classes for querying the inertial measurement unit on the
Sense HAT.
"""

from __future__ import (
    unicode_literals,
    absolute_import,
    print_function,
    division,
)

import time
from math import degrees
from collections import namedtuple

from .settings import SenseSettings

# Make Py2's str and range equivalent to Py3's
str = type('')  # pylint: disable=redefined-builtin,invalid-name


class IMUState(namedtuple('IMUState', ('compass', 'gyro', 'accel', 'orient'))):
    """
    A :func:`~collections.namedtuple` representing a single reading from the
    Inertial Measurement Unit (IMU). The fields are as follows:

    .. attribute:: compass

        An :attr:`IMUVector` tuple containing the raw values from the
        magnetometer in µT (`micro-teslas`_).

    .. attribute:: gyro

        An :attr:`IMUVector` tuple containing the raw values from the
        gyroscope in `radians / second <radians-per-second>`_.

    .. attribute:: accel

        An :attr:`IMUVector` tuple containing the raw values from the
        accelerometer in `standard gravities`_ (g).

    .. attribute:: orient

        The orientation of the HAT, as calculated from the three sensors,
        presented as an :class:`IMUOrient` instance.

    .. _micro-teslas: https://en.wikipedia.org/wiki/Tesla_(unit)
    .. _radians-per-second: https://en.wikipedia.org/wiki/Radian_per_second
    .. _standard gravities: https://en.wikipedia.org/wiki/Standard_gravity
    """
    __slots__ = ()


class IMUVector(namedtuple('IMUVector', ('x', 'y', 'z'))):
    """
    A :func:`~collections.namedtuple` representing a three-dimensional vector
    with *x*, *y*, and *z* components.  This is used to represent the output of
    the three IMU sensors (magnetometer, gryoscope, and accelerometer).

    .. attention:: TODO Add HAT-specific vector directions diagram
    """
    # TODO Consider splitting Vector out of picraft and re-using it here
    __slots__ = ()
    def __repr__(self):
        return 'IMUVector(x=%g, y=%g, z=%g)' % self


class IMUOrient(namedtuple('IMUOrient', ('roll', 'pitch', 'yaw'))):
    """
    A :func:`~collections.namedtuple` representing the orientation of the Sense
    HAT in radians (though the display is provided in degrees for human
    convenience) as `roll, pitch, and yaw`_.

    .. attention:: TODO add HAT-specific roll, pitch, yaw diagram

    .. _roll, pitch, and yaw: https://en.wikipedia.org/wiki/Aircraft_principal_axes
    """
    __slots__ = ()
    def __repr__(self):
        return 'IMUOrient(roll=%g (%.1f°), pitch=%g (%.1f°), yaw=%g (%.1f°))' % (
            self.roll, degrees(self.roll), self.pitch, degrees(self.pitch),
            self.yaw, degrees(self.yaw))


class SenseIMU(object):
    """
    The :class:`SenseIMU` class represents the Inertial Measurement Unit (IMU)
    on the Sense HAT. Users can either instantiate the class themselves, or can
    access an instance from :attr:`SenseHAT.imu`.

    The *settings* parameter can be used to point to alternate settings files
    but it is strongly recommended you leave this at the default as this can
    affect the calibration of the IMU.

    If the *emulate* parameter is ``True``, the instance will connect to the
    IMU in the `desktop Sense HAT emulator`_ instead of the "real" Sense HAT
    IMU.

    .. _desktop Sense HAT emulator: https://sense-emu.readthedocs.io/
    """

    __slots__ = (
        '_rotation',
        '_settings',
        '_imu',
        '_interval',
        '_sensors',
        '_readings',
        '_last_read',
    )

    def __init__(self, settings=None, emulate=False):
        if emulate:
            from sense_emu import RTIMU
        else:
            import RTIMU
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
            IMUVector(None, None, None),
            IMUVector(None, None, None),
            IMUVector(None, None, None),
            IMUOrient(None, None, None)
        )
        self._rotation = 0
        self._last_read = None

    def close(self):
        """
        Call the :meth:`close` method to close the inertial measurement unit
        interface and free up any background resources. The method is
        idempotent (you can call it multiple times without error) and after it
        is called, any operations on the inertial measurement unit may return
        an error (but are not guaranteed to do so).
        """
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
        """
        Return the current state of the inertial measurement unit as an
        :class:`IMUState` tuple.

        .. note::

            This method will wait until the next set of readings are available,
            and then return them. Hence it is suitable for use in a loop
            without additional waits, although it may be simpler to simply
            treat the instance as an iterator in that case.

            This is in contrast to reading the :attr:`gyro`, :attr:`accel`,
            :attr:`compass`, and :attr:`orient` attributes which always return
            immediately.
        """
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
            dat = self._imu.getIMUData()
            self._readings = IMUState(
                IMUVector(*self._rotate(*dat['compass']))
                    if dat.get('compassValid', False) else None,
                IMUVector(*self._rotate(*dat['gyro']))
                    if dat.get('gyroValid', False) else None,
                IMUVector(*self._rotate(*dat['accel']))
                    if dat.get('accelValid', False) else None,
                # TODO what about rotation for fusion-pose?
                IMUOrient(*dat['fusionPose'])
                    if dat.get('fusionPoseValid', False) else None,
            )
            self._last_read = now

    def _rotate(self, x, y, z):
        return {
            0: (x, y, z),
            90: (-y, x, z),
            180: (-x, -y, z),
            270: (y, -x, z),
        }[self._rotation]

    @property
    def name(self):
        """
        Returns the name of the IMU chip. On the Sense HAT this should always
        be "LSM9DS1".
        """
        return self._imu.IMUName()

    @property
    def compass(self):
        """
        Return the current reading from the magnetometer as a 3-dimensional
        :class:`IMUVector` tuple. The reading is measured in in µT
        (`micro-teslas`_).

        .. _micro-teslas: https://en.wikipedia.org/wiki/Tesla_(unit)
        """
        self._read(False)
        return self._readings.compass

    @property
    def gyro(self):
        """
        Return the current reading from the gyroscope as a 3-dimensional
        :class:`IMUVector` tuple. The reading is measured in
        `radians-per-second`_.

        .. _radians-per-second: https://en.wikipedia.org/wiki/Radian_per_second
        """
        self._read(False)
        return self._readings.gyro

    @property
    def accel(self):
        """
        Return the current reading from the accelerometer as a 3-dimensional
        :class:`IMUVector` tuple. The reading is measured in
        `standard gravities`_.

        .. _standard gravities: https://en.wikipedia.org/wiki/Standard_gravity
        """
        self._read(False)
        return self._readings.accel

    @property
    def orient(self):
        """
        Return the current calculated orientation of the board as a
        :class:`IMUOrient` tuple containing `roll, pitch, and yaw`_ in
        `radians`_.

        .. note::

            The sensors that are used in determining the orientation are
            specified in the :attr:`sensors` property.

            The orientation of the board is only calculated when the sensors
            are read. The drift of certain sensors (the gyroscope in
            particular) mean that reading the orientation more frequently can
            result in greater accuracy.

        .. _radians: https://en.wikipedia.org/wiki/Radian
        .. _roll, pitch, and yaw: https://en.wikipedia.org/wiki/Aircraft_principal_axes
        """
        self._read(False)
        return self._readings.orient

    @property
    def sensors(self):
        """
        Controls which sensors are used for calculating the :attr:`orient`
        property.
        """
        return self._sensors

    @sensors.setter
    def sensors(self, value):
        if isinstance(value, (bytes, str)):
            value = {value}
        value = {
            s.decode('ascii')
            if isinstance(s, bytes) else s for s in value
        }
        clean = {'compass', 'gyro', 'accel'} & set(value)
        if clean != value:
            raise ValueError('invalid sensor "%s"' % (value - clean).pop())
        self._sensors = frozenset(clean)
        self._imu.setCompassEnable('compass' in self._sensors)
        self._imu.setGyroEnable('gyro' in self._sensors)
        self._imu.setAccelEnable('accel' in self._sensors)

    @property
    def rotation(self):
        """
        Specifies the rotation about the Z axis applied to IMU readings as a
        multiple of 90 degrees. When rotation is 0 (the default), positive X
        is toward the joystick, and positive Y is away from the GPIO pins:

        .. image:: images/rotation_0.*

        When rotation is 90, positive X is toward the GPIO pins, and positive
        Y is toward the joystick:

        .. image:: images/rotation_90.*

        The other two rotations are trivial to derive from this.

        .. note::

            This property is updated by the unifying :attr:`SenseHAT.rotation`
            attribute.
        """
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        # TODO If rotation is modified we should update the current
        # self._readings
        if value % 90:
            raise ValueError('rotation must be a multiple of 90')
        self._rotation = value % 360
