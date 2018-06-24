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

from time import sleep
from itertools import cycle

import pytest

from pisense import *

try:
    from unittest import mock
except ImportError:
    import mock


# See conftest for custom fixture definitions


def test_imu_init(Settings, RTIMU):
    # ALWAYS mock out Settings as otherwise instantiation attempts to write
    # to various files...
    imu = SenseIMU('/etc/foo.ini')
    try:
        assert Settings.call_args_list == [mock.call('/etc/foo')]
        assert RTIMU.call_count
        assert RTIMU().setCompassEnable.call_args == mock.call(True)
        assert RTIMU().setGyroEnable.call_args == mock.call(True)
        assert RTIMU().setAccelEnable.call_args == mock.call(True)
    finally:
        imu.close()


def test_imu_init_with_settings(Settings, RTIMU):
    settings = SenseSettings('/etc/foo.ini')
    imu = SenseIMU(settings)
    try:
        assert RTIMU.call_args_list == [mock.call(settings.settings)]
    finally:
        imu.close()


def test_imu_init_fail(Settings, RTIMU):
    RTIMU().IMUInit.return_value = False
    with pytest.raises(RuntimeError):
        SenseIMU()


def test_imu_close_idempotent(Settings, RTIMU):
    imu = SenseIMU('/etc/foo.ini')
    imu.close()
    with pytest.raises(AttributeError):
        imu.read()
    imu.close()


def test_imu_context_handler(Settings, RTIMU, imu_cooked_reading):
    with SenseIMU('/etc/foo.ini') as imu:
        assert imu.read() == imu_cooked_reading
    with pytest.raises(AttributeError):
        imu.read()


def test_imu_iter(Settings, RTIMU, imu_raw_reading, imu_cooked_reading):
    with SenseIMU() as imu:
        it = iter(imu)
        assert next(it) == imu_cooked_reading
        assert next(it) == imu_cooked_reading
        assert RTIMU().getIMUData.call_count == 2
        invalid_orient = imu_raw_reading.copy()
        invalid_orient['fusionPoseValid'] = False
        RTIMU().getIMUData.side_effect = cycle([invalid_orient, imu_raw_reading])
        assert next(it) == imu_cooked_reading
        assert RTIMU().getIMUData.call_count == 4


def test_imu_attr(Settings, RTIMU, compass_reading, gyro_reading, accel_reading, fusion_reading):
    with SenseIMU() as imu:
        assert imu.name == 'LSM9DS1'
        assert imu.compass == IMUVector(*compass_reading)
        assert imu.gyro == IMUVector(*gyro_reading)
        assert imu.accel == IMUVector(*accel_reading)
        assert imu.orient == IMUOrient(*fusion_reading)


def test_imu_sensors(Settings, RTIMU):
    with SenseIMU() as imu:
        assert imu.sensors == {'compass', 'accel', 'gyro'}
        imu.sensors = {'accel', b'gyro'}
        assert RTIMU().setCompassEnable.call_args == mock.call(False)
        assert RTIMU().setAccelEnable.call_args == mock.call(True)
        assert RTIMU().setGyroEnable.call_args == mock.call(True)
        with pytest.raises(ValueError):
            imu.sensors = {'foo'}


def test_imu_sensors_str(Settings, RTIMU):
    with SenseIMU() as imu:
        imu.sensors = 'accel'
        assert imu.sensors == {'accel'}
        imu.sensors = b'gyro'
        assert imu.sensors == {'gyro'}


def test_imu_read_delay(Settings, RTIMU, accel_reading, imu_raw_reading, imu_cooked_reading):
    with SenseIMU() as imu:
        next_raw = imu_raw_reading.copy()
        next_raw['accel'] = (0.0, 0.0, 0.9)
        next_read = imu_cooked_reading._replace(accel=IMUVector(*next_raw['accel']))
        RTIMU().getIMUData.side_effect = cycle([imu_raw_reading, next_raw])
        assert imu.accel == IMUVector(*accel_reading)
        sleep(imu._interval)
        assert imu.accel == next_read.accel
