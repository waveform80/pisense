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

import mock
import pytest
from time import sleep
from itertools import cycle
from pisense import *



COMPASS_READING = (0.2, 5.6, 1.2)
GYRO_READING = (0.01, 0.1, 0.0)
ACCEL_READING = (0.0, 0.1, 1.0)
FUSION_READING = (0.0, 0.01, 0.12)

VALID_RAW = {
    'compassValid': True,
    'compass': COMPASS_READING,
    'gyroValid': True,
    'gyro': GYRO_READING,
    'accelValid': True,
    'accel': ACCEL_READING,
    'fusionPoseValid': True,
    'fusionPose': FUSION_READING,
}
VALID_READ = IMUState(
    IMUVector(*COMPASS_READING),
    IMUVector(*GYRO_READING),
    IMUVector(*ACCEL_READING),
    IMUOrient(*FUSION_READING),
)


@pytest.fixture()
def Settings(request):
    patcher = mock.patch('RTIMU.Settings')
    request.addfinalizer(patcher.stop)
    return patcher.start()


@pytest.fixture()
def RTIMU(request):
    patcher = mock.patch('RTIMU.RTIMU')
    request.addfinalizer(patcher.stop)
    result = patcher.start()
    result.return_value.IMUName.return_value = 'LSM9DS1'
    result.return_value.IMUGetPollInterval.return_value = 3
    result.return_value.IMURead.return_value = True
    result.return_value.getIMUData.return_value = VALID_RAW
    return result


def test_imu_init(Settings, RTIMU):
    # ALWAYS mock out Settings as otherwise instantiation attempts to write
    # to various files...
    imu = SenseIMU('/etc/foo.ini')
    try:
        Settings.assert_called_once_with('/etc/foo')
        RTIMU.assert_called()
        RTIMU().setCompassEnable.assert_called_with(True)
        RTIMU().setGyroEnable.assert_called_with(True)
        RTIMU().setAccelEnable.assert_called_with(True)
    finally:
        imu.close()


def test_imu_init_with_settings(Settings, RTIMU):
    settings = SenseSettings('/etc/foo.ini')
    imu = SenseIMU(settings)
    try:
        RTIMU.assert_called_once_with(settings.settings)
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


def test_imu_context_handler(Settings, RTIMU):
    with SenseIMU('/etc/foo.ini') as imu:
        assert imu.read() == VALID_READ
    with pytest.raises(AttributeError):
        imu.read()


def test_imu_iter(Settings, RTIMU):
    with SenseIMU() as imu:
        it = iter(imu)
        assert next(it) == VALID_READ
        assert next(it) == VALID_READ
        assert RTIMU().getIMUData.call_count == 2
        invalid_orient = VALID_RAW.copy()
        invalid_orient['fusionPoseValid'] = False
        RTIMU().getIMUData.side_effect = cycle([invalid_orient, VALID_RAW])
        assert next(it) == VALID_READ
        assert RTIMU().getIMUData.call_count == 4


def test_imu_attr(Settings, RTIMU):
    with SenseIMU() as imu:
        assert imu.name == 'LSM9DS1'
        assert imu.compass == IMUVector(*COMPASS_READING)
        assert imu.gyro == IMUVector(*GYRO_READING)
        assert imu.accel == IMUVector(*ACCEL_READING)
        assert imu.orient == IMUOrient(*FUSION_READING)


def test_imu_sensors(Settings, RTIMU):
    with SenseIMU() as imu:
        assert imu.sensors == {'compass', 'accel', 'gyro'}
        imu.sensors = {'accel', b'gyro'}
        RTIMU().setCompassEnable.assert_called_with(False)
        RTIMU().setAccelEnable.assert_called_with(True)
        RTIMU().setGyroEnable.assert_called_with(True)
        with pytest.raises(ValueError):
            imu.sensors = {'foo'}


def test_imu_sensors_str(Settings, RTIMU):
    with SenseIMU() as imu:
        imu.sensors = 'accel'
        assert imu.sensors == {'accel'}
        imu.sensors = b'gyro'
        assert imu.sensors == {'gyro'}


def test_imu_read_delay(Settings, RTIMU):
    with SenseIMU() as imu:
        next_raw = VALID_RAW.copy()
        next_raw['accel'] = (0.0, 0.0, 0.9)
        next_read = VALID_READ._replace(accel=IMUVector(*next_raw['accel']))
        RTIMU().getIMUData.side_effect = cycle([VALID_RAW, next_raw])
        assert imu.accel == IMUVector(*ACCEL_READING)
        sleep(imu._interval)
        assert imu.accel == next_read.accel
