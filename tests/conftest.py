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

import io
import os
import sys
import glob
import mmap
import fcntl
import numpy as np

import pytest
from colorzero import Color

# Terrible hack to ensure we can test on non-Pi platforms (must be done before
# importing pisense)
try:
    from unittest import mock
except ImportError:
    import mock
sys.modules['RTIMU'] = mock.Mock()

from pisense import *



@pytest.fixture()
def Settings(request):
    patcher = mock.patch('RTIMU.Settings')
    request.addfinalizer(patcher.stop)
    return patcher.start()


@pytest.fixture()
def compass_reading():
    return (0.2, 5.6, 1.2)


@pytest.fixture()
def gyro_reading():
    return (0.01, 0.1, 0.0)


@pytest.fixture()
def accel_reading():
    return (0.0, 0.1, 1.0)


@pytest.fixture()
def fusion_reading():
    return (0.0, 0.01, 0.12)


@pytest.fixture()
def imu_raw_reading(compass_reading, gyro_reading, accel_reading, fusion_reading):
    return {
        'compassValid': True,
        'compass': compass_reading,
        'gyroValid': True,
        'gyro': gyro_reading,
        'accelValid': True,
        'accel': accel_reading,
        'fusionPoseValid': True,
        'fusionPose': fusion_reading,
    }


@pytest.fixture()
def imu_cooked_reading(compass_reading, gyro_reading, accel_reading, fusion_reading):
    return IMUState(
        IMUVector(*compass_reading),
        IMUVector(*gyro_reading),
        IMUVector(*accel_reading),
        IMUOrient(*fusion_reading),
    )


@pytest.fixture()
def RTIMU(request, imu_raw_reading):
    patcher = mock.patch('RTIMU.RTIMU')
    request.addfinalizer(patcher.stop)
    result = patcher.start()
    result.return_value.IMUName.return_value = 'LSM9DS1'
    result.return_value.IMUGetPollInterval.return_value = 3
    result.return_value.IMURead.return_value = True
    result.return_value.getIMUData.return_value = imu_raw_reading
    return result


@pytest.fixture()
def RTPressure(request, Settings):
    # ALWAYS mock out Settings as otherwise instantiation attempts to write
    # to various files
    patcher = mock.patch('RTIMU.RTPressure')
    request.addfinalizer(patcher.stop)
    result = patcher.start()
    result.return_value.pressureRead.return_value = (True, 1000.0, True, 20.0)
    return result


@pytest.fixture()
def RTHumidity(request, Settings):
    # ALWAYS mock out Settings as otherwise instantiation attempts to write
    # to various files
    patcher = mock.patch('RTIMU.RTHumidity')
    request.addfinalizer(patcher.stop)
    result = patcher.start()
    result.return_value.humidityRead.return_value = (True, 50.0, True, 22.0)
    return result


# TODO screen_file and stick_device conflict in both patching io.open and
# glob.glob. This makes testing impossible on a non-Pi platform currently;
# perhaps combine the open/glob patches into a separate fixture?

@pytest.fixture()
def screen_file(request, tmpdir, _open=io.open, _glob=glob.glob):
    fbfile = io.open(str(tmpdir.join('fb')), 'wb+', buffering=0)
    fbfile.write(b'\x00' * 128)
    fbfile.flush()
    fbfile.seek(0)
    def glob_patch(pattern):
        if pattern == '/sys/class/graphics/fb*':
            return ['/sys/class/graphics/fb%d' % i for i in range(2)]
        else:
            return _glob(pattern)
    def open_patch(filename, mode, *args, **kwargs):
        if filename == '/sys/class/graphics/fb0/name':
            return io.StringIO(SenseScreen.SENSE_HAT_FB_NAME)
        elif filename == '/dev/fb0':
            return _open(str(tmpdir.join('fb')), mode, *args, **kwargs)
        else:
            return _open(filename, mode, *args, **kwargs)
    glob_mock = mock.patch('glob.glob', side_effect=glob_patch)
    open_mock = mock.patch('io.open', side_effect=open_patch)
    def fin():
        fbfile.close()
        glob_mock.stop()
        open_mock.stop()
    request.addfinalizer(fin)
    glob_mock.start()
    open_mock.start()
    return fbfile


@pytest.fixture()
def screen_array(request, screen_file):
    fbmem = mmap.mmap(screen_file.fileno(), 128)
    def fin():
        fbmem.close()
    request.addfinalizer(fin)
    return np.frombuffer(fbmem, dtype=np.uint16).reshape((8, 8))


@pytest.fixture()
def screen_gamma(request, screen_file, _ioctl=fcntl.ioctl):
    gamma = DEFAULT_GAMMA[:]
    def ioctl_patch(f, ctl, buf):
        if f.name == screen_file.name:
            if ctl == SenseScreen.GET_GAMMA:
                buf[:] = gamma
                return 0
            elif ctl == SenseScreen.RESET_GAMMA:
                gamma[:] = DEFAULT_GAMMA[:]
                return 0
            elif ctl == SenseScreen.SET_GAMMA:
                gamma[:] = buf
                return 0
        return _ioctl(f, ctl, buf)
    ioctl_mock = mock.patch('fcntl.ioctl', side_effect=ioctl_patch)
    def fin():
        ioctl_mock.stop()
    request.addfinalizer(fin)
    ioctl_mock.start()


@pytest.fixture()
def stick_device(request, _open=io.open, _glob=glob.glob):
    rpipe, wpipe = os.pipe()
    def glob_patch(pattern):
        if pattern == '/sys/class/input/event*':
            return ['/sys/class/input/event%d' % i for i in range(5)]
        else:
            return _glob(pattern)
    def open_patch(filename, mode, *args, **kwargs):
        if filename == '/sys/class/input/event0/device/name':
            return io.StringIO(SenseStick.SENSE_HAT_EVDEV_NAME)
        elif filename == '/dev/input/event0':
            return os.fdopen(rpipe, mode, *args, **kwargs)
        else:
            return _open(filename, mode, *args, **kwargs)
    glob_mock = mock.patch('glob.glob', side_effect=glob_patch)
    open_mock = mock.patch('io.open', side_effect=open_patch)
    def fin():
        glob_mock.stop()
        open_mock.stop()
    request.addfinalizer(fin)
    glob_mock.start()
    open_mock.start()
    return os.fdopen(wpipe, 'wb', buffering=0)


@pytest.fixture()
def HAT(Settings, RTIMU, RTPressure, RTHumidity, stick_device, screen_array):
    # This fixture is a convenience for the "full HAT" tests that just ensures
    # all hardware is being emulated for the duration of the test
    return
