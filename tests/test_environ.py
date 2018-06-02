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
from itertools import cycle
from time import sleep
from pisense import *


def test_environ_init():
    # ALWAYS mock out Settings as otherwise instantiation attempts to write
    # to various files...
    with mock.patch('RTIMU.Settings') as Settings, \
            mock.patch('RTIMU.RTPressure') as RTPressure, \
            mock.patch('RTIMU.RTHumidity') as RTHumidity:
        env = SenseEnviron('/etc/foo.ini')
        Settings.assert_called_once_with('/etc/foo')
        RTPressure.assert_called()
        RTHumidity.assert_called()


def test_environ_init_with_settings():
    # ALWAYS mock out Settings as otherwise instantiation attempts to write
    # to various files...
    with mock.patch('RTIMU.Settings') as Settings, \
            mock.patch('RTIMU.RTPressure') as RTPressure, \
            mock.patch('RTIMU.RTHumidity') as RTHumidity:
        settings = SenseSettings('/etc/foo.ini')
        env = SenseEnviron(settings)
        RTPressure.assert_called_once_with(settings.settings)
        RTHumidity.assert_called_once_with(settings.settings)


def test_environ_init_fail():
    with mock.patch('RTIMU.Settings') as Settings, \
            mock.patch('RTIMU.RTPressure') as RTPressure, \
            mock.patch('RTIMU.RTHumidity') as RTHumidity:
        RTPressure().pressureInit.return_value = False
        with pytest.raises(RuntimeError):
            SenseEnviron()
        RTPressure().pressureInit.return_value = True
        RTHumidity().humidityInit.return_value = False
        with pytest.raises(RuntimeError):
            SenseEnviron()


def test_environ_read():
    with mock.patch('RTIMU.Settings') as Settings, \
            mock.patch('RTIMU.RTPressure') as RTPressure, \
            mock.patch('RTIMU.RTHumidity') as RTHumidity:
        RTPressure().pressureRead.return_value = (True, 1000.0, True, 20.0)
        RTHumidity().humidityRead.return_value = (True, 50.0, True, 22.0)
        env = SenseEnviron()
        assert env.read() == EnvironReadings(1000.0, 50.0, 22.0)


def test_environ_close_idempotent():
    with mock.patch('RTIMU.Settings') as Settings, \
            mock.patch('RTIMU.RTPressure') as RTPressure, \
            mock.patch('RTIMU.RTHumidity') as RTHumidity:
        env = SenseEnviron()
        env.close()
        with pytest.raises(AttributeError):
            env.read()
        env.close()


def test_environ_context_handler():
    with mock.patch('RTIMU.Settings') as Settings, \
            mock.patch('RTIMU.RTPressure') as RTPressure, \
            mock.patch('RTIMU.RTHumidity') as RTHumidity:
        RTPressure().pressureRead.return_value = (True, 1000.0, True, 20.0)
        RTHumidity().humidityRead.return_value = (True, 50.0, True, 22.0)
        with SenseEnviron() as env:
            assert env.read() == EnvironReadings(1000.0, 50.0, 22.0)
        with pytest.raises(AttributeError):
            env.read()


def test_environ_iter():
    with mock.patch('RTIMU.Settings') as Settings, \
            mock.patch('RTIMU.RTPressure') as RTPressure, \
            mock.patch('RTIMU.RTHumidity') as RTHumidity:
        RTPressure().pressureRead.side_effect = cycle([
            (True, 1000.0, True, 20.0),
            (True, 1000.5, True, 20.7),
        ])
        RTHumidity().humidityRead.side_effect = cycle([
            (True, 50.0, True, 22.0),
            (True, 51.0, True, 21.7),
        ])
        env = SenseEnviron()
        it = iter(env)
        assert next(it) == (1000.0, 50.0, 22.0)
        assert next(it) == (1000.5, 51.0, 21.7)


def test_environ_attr():
    with mock.patch('RTIMU.Settings') as Settings, \
            mock.patch('RTIMU.RTPressure') as RTPressure, \
            mock.patch('RTIMU.RTHumidity') as RTHumidity:
        RTPressure().pressureRead.return_value = (True, 1000.0, True, 20.0)
        RTHumidity().humidityRead.return_value = (True, 50.0, True, 22.0)
        env = SenseEnviron()
        assert env.pressure == 1000.0
        assert env.humidity == 50.0
        assert env.temperature == 22.0


def test_environ_temp_sources():
    with mock.patch('RTIMU.Settings') as Settings, \
            mock.patch('RTIMU.RTPressure') as RTPressure, \
            mock.patch('RTIMU.RTHumidity') as RTHumidity:
        RTPressure().pressureRead.return_value = (True, 1000.0, True, 20.0)
        RTHumidity().humidityRead.return_value = (True, 50.0, True, 22.0)
        env = SenseEnviron()
        assert env.temp_source is temp_humidity
        assert env.temperature == 22.0
        env.temp_source = temp_pressure
        assert env.temperature == 20.0
        env.temp_source = temp_both
        assert env.temperature == (20.0, 22.0)
        env.temp_source = temp_average
        assert env.temperature == 21.0
        with pytest.raises(ValueError):
            env.temp_source = lambda x: x


def test_environ_temp_average():
    with mock.patch('RTIMU.Settings') as Settings, \
            mock.patch('RTIMU.RTPressure') as RTPressure, \
            mock.patch('RTIMU.RTHumidity') as RTHumidity:
        RTPressure().pressureRead.return_value = (True, 1000.0, True, 20.0)
        RTHumidity().humidityRead.return_value = (True, 50.0, True, 22.0)
        env = SenseEnviron()
        env.temp_source = temp_average
        assert env.temperature == 21.0
        RTPressure().pressureRead.return_value = (True, 1000.0, False, 20.0)
        assert env.read().temperature == 22.0
        RTHumidity().humidityRead.return_value = (True, 50.0, False, 22.0)
        RTPressure().pressureRead.return_value = (True, 1000.0, True, 20.0)
        assert env.read().temperature == 20.0
        RTPressure().pressureRead.return_value = (True, 1000.0, False, 20.0)
        assert env.read().temperature is None


def test_environ_read_delay():
    with mock.patch('RTIMU.Settings') as Settings, \
            mock.patch('RTIMU.RTPressure') as RTPressure, \
            mock.patch('RTIMU.RTHumidity') as RTHumidity:
        RTPressure().pressureRead.side_effect = cycle([
            (True, 1000.0, True, 20.0),
            (True, 1000.5, True, 20.7),
        ])
        RTHumidity().humidityRead.side_effect = cycle([
            (True, 50.0, True, 22.0),
            (True, 51.0, True, 21.7),
        ])
        env = SenseEnviron()
        assert env.pressure == 1000.0
        # Ensure a rapid successive read returns the same value, but a read
        # after the interval wait returns a new value
        assert env.pressure == 1000.0
        sleep(env._interval)
        assert env.pressure == 1000.5
