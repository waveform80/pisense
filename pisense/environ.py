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
Defines the :class:`SenseEnviron` and :class:`EnvironReadings` classes for
querying the environmental sensors on the Sense HAT.
"""

from __future__ import (
    unicode_literals,
    absolute_import,
    print_function,
    division,
)

import time
from collections import namedtuple

from .settings import SenseSettings


class EnvironReadings(namedtuple('EnvironReadings', ('pressure', 'humidity', 'temperature'))):
    """
    A :func:`~collections.namedtuple` representing the readings from the
    environmental sensors as a named 3-tuple containing the fields *pressure*
    (in mbar or hPa), *humidity* (in %RH), and *temperature* (in °C)
    respectively.
    """
    __slots__ = ()
    def __repr__(self):
        return 'EnvironReadings(pressure=%g, humidity=%g, temperature=%g)' % self


def temp_pressure(p_temp, h_temp):
    """
    Use this function as :attr:`~SenseEnviron.temp_source` if you want
    to read temperature from the pressure sensor only. This is the default.
    """
    # pylint: disable=unused-argument
    return p_temp


def temp_humidity(p_temp, h_temp):
    """
    Use this function as :attr:`~SenseEnviron.temp_source` if you want
    to read temperature from the humidity sensor only.
    """
    # pylint: disable=unused-argument
    return h_temp


def temp_average(p_temp, h_temp):
    """
    Use this function as :attr:`~SenseEnviron.temp_source` if you wish
    to read the average of both the pressure and humidity sensor's
    temperatures.
    """
    if p_temp is None:
        return h_temp
    elif h_temp is None:
        return p_temp
    else:
        return (p_temp + h_temp) / 2


def temp_both(p_temp, h_temp):
    """
    Use this function as :attr:`~SenseEnviron.temp_source` if you wish
    to return both the pressure and humidity sensor's temperature readings as a
    tuple from the :attr:`~SenseEnviron.temperature` attribute.
    """
    return p_temp, h_temp


class SenseEnviron(object):
    """
    The :class:`SenseEnviron` class represents the suite of environmental
    sensors on the Sense HAT. Users can either instantiate this class
    themselves, or can access an instance from :attr:`SenseHAT.environ`.

    The :attr:`temperature`, :attr:`pressure`, and :attr:`humidity` attributes
    can be queried to read the current values from the sensors. Alternatively,
    the instance can be treated as an iterator in which case readings will be
    yielded as they are detected::

        hat = SenseHAT()
        for reading in hat.environ:
            print(reading.temperature)

    Because both the pressure and humidity sensors contain a temperature
    sensor, a source must be selected for the temperature reading. By default
    this is from the pressure sensor only, but you can specify a function for
    :attr:`temperature_source` which, given the two temperature readings
    returns the reading you are interested in, or some combination there-of.

    The *settings* parameter can be used to point to alternate settings files.
    The *temp_source* parameter provides an initial value for the
    :attr:`temp_source` attribute (this defaults to :func:`temp_humidity`). If
    the *emulate* parameter is ``True``, the instance will connect to the
    environment sensors in the `desktop Sense HAT emulator`_ instead of the
    "real" Sense HAT's sensors.

    .. _desktop Sense HAT emulator: https://sense-emu.readthedocs.io/
    """

    __slots__ = (
        '_settings',
        '_p_sensor',
        '_h_sensor',
        '_readings',
        '_temp_source',
        '_interval',
        '_last_read',
    )

    def __init__(self, settings=None, temp_source=temp_humidity,
                 emulate=False):
        if emulate:
            from sense_emu import RTIMU
        else:
            import RTIMU
        if not isinstance(settings, SenseSettings):
            settings = SenseSettings(settings)
        self._settings = settings
        self._p_sensor = RTIMU.RTPressure(self._settings.settings)
        self._h_sensor = RTIMU.RTHumidity(self._settings.settings)
        if not self._p_sensor.pressureInit():
            raise RuntimeError('Pressure sensor initialization failed')
        if not self._h_sensor.humidityInit():
            raise RuntimeError('Humidity sensor initialization failed')
        self._readings = EnvironReadings(None, None, None)
        self._temp_source = temp_source
        self._interval = 0.04
        self._last_read = None

    def close(self):
        """
        Call the :meth:`close` method to close the environmental sensor
        interface and free up any background resources. The method is
        idempotent (you can call it multiple times without error) and after it
        is called, any operations on the environmental sensors may return an
        error (but are not guaranteed to do so).
        """
        self._p_sensor = None
        self._h_sensor = None
        self._settings = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

    def __iter__(self):
        while True:
            yield self.read()

    def read(self):
        """
        Return the current state of all environmental sensors as an
        :class:`EnvironReadings` tuple.

        .. note::

            This method will wait until the next set of readings are available,
            and then return them. Hence it is suitable for use in a loop
            without additional waits, although it may be simpler to simply
            treat the instance as an iterator in that case.

            This is in contrast to reading the :attr:`pressure`,
            :attr:`humidity`, and :attr:`temperature` attributes which always
            return immediately.
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
        p_valid, p_value, tp_valid, tp_value = self._p_sensor.pressureRead()
        h_valid, h_value, th_valid, th_value = self._h_sensor.humidityRead()
        self._readings = EnvironReadings(
            pressure=p_value if p_valid else None,
            humidity=h_value if h_valid else None,
            temperature=self._temp_source(
                tp_value if tp_valid else None,
                th_value if th_valid else None)
            if tp_valid or th_valid else None)
        self._last_read = now

    @property
    def pressure(self):
        """
        Return the current pressure reading from the environmental sensors.
        The pressure is measured in `millibars`_ (aka `hectopascals`_).

        .. _millibars: https://en.wikipedia.org/wiki/Bar_(unit)
        .. _hectopascals: https://en.wikipedia.org/wiki/Pascal_(unit)
        """
        self._read(False)
        return self._readings.pressure

    @property
    def humidity(self):
        """
        Return the current humidity reading from the environmental sensors.
        The humidity is measured as a % of `relative humidity`_.

        .. _relative humidity: https://en.wikipedia.org/wiki/Relative_humidity
        """
        self._read(False)
        return self._readings.humidity

    @property
    def temperature(self):
        """
        Return the current temperature reading from the environment sensors.
        The temperature is measured in degrees `celsius`_.

        .. _celsius: https://en.wikipedia.org/wiki/Celsius
        """
        self._read(False)
        return self._readings.temperature

    @property
    def temp_source(self):
        """
        Specify the conversion function for the temperature sensors.

        The Sense HAT contains two temperature sensors, one in the humidity
        sensor, and one in the pressure sensor. The :attr:`temp_source`
        property contains the function that is used to determine how the
        temperature is reported from the two sources. The function must take
        two parameters (the readings from the humidity and pressure sensors
        respectively) and can return whatever you wish to see as the value of
        the :attr:`temperature` property (including a tuple of both
        temperatures).

        The default value is :func:`temp_humidity` which simply returns the
        reading from the humidity sensor, discarding the the pressure sensor
        reading.

        .. warning::

            You may be tempted to average the two readings under the assumption
            that this will provide more accuracy. This is almost certainly not
            the case!
        """
        return self._temp_source

    @temp_source.setter
    def temp_source(self, value):
        try:
            value(20, 22)
        except TypeError:
            raise ValueError('temp_source must be a callable that accepts '
                             '2 parameters')
        self._temp_source = value
        self._last_read = None
