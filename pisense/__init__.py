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

from .exc import (
    SenseWarning,
    SenseStickWarning,
    SenseStickBufferFull,
    SenseStickCallbackRead,
)
from .images import (
    color,
    buf_to_image, buf_to_rgb888, buf_to_rgb,
    image_to_rgb565, rgb565_to_image,
    image_to_rgb888, rgb888_to_image,
    image_to_rgb,
    rgb888_to_rgb565, rgb565_to_rgb888,
    rgb_to_rgb888, rgb888_to_rgb,
    rgb_to_rgb565, rgb565_to_rgb,
)
from .screen import SenseScreen, ScreenArray, array, default_gamma, low_gamma
from .easings import linear, ease_in, ease_out, ease_in_out
from .anim import draw_text, scroll_text, fade_to, slide_to, zoom_to
from .stick import SenseStick, StickEvent
from .imu import SenseIMU, IMUState, IMUVector, IMUOrient
from .environ import (
    SenseEnviron, EnvironReadings,
    temp_pressure, temp_humidity, temp_average, temp_both
)
from .settings import SenseSettings


class SenseHAT(object):
    """
    An instance of this class represents the Sense HAT as a whole. It provides
    attributes for objects that represent each component of the HAT, including:

    * :attr:`stick` for the joystick

    * :attr:`screen` for the display

    * :attr:`environ` for the environmental sensors

    * :attr:`imu` for the Inertial Measurement Unit (IMU)

    The *settings* parameter can be used to point to alternate settings files
    but it is strongly recommended you leave this at the default as this can
    affect the calibration of the IMU. Other keyword arguments are used in the
    initialization of the subordinate objects; see the documentation for their
    classes for further information.
    """

    def __init__(self, settings='/etc/RTIMULib.ini', **kwargs):
        super(SenseHAT, self).__init__()
        # Old-style kw-only args...
        fps = kwargs.pop('fps', 15)
        easing = kwargs.pop('easing', linear)
        max_events = kwargs.pop('max_events', 100)
        flush_input = kwargs.pop('flush_input', True)
        if kwargs:
            raise TypeError("unexpected keyword argument %r" %
                            kwargs.popitem()[0])
        self._settings = SenseSettings(settings)
        self._screen = SenseScreen(fps, easing)
        self._stick = SenseStick(max_events, flush_input)
        self._imu = SenseIMU(self._settings)
        self._environ = SenseEnviron(self._settings)

    def close(self):
        if self._environ is not None:
            self._environ.close()
            self._environ = None
        if self._imu is not None:
            self._imu.close()
            self._imu = None
        if self._stick is not None:
            self._stick.close()
            self._stick = None
        if self._screen is not None:
            self._screen.close()
            self._screen = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

    @property
    def screen(self):
        """
        Returns a :class:`SenseScreen` object representing the Sense HAT's
        display.
        """
        return self._screen

    @property
    def stick(self):
        """
        Returns a :class:`SenseStick` object representing the Sense HAT's
        joystick.
        """
        return self._stick

    @property
    def imu(self):
        """
        Returns a :class:`SenseIMU` object representing the `Inertial
        Measurement Unit`_ (IMU) on the Sense HAT.

        .. _Inertial Measurement Unit: https://en.wikipedia.org/wiki/Inertial_measurement_unit
        """
        return self._imu

    @property
    def environ(self):
        """
        Returns a :class:`SenseEnviron` object representing the environmental
        sensors on the Sense HAT.
        """
        return self._environ

    @property
    def rotation(self):
        """
        Gets or sets the rotation (around the Z-axis) of the Sense HAT. When
        querying this attribute, only the screen's rotation is queried. When
        set, the attribute affects the screen, joystick, *and* IMU.
        """
        return self._screen.rotation

    @rotation.setter
    def rotation(self, value):
        self._screen.rotation = value
        self._stick.rotation = value
        # TODO update IMU
