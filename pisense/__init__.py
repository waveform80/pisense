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
The :mod:`pisense` module is the main namespace for the pisense package; it
imports (and exposes) all publically accessible classes, functions, and
constants from all the modules beneath it for convenience. It also defines
the top-level :class:`SenseHAT` class.
"""

from __future__ import (
    unicode_literals,
    absolute_import,
    print_function,
    division,
)

import io
import os
import warnings
import subprocess

from .exc import (
    SenseWarning,
    SenseHATReinit,
    SenseStickWarning,
    SenseStickBufferFull,
    SenseStickCallbackRead,
)
from .formats import (
    color_dtype,
    buf_to_image, buf_to_rgb888, buf_to_rgb, iter_to_rgb,
    image_to_rgb565, rgb565_to_image,
    image_to_rgb888, rgb888_to_image,
    image_to_rgb, rgb_to_image,
    rgb888_to_rgb565, rgb565_to_rgb888,
    rgb_to_rgb888, rgb888_to_rgb,
    rgb_to_rgb565, rgb565_to_rgb,
)
from .array import ScreenArray, array
from .screen import SenseScreen, DEFAULT_GAMMA, LOW_GAMMA
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

    One particular keyword argument, *emulate*, takes its default from an
    environment variable: ``PISENSE_EMULATE``. If set, this must be an integer
    number, typically 0 or 1 (0 is assumed if the variable is not set). This
    argument indicates whether the instance should attach to the "real" Sense
    HAT or the desktop `Sense HAT emulator`_. The environment variable is
    particularly useful as it means scripts can be tested against the emulator
    without alteration. For example:

    .. code-block:: console

        $ PISENSE_EMULATE=1 python examples/rainbow.py

    .. warning::

        Your script should not attempt to create more than one instance of this
        class (given it represents a single piece of hardware). If you attempt
        to do so a :exc:`SenseHATReinit` warning will be raised and the
        existing instance will be returned.

    .. _Sense HAT emulator: https://sense-emu.readthedocs.io/
    """
    __slots__ = ('_settings', '_screen', '_stick', '_imu', '_environ')
    hat = None

    def __new__(cls, settings='/etc/RTIMULib.ini', **kwargs):
        if SenseHAT.hat is not None:
            warnings.warn(
                SenseHATReinit("The SenseHAT class has already been "
                               "instantiated; returning existing instance"))
            self = SenseHAT.hat
        else:
            self = super(SenseHAT, cls).__new__(cls)
            self._settings = None
            self._screen = None
            self._stick = None
            self._imu = None
            self._environ = None
            try:
                SenseHAT.hat = self
                # Old-style kw-only args...
                fps = kwargs.pop('fps', 15)
                easing = kwargs.pop('easing', linear)
                max_events = kwargs.pop('max_events', 100)
                flush_input = kwargs.pop('flush_input', True)
                emulate_default = bool(int(os.environ.get(
                    'PISENSE_EMULATE', '0')))
                emulate = kwargs.pop('emulate', emulate_default)
                if kwargs:
                    raise TypeError("unexpected keyword argument %r" %
                                    kwargs.popitem()[0])
                if emulate:
                    from sense_emu.lock import EmulatorLock
                    lock = EmulatorLock('sense_emu')
                    if not lock.wait(1):
                        # XXX All this stuff should be a method on EmulatorLock
                        warnings.warn(Warning('No emulator detected; '
                                              'spawning sense_emu_gui'))
                        with io.open(os.devnull, 'r+b') as devnull:
                            try:
                                setpgrp = os.setpgrp
                            except AttributeError:
                                setpgrp = None
                            # setpgrp is called to spawn a new process group,
                            # ensuring that signals from the interpreter (e.g.
                            # the user pressing Ctrl+C) don't get sent to the
                            # emulator too
                            subprocess.Popen(['sense_emu_gui'],
                                             preexec_fn=setpgrp,
                                             stdin=devnull, stdout=devnull,
                                             stderr=devnull, close_fds=True)
                            if not lock.wait(10):
                                raise RuntimeError('Failed to launch emulator')
                # pylint: disable=protected-access
                self._settings = SenseSettings(settings, emulate=emulate)
                self._screen = SenseScreen(fps, easing, emulate=emulate)
                self._stick = SenseStick(max_events, flush_input, emulate=emulate)
                self._imu = SenseIMU(self._settings, emulate=emulate)
                self._environ = SenseEnviron(self._settings, emulate=emulate)
            except:
                self.close()
                raise
        return self

    def close(self):
        """
        Call the :meth:`close` method to close the Sense HAT interface and free
        up any background resources. The method is idempotent (you can call it
        multiple times without error) and after it is called, any operations on
        the Sense HAT may return an error (but are not guaranteed to do so).
        """
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
        SenseHAT.hat = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

    @property
    def settings(self):
        """
        Returns a :class:`SenseSettings` object representing the Sense HAT's
        configuration settings.
        """
        return self._settings

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
        self._imu.rotation = value
