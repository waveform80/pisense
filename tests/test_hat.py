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

import warnings

import pytest

from pisense import *

try:
    from unittest import mock
except ImportError:
    import mock


# See conftest for custom fixture definitions


def test_hat_init(HAT):
    hat = SenseHAT()
    try:
        assert isinstance(hat.settings, SenseSettings)
        assert isinstance(hat.screen, SenseScreen)
        assert isinstance(hat.stick, SenseStick)
        assert isinstance(hat.environ, SenseEnviron)
        assert isinstance(hat.imu, SenseIMU)
    finally:
        hat.close()


def test_hat_init_bad_args(HAT):
    with pytest.raises(TypeError):
        SenseHAT(foo=1)


def test_hat_singleton(HAT):
    with warnings.catch_warnings(record=True) as w:
        hat = SenseHAT()
        try:
            hat2 = SenseHAT()
            assert len(w) == 1
            assert w[0].category == SenseHATReinit
            assert hat is hat2
        finally:
            hat.close()


def test_hat_close_idempotent(HAT):
    hat = SenseHAT()
    hat.close()
    with pytest.raises(AttributeError):
        hat.stick.read()
    with pytest.raises(AttributeError):
        hat.rotation
    hat.close()


def test_hat_context_handler(HAT):
    with SenseHAT() as hat:
        pass
    with pytest.raises(AttributeError):
        hat.stick.read()
    with pytest.raises(AttributeError):
        hat.rotation


def test_hat_rotation(HAT):
    with SenseHAT() as hat:
        assert hat.rotation == 0
        hat.rotation = 90
        assert hat.screen.rotation == 90
        assert hat.stick.rotation == 90
        #assert hat.imu.rotation == 90
        with pytest.raises(ValueError):
            hat.rotation = 45
