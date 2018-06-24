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
Defines various easing functions for the screen animation methods.
"""

from __future__ import (
    unicode_literals,
    absolute_import,
    print_function,
    division,
)


def linear(steps):
    """
    Linear easing function; yields *steps* values between 0.0 and 1.0.

    .. image:: images/linear.*
        :align: center

    This is the default easing function which simply progresses the animation
    at a constant rate from start to finish.
    """
    if steps <= 0:
        raise ValueError('steps must be a positive integer >0')
    elif steps == 1:
        yield 1.0
    else:
        for t in range(steps):
            yield t / (steps - 1)


def ease_in(steps):
    """
    Quadratic ease-in function; yields *steps* values between 0.0 and 1.0.

    .. image:: images/ease_in.*
        :align: center

    This function starts the animation off slowly, and builds speed as it
    progresses, finishing abruptly.
    """
    for t in linear(steps):
        yield t ** 2


def ease_out(steps):
    """
    Quadratic ease-out function; yields *steps* values between 0.0 and 1.0.

    .. image:: images/ease_out.*
        :align: center

    This function starts the animation suddenly and then eases it gradually
    to a halt.
    """
    for t in linear(steps):
        yield t * (2 - t)


def ease_in_out(steps):
    """
    Quadratic ease-in-out function; yields *steps* values between 0.0 and 1.0.

    .. image:: images/ease_in_out.*
        :align: center

    This function starts the animation gradually, progresses rapidly at the
    mid-point, and eases gently to a halt.
    """
    for t in linear(steps):
        yield 2 * t ** 2 if t < 0.5 else (4 - 2 * t) * t - 1
