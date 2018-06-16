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
try:
    from itertools import izip as zip
except ImportError:
    pass


import mock
import pytest
from itertools import tee
from math import isclose
from pisense import *


def pairwise(it):
    a, b = tee(it)
    next(b, None)
    return zip(a, b)


def test_linear():
    assert list(linear(0)) == []
    assert list(linear(1)) == [1]
    assert list(linear(2)) == [0, 1]
    assert list(linear(3)) == [0, 0.5, 1]
    deltas = [b - a for a, b in pairwise(linear(100))]
    assert all(isclose(d, 1/99) for d in deltas)


def test_ease_in():
    assert list(ease_in(0)) == []
    assert list(ease_in(1)) == [1]
    assert list(ease_in(2)) == [0, 1]
    assert list(ease_in(3)) == [0, 0.25, 1]
    deltas = [b - a for a, b in pairwise(ease_in(100))]
    assert all(b > a for a, b in pairwise(deltas))


def test_ease_out():
    assert list(ease_out(0)) == []
    assert list(ease_out(1)) == [1]
    assert list(ease_out(2)) == [0, 1]
    assert list(ease_out(3)) == [0, 0.75, 1]
    deltas = [b - a for a, b in pairwise(ease_out(100))]
    assert all(b < a for a, b in pairwise(deltas))


def test_ease_in_out():
    assert list(ease_in_out(0)) == []
    assert list(ease_in_out(1)) == [1]
    assert list(ease_in_out(2)) == [0, 1]
    assert list(ease_in_out(3)) == [0, 0.5, 1]
    deltas = [b - a for a, b in pairwise(ease_in_out(100))]
    assert all(b > a for a, b in pairwise(deltas[:50]))
    assert all(b < a for a, b in pairwise(deltas[50:]))
