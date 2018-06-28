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
native_str = str
str = type('')


import math
import struct

import pytest
import numpy as np
from colorzero import Color

from pisense import *

try:
    from unittest import mock
except ImportError:
    import mock


# See conftest for custom fixture definitions


@pytest.fixture()
def checker_array(request):
    r = Color('red')
    g = Color('green')
    _ = Color('black')
    arr = array([
        r, r, r, r, _, _, _, _,
        r, r, r, r, _, _, _, _,
        r, r, r, r, _, _, _, _,
        r, r, r, r, _, _, _, _,
        _, _, _, _, g, g, g, g,
        _, _, _, _, g, g, g, g,
        _, _, _, _, g, g, g, g,
        _, _, _, _, g, g, g, g,
    ])
    return arr


def test_array_constructor():
    arr_rgb = np.empty((8, 8), dtype=color)
    arr_rgb[...] = Color('black')
    assert (array() == arr_rgb).all()
    assert isinstance(array(), ScreenArray)
    arr_rgb[...] = Color('red')
    assert (array(Color('red')) == arr_rgb).all()
    assert isinstance(array(Color('red')), ScreenArray)
    assert (array(arr_rgb) == arr_rgb).all()
    assert isinstance(array(arr_rgb), ScreenArray)
    arr_rgb888 = np.empty((8, 8, 3), dtype=np.uint8)
    arr_rgb888[..., 0] = 255
    arr_rgb888[..., 1] = 0
    arr_rgb888[..., 2] = 0
    assert (array(arr_rgb888) == arr_rgb).all()
    assert isinstance(array(arr_rgb888), ScreenArray)
    assert (array(b'\xFF\x00\x00' * 64) == arr_rgb).all()
    assert isinstance(array(b'\xFF\x00\x00' * 64), ScreenArray)
    assert (array([Color('red')] * 64) == arr_rgb).all()
    assert isinstance(array([Color('red')] * 64), ScreenArray)


def test_array_ufunc():
    black = array(Color('black'))
    white = array(Color('white'))
    red = array(Color('red'))
    cyan = array(Color('cyan'))
    gray = array(Color(0.5, 0.5, 0.5))
    dark_gray = array(Color(0.25, 0.25, 0.25))
    identity = np.identity(8, np.float32)
    identity = np.dstack((identity,) * 3).view(color)

    # Test standard ops
    assert (black + black == black).all()
    assert (white + black == white).all()
    assert (white - red == cyan).all()
    assert (red * white == red).all()
    assert (white / 2 == gray).all()
    assert (white // 2 == black).all()
    assert (white % 2 == white).all()
    assert (gray ** 2 == dark_gray).all()

    # Test reverse ops
    assert (0.5 + gray == white).all()
    assert (2 * gray == white).all()
    assert (white.view(color, np.ndarray) - white == black).all()
    assert (white.view(color, np.ndarray) / white == white).all()
    assert (gray.view(color, np.ndarray) // white == black).all()
    assert (1 % white == black).all()
    assert (1 ** gray == white).all()

    # Make sure it's not just simple fields that work
    top = red.copy()
    bottom = red.copy()
    top[4:, :] = Color('black')
    bottom[:4, :] = Color('black')
    assert (top + bottom == red).all()

    # Test in-place ops
    a = black.copy()
    a += white
    assert (a == white).all()
    a *= 0.5
    assert (a == gray).all()
    a *= (red * 2)
    assert (a == red).all()
    a /= 1
    assert (a == red).all()
    a //= 2
    assert (a == black).all()
    a = white.copy()
    a -= red
    assert (a == cyan).all()
    a %= 2
    assert (a == cyan).all()
    a **= 2
    assert (a == cyan).all()

    # Test unary ops
    a = black.copy()
    a += +white
    assert (a == white).all()
    a += -red
    assert (a == cyan).all()
    a -= red
    a = abs(a)
    assert (a == white).all()

    # Test logical ops
    assert (gray < white).all()
    assert (red <= white).all()
    assert (white <= white).all()
    assert (white == white).all()
    assert (black != white).all()
    assert (white > gray).all()
    assert (white >= red).all()
    assert (white >= white).all()

    # Test clip
    assert (np.clip(red * 2, 0, 1) == red).all()
    a = red.copy()
    np.clip(a * 2, 0, 1, a)
    assert (a == red).all()


def test_array_update_screen():
    screen = array(Color('black'))
    screen._screen = mock.Mock()
    red = array(Color('red'))
    screen[:4][:, :4] = red[:4, :4]
    r = Color('red')
    b = Color('black')
    assert (screen._screen.array == array([
        r, r, r, r, b, b, b, b,
        r, r, r, r, b, b, b, b,
        r, r, r, r, b, b, b, b,
        r, r, r, r, b, b, b, b,
        b, b, b, b, b, b, b, b,
        b, b, b, b, b, b, b, b,
        b, b, b, b, b, b, b, b,
        b, b, b, b, b, b, b, b,
    ])).all()


def test_array_format_spec(checker_array):
    assert '{0:e#:c0:w80:o>}'.format(checker_array) == (
        '####    \n'
        '####    \n'
        '####    \n'
        '####    \n'
        '    ####\n'
        '    ####\n'
        '    ####\n'
        '    ####')
    assert '{0:e##: c0: w12 : o>}'.format(checker_array) == (
        '########  >\n'
        '########  >\n'
        '########  >\n'
        '########  >\n'
        '        ##>\n'
        '        ##>\n'
        '        ##>\n'
        '        ##>')
    assert '{0: e##: c0 :w11:o>}'.format(checker_array) == (
        '########  >\n'
        '########  >\n'
        '########  >\n'
        '########  >\n'
        '        ##>\n'
        '        ##>\n'
        '        ##>\n'
        '        ##>')
    r = '\x1b[1;31m##\x1b[0m'
    g = '\x1b[32m##\x1b[0m'
    _ = '\x1b[30m##\x1b[0m'
    assert '{0: ::e##:c8 : :w80:o>}'.format(checker_array) == '\n'.join([
        ''.join([r, r, r, r, _, _, _, _]),
        ''.join([r, r, r, r, _, _, _, _]),
        ''.join([r, r, r, r, _, _, _, _]),
        ''.join([r, r, r, r, _, _, _, _]),
        ''.join([_, _, _, _, g, g, g, g]),
        ''.join([_, _, _, _, g, g, g, g]),
        ''.join([_, _, _, _, g, g, g, g]),
        ''.join([_, _, _, _, g, g, g, g]),
    ])
    with pytest.raises(ValueError):
        '{0:f0}'.format(checker_array)


def test_array_format_detect(checker_array):
    result = (
        '########  >\n'
        '########  >\n'
        '########  >\n'
        '########  >\n'
        '        ##>\n'
        '        ##>\n'
        '        ##>\n'
        '        ##>'
    )
    with mock.patch('sys.stdout') as stdout, \
            mock.patch('os.isatty') as isatty:
        with mock.patch.dict('os.environ', {'COLUMNS': '12', 'LINES': '20'}):
            stdout.fileno.side_effect = IOError(22, 'invalid operation')
            assert '{0:e##:o>}'.format(checker_array) == result
            stdout.fileno.side_effect = lambda: 1
            isatty.return_value = False
            assert '{0:e##:o>}'.format(checker_array) == result
        with mock.patch.dict('os.environ', {}):
            assert '{0:e#:o>}'.format(checker_array) == (
                '####    \n'
                '####    \n'
                '####    \n'
                '####    \n'
                '    ####\n'
                '    ####\n'
                '    ####\n'
                '    ####')
        with mock.patch('fcntl.ioctl') as ioctl:
            ioctl.side_effect = lambda fn, ctl, data: struct.pack(
                native_str('hhhh'), 20, 12, 0, 0)
            assert '{0:e##:o>}'.format(checker_array) == result


def test_array_show(checker_array):
    with mock.patch('pisense.screen.ScreenArray.__format__') as fmt:
        fmt.return_value = ''
        checker_array.show('#', '0', 80, '>')
        assert fmt.call_args == mock.call('e#:c0:w80:o>')
        checker_array.show('##')
        assert fmt.call_args == mock.call('e##')
        checker_array.show()
        assert fmt.call_args == mock.call('')
