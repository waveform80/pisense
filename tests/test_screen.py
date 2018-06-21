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


import io
import os
import mock
import mmap
import glob
import errno
import struct
import pytest
import numpy as np
from colorzero import Color
from pisense import *


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


@pytest.fixture()
def screen_device(request, tmpdir, _open=io.open, _glob=glob.glob):
    fbfile = io.open(str(tmpdir.join('fb')), 'wb+', buffering=0)
    fbfile.write(b'\x00' * 128)
    fbfile.flush()
    fbfile.seek(0)
    fbmem = mmap.mmap(fbfile.fileno(), 128)
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
        fbmem.close()
        fbfile.close()
        glob_mock.stop()
        open_mock.stop()
    request.addfinalizer(fin)
    glob_mock.start()
    open_mock.start()
    return np.frombuffer(fbmem, dtype=np.uint16).reshape((8, 8))


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
    red = array(Color('red'))
    top = red.copy()
    bottom = red.copy()
    top[4:, :] = Color('black')
    bottom[:4, :] = Color('black')
    assert (top + bottom == red).all()
    assert (np.clip(red * 2, 0, 1, red) == red).all()


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


def test_format_spec(checker_array):
    assert '{0:e#:c0:w80:o>}'.format(checker_array) == (
        '####    \n'
        '####    \n'
        '####    \n'
        '####    \n'
        '        \n'
        '        \n'
        '        \n'
        '        ')
    assert '{0:e##: c0: w12 : o>}'.format(checker_array) == (
        '########  >\n'
        '########  >\n'
        '########  >\n'
        '########  >\n'
        '          >\n'
        '          >\n'
        '          >\n'
        '          >')
    assert '{0: e##: c0 :w11:o>}'.format(checker_array) == (
        '########  >\n'
        '########  >\n'
        '########  >\n'
        '########  >\n'
        '          >\n'
        '          >\n'
        '          >\n'
        '          >')
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


def test_format_detect(checker_array):
    result = (
        '########  >\n'
        '########  >\n'
        '########  >\n'
        '########  >\n'
        '          >\n'
        '          >\n'
        '          >\n'
        '          >'
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
                '        \n'
                '        \n'
                '        \n'
                '        ')
        with mock.patch('fcntl.ioctl') as ioctl:
            ioctl.side_effect = lambda fn, ctl, data: struct.pack(
                native_str('hhhh'), 20, 12, 0, 0)
            assert '{0:e##:o>}'.format(checker_array) == result


def test_array_show(checker_array):
    with mock.patch('sys.stdout', io.StringIO()) as stdout:
        checker_array.show('#', '0', 80, '>')
        assert stdout.getvalue() == (
                '####    \n'
                '####    \n'
                '####    \n'
                '####    \n'
                '        \n'
                '        \n'
                '        \n'
                '        \n')


def test_screen_init(screen_device):
    # Ensure initialization opens the correct device (our mocked screen device)
    # and that it doesn't alter the content of the device
    screen_device[...] = 0xFFFF
    screen = SenseScreen()
    try:
        expected = np.array([Color('white')] * 64, dtype=color).reshape((8, 8))
        assert (screen.array == expected).all()
    finally:
        screen.close()
    screen_device[...] = 0x0000
    screen = SenseScreen()
    try:
        expected = np.array([Color('black')] * 64, dtype=color).reshape((8, 8))
        assert (screen.array == expected).all()
    finally:
        screen.close()


def test_screen_not_found():
    _glob = glob.glob
    _open = io.open
    buffers = ['/sys/class/graphics/fb%d' % i for i in range(2)]
    names = {buf + '/name': 'foo' for buf in buffers}
    with mock.patch('glob.glob') as glob_mock:
        glob_mock.side_effect = lambda pattern: (
            buffers if pattern == '/sys/class/graphics/fb*' else _glob(pattern)
        )
        with mock.patch('io.open') as open_mock:
            open_mock.side_effect = lambda filename, mode, *args, **kwargs: (
                io.StringIO(names[filename]) if filename in names else
                _open(filename, mode, *args, **kwargs)
            )
            with pytest.raises(RuntimeError):
                SenseScreen()


def test_screen_init_fail():
    _glob = glob.glob
    _open = io.open
    buffers = ['/sys/class/graphics/fb%d' % i for i in range(2)]
    names = {
        buf + '/name': IOError(
            errno.EACCES if '1' in buf else errno.ENOENT, 'Error'
        )
        for buf in buffers
    }
    with mock.patch('glob.glob') as glob_mock:
        glob_mock.side_effect = lambda pattern: (
            buffers if pattern == '/sys/class/graphics/fb*' else _glob(pattern)
        )
        with mock.patch('io.open') as open_mock:
            def open_patch(filename, mode, *args, **kwargs):
                if filename in names:
                    raise names[filename]
                else:
                    return _open(filename, mode, *args, **kwargs)
            open_mock.side_effect = open_patch
            with pytest.raises(IOError):
                SenseScreen()


def test_screen_close_idempotent(screen_device):
    screen = SenseScreen()
    screen.close()
    with pytest.raises(AttributeError):
        screen.array
    screen.close()


def test_screen_context_handler(screen_device):
    with SenseScreen() as screen:
        pass
    with pytest.raises(AttributeError):
        screen.array


def test_screen_raw(screen_device):
    screen_device[...] = 0xFFFF
    with SenseScreen() as screen:
        expected = np.array([0xFFFF] * 64, np.uint16).reshape((8, 8))
        assert (screen.raw == expected).all()
        expected = np.array([0x0000] * 64, np.uint16).reshape((8, 8))
        screen.raw = expected
        assert (screen_device == expected).all()
