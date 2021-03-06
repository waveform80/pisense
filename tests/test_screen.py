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
import glob
import errno

import pytest
import numpy as np
from colorzero import Color
from PIL import Image

from pisense import *

try:
    from unittest import mock
except ImportError:
    import mock


# See conftest for custom fixture definitions


def test_screen_init(screen_array):
    # Ensure initialization opens the correct device (our mocked screen device)
    # and that it doesn't alter the content of the device
    screen_array[...] = 0xFFFF
    screen = SenseScreen()
    try:
        expected = np.array([Color('white')] * 64, dtype=color_dtype).reshape((8, 8))
        assert (screen.array == expected).all()
    finally:
        screen.close()
    screen_array[...] = 0x0000
    screen = SenseScreen()
    try:
        expected = np.array([Color('black')] * 64, dtype=color_dtype).reshape((8, 8))
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


def test_screen_close_idempotent(screen_array):
    screen = SenseScreen()
    screen.close()
    with pytest.raises(AttributeError):
        screen.array
    screen.close()


def test_screen_context_handler(screen_array):
    with SenseScreen() as screen:
        pass
    with pytest.raises(AttributeError):
        screen.array


def test_screen_raw(screen_array):
    screen_array[...] = 0xFFFF
    with SenseScreen() as screen:
        expected = np.array([0xFFFF] * 64, np.uint16).reshape((8, 8))
        assert (screen.raw == expected).all()
        expected = np.array([0x0000] * 64, np.uint16).reshape((8, 8))
        screen.raw = expected
        assert (screen_array == expected).all()


def test_screen_gamma(screen_gamma):
    with SenseScreen() as screen:
        assert len(screen.gamma) == 32
        assert all(isinstance(i, int) for i in screen.gamma)
        screen.gamma = LOW_GAMMA
        assert screen.gamma == LOW_GAMMA
        screen.gamma = None # reset gamma
        assert screen.gamma == DEFAULT_GAMMA
        with pytest.raises(ValueError):
            screen.gamma = [1]
        with pytest.raises(ValueError):
            screen.gamma = [64] * 32
        with pytest.raises(ValueError):
            screen.gamma = [0.2] * 32
        with pytest.raises(ValueError):
            screen.gamma = ['foo'] * 32


def test_screen_array(screen_array):
    with SenseScreen() as screen:
        data = [Color('yellow')] * 32 + [Color('magenta')] * 32
        expected = np.array(data, color_dtype)
        screen.array = expected
        assert (screen_array == np.array(
            [c.rgb565 for c in data], np.uint16
        ).reshape((8, 8))).all()
        data = [(1.0, 0.0, 0.0)] * 32 + [(0.0, 0.0, 1.0)] * 32
        expected = np.array(data).reshape((8, 8, 3))
        screen.array = expected
        assert (screen_array == np.array(
            [Color(*c).rgb565 for c in data], np.uint16
        ).reshape((8, 8))).all()
        data = [Color('green')] * 32 + [Color('black')] * 32
        screen.array = data
        assert (screen_array == np.array(
            [c.rgb565 for c in data], np.uint16
        ).reshape((8, 8))).all()


def test_screen_vflip(screen_array):
    screen_array[:] = np.arange(64).reshape((8, 8))
    expected = screen_array.copy()
    with SenseScreen() as screen:
        assert not screen.vflip
        screen.vflip = True
        assert screen.vflip
        assert (screen_array == np.flipud(expected)).all()
        screen.vflip = False
        assert (screen_array == expected).all()


def test_screen_hflip(screen_array):
    screen_array[:] = np.arange(64).reshape((8, 8))
    expected = screen_array.copy()
    with SenseScreen() as screen:
        assert not screen.hflip
        screen.hflip = True
        assert screen.hflip
        assert (screen_array == np.fliplr(expected)).all()
        screen.hflip = False
        assert (screen_array == expected).all()


def test_screen_rotate(screen_array):
    screen_array[:] = np.arange(64).reshape((8, 8))
    expected = screen_array.copy()
    with SenseScreen() as screen:
        assert screen.rotation == 0
        screen.rotation = 90
        assert screen.rotation == 90
        assert (screen_array == np.rot90(expected)).all()
        screen.rotation = 540
        assert screen.rotation == 180
        assert (screen_array == np.rot90(expected, 2)).all()
        with pytest.raises(ValueError):
            screen.rotation = 45


def test_screen_clear(screen_array):
    screen_array[:] = np.arange(64).reshape((8, 8))
    with SenseScreen() as screen:
        expected = np.array([0] * 64, np.uint16).reshape((8, 8))
        screen.clear()
        assert (screen_array == expected).all()
        expected = np.array([Color('red').rgb565] * 64, np.uint16).reshape((8, 8))
        screen.clear((1, 0, 0))
        assert (screen_array == expected).all()


def test_screen_image(screen_array):
    expected = [Color('yellow')] * 32 + [Color('magenta')] * 32
    expected = np.array([c.rgb565 for c in expected], np.uint16).reshape((8, 8))
    screen_array[:] = expected
    with SenseScreen() as screen:
        assert (image_to_rgb565(screen.image()) == expected).all()


def test_screen_draw(screen_array):
    expected = [Color('yellow')] * 32 + [Color('magenta')] * 32
    expected = np.array([c.rgb565 for c in expected], np.uint16).reshape((8, 8))
    with SenseScreen() as screen:
        screen.draw(rgb565_to_image(expected))
        assert (screen_array == expected).all()
        with pytest.raises(ValueError):
            screen.draw(rgb565_to_image(expected).crop((0, 0, 4, 4)))


def test_screen_play(screen_array):
    with mock.patch('time.sleep') as sleep:
        played = []
        def sleep_patch(delay):
            played.append((delay, screen_array.copy()))
        sleep.side_effect = sleep_patch
        with SenseScreen(fps=1) as screen:
            animation = [
                np.array([Color(c).rgb565] * 64, np.uint16).reshape((8, 8))
                for c in ('red', 'green', 'blue')
            ]
            screen.play(animation)
            assert len(animation) == len(played)
            for aframe, (pdelay, pframe) in zip(animation, played):
                assert pdelay == 1
                assert (aframe == pframe).all()
            del played[:]  # XXX 2.7 compat for .clear()
            animation = [
                np.array([Color(c).rgb565] * 64, np.uint16).reshape((8, 8))
                for c in ('yellow', 'magenta', 'cyan')
            ]
            screen.fps = 20
            screen.play([rgb565_to_image(frame) for frame in animation])
            assert len(animation) == len(played)
            for aframe, (pdelay, pframe) in zip(animation, played):
                assert pdelay == 1 / 20
                assert (aframe == pframe).all()


def test_screen_scroll_text(screen_array):
    # We already test scroll_text produces correct output in test_anim, and
    # that play faithfully reproduces the frames its given so here we just
    # check it produces the desired number of frames for play()
    with mock.patch('pisense.screen.SenseScreen.play') as play:
        with SenseScreen() as screen:
            screen.scroll_text('Hello', duration=1, fps=10)
            assert play.call_count == 1
            assert len(play.call_args[0][0]) == 10
            play.reset_mock()
            screen.scroll_text('Hello', duration=2, fps=30)
            assert play.call_count == 1
            assert len(play.call_args[0][0]) == 60


def test_screen_fade_to(screen_array):
    # Same story here; see notes in test_screen_scroll_text
    with mock.patch('pisense.screen.SenseScreen.play') as play:
        with SenseScreen() as screen:
            white = Image.new('RGB', (8, 8), (1, 1, 1))
            screen.fade_to(white, duration=1, fps=10)
            assert play.call_count == 1
            assert len(play.call_args[0][0]) == 10
            play.reset_mock()
            screen.fade_to(white, duration=2, fps=30)
            assert play.call_count == 1
            assert len(play.call_args[0][0]) == 60


def test_screen_slide_to(screen_array):
    # Same story here; see notes in test_screen_scroll_text
    with mock.patch('pisense.screen.SenseScreen.play') as play:
        with SenseScreen() as screen:
            red = Image.new('RGB', (8, 8), (1, 0, 0))
            screen.slide_to(red, duration=1, fps=10)
            assert play.call_count == 1
            assert len(play.call_args[0][0]) == 10
            play.reset_mock()
            screen.slide_to(red, duration=2, fps=30)
            assert play.call_count == 1
            assert len(play.call_args[0][0]) == 60


def test_screen_zoom_to(screen_array):
    # Same story here; see notes in test_screen_scroll_text
    with mock.patch('pisense.screen.SenseScreen.play') as play:
        with SenseScreen() as screen:
            red = Image.new('RGB', (8, 8), (1, 0, 0))
            screen.zoom_to(red, duration=1, fps=10)
            assert play.call_count == 1
            assert len(play.call_args[0][0]) == 10
            play.reset_mock()
            screen.zoom_to(red, duration=2, fps=30)
            assert play.call_count == 1
            assert len(play.call_args[0][0]) == 60
