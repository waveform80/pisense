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

import io
import glob
import errno
import struct
import warnings
from time import sleep, mktime
from datetime import datetime
from threading import Event

import pytest

from pisense import *

try:
    from unittest import mock
except ImportError:
    import mock


# See conftest for custom fixture definitions


def make_event(e, event_type=SenseStick.EV_KEY):
    # XXX 2.7 compat method of deriving POSIX timestamp from local naive datetime
    posix_timestamp = mktime(
        (e.timestamp.year, e.timestamp.month, e.timestamp.day,
         e.timestamp.hour, e.timestamp.minute, e.timestamp.second,
         -1, -1, -1)) + e.timestamp.microsecond / 1e6
    sec, usec = divmod(posix_timestamp, 1)
    direction = {
        'up': SenseStick.KEY_UP,
        'down': SenseStick.KEY_DOWN,
        'left': SenseStick.KEY_LEFT,
        'right': SenseStick.KEY_RIGHT,
        'enter': SenseStick.KEY_ENTER,
    }[e.direction]
    return struct.pack(SenseStick.EVENT_FORMAT,
                       int(sec), int(round(usec * 1000000)),
                       event_type, direction,
                       SenseStick.STATE_RELEASE if not e.pressed else
                       SenseStick.STATE_PRESS if not e.held else
                       SenseStick.STATE_HOLD)


def test_stick_init(stick_device):
    stick = SenseStick(10, flush_input=False)
    try:
        assert not stick._flush
        assert stick._buffer.maxsize == 10
    finally:
        stick.close()


def test_stick_init_not_found():
    _glob = glob.glob
    _open = io.open
    events = ['/sys/class/input/event%d' % i for i in range(5)]
    names = {event + '/device/name': 'foo' for event in events}
    with mock.patch('glob.glob') as glob_mock:
        glob_mock.side_effect = lambda pattern: (
            events if pattern == '/sys/class/input/event*' else _glob(pattern)
        )
        with mock.patch('io.open') as open_mock:
            open_mock.side_effect = lambda filename, mode, *args, **kwargs: (
                io.StringIO(names[filename]) if filename in names else
                _open(filename, mode, *args, **kwargs)
            )
            with pytest.raises(RuntimeError):
                SenseStick()


def test_stick_init_fail():
    _glob = glob.glob
    _open = io.open
    events = ['/sys/class/input/event%d' % i for i in range(5)]
    names = {
        event + '/device/name': IOError(
            errno.EACCES if '4' in event else errno.ENOENT, 'Error'
        )
        for event in events
    }
    with mock.patch('glob.glob') as glob_mock:
        glob_mock.side_effect = lambda pattern: (
            events if pattern == '/sys/class/input/event*' else _glob(pattern)
        )
        with mock.patch('io.open') as open_mock:
            def open_patch(filename, mode, *args, **kwargs):
                if filename in names:
                    raise names[filename]
                else:
                    return _open(filename, mode, *args, **kwargs)
            open_mock.side_effect = open_patch
            with pytest.raises(IOError):
                SenseStick()


def test_stick_close_idemoptent(stick_device):
    stick = SenseStick()
    stick.close()
    with pytest.raises(AttributeError):
        stick.read()
    stick.close()


def test_stick_context_handler(stick_device):
    with SenseStick() as stick:
        pass
    with pytest.raises(AttributeError):
        stick.read()


def test_stick_read(stick_device):
    with SenseStick() as stick:
        evt = StickEvent(datetime.now(), 'up', True, False)
        stick_device.write(make_event(evt))
        assert stick.read() == evt
        assert stick.read(0.01) is None


def test_stick_read_non_key_event(stick_device):
    with SenseStick() as stick:
        evt = StickEvent(datetime.now(), 'up', True, False)
        # Test the device silently ignores event types other than key
        stick_device.write(make_event(evt, event_type=0x02))
        stick_device.write(make_event(evt))
        assert stick.read() == evt
        assert stick.read(0.01) is None


def test_stick_iter(stick_device):
    with SenseStick() as stick:
        evt1 = StickEvent(datetime.now(), 'up', True, False)
        evt2 = StickEvent(datetime.now(), 'up', True, True)
        evt3 = StickEvent(datetime.now(), 'up', False, True)
        stick_device.write(make_event(evt1))
        stick_device.write(make_event(evt2))
        stick_device.write(make_event(evt3))
        it = iter(stick)
        assert next(it) == evt1
        assert next(it) == evt2


def test_stick_stream(stick_device):
    with SenseStick() as stick:
        evt = StickEvent(datetime.now(), 'up', True, False)
        stick_device.write(make_event(evt))
        assert not stick.stream
        stick.stream = True
        assert stick.stream
        it = iter(stick)
        while True:
            read_evt = next(it)
            if read_evt is not None:
                break
        assert read_evt == evt
        assert next(it) is None


def test_stick_rotation(stick_device):
    with SenseStick() as stick:
        evt1 = StickEvent(datetime.now(), 'up', True, False)
        evt2 = StickEvent(datetime.now(), 'up', False, False)
        evt3 = StickEvent(datetime.now(), 'left', True, False)
        evt4 = StickEvent(datetime.now(), 'left', False, False)
        assert stick.rotation == 0
        it = iter(stick)
        stick_device.write(make_event(evt1))
        stick_device.write(make_event(evt2))
        assert next(it) == evt1
        assert next(it) == evt2
        stick.rotation = 90
        assert stick.rotation == 90
        stick_device.write(make_event(evt3))
        stick_device.write(make_event(evt4))
        assert next(it) == evt3._replace(direction='up')
        assert next(it) == evt4._replace(direction='up')
        with pytest.raises(ValueError):
            stick.rotation = 45


def test_stick_buffer_filled(stick_device):
    with warnings.catch_warnings(record=True) as w:
        with SenseStick(max_events=2) as stick:
            events = [
                StickEvent(datetime.now(), direction, pressed, False)
                for direction, pressed in [
                        ('up', True),
                        ('up', False),
                        ('left', True),
                        ('left', False),
                ]
            ]
            stick_device.write(b''.join(make_event(e) for e in events))
            while stick._buffer.empty():
                sleep(0.01)
            sleep(0.1) # let the tiny buffer fill and overflow
            stick.read()
            stick.read()
        assert len(w) == 1
        assert w[0].category == SenseStickBufferFull


def test_stick_callbacks(stick_device):
    with SenseStick() as stick:
        directions = ['left', 'right', 'up', 'down', 'enter']
        events = [
            StickEvent(datetime.now(), direction, pressed, False)
            for direction, pressed in [
                (d, on_off)
                for d in directions
                for on_off in (True, False)
            ]
        ]
        markers = [Event() for event in events]
        def handler(e):
            for event, marker in zip(events, markers):
                if e == event:
                    marker.set()
        for d in directions:
            setattr(stick, 'when_' + d, handler)
        for d in directions:
            assert getattr(stick, 'when_' + d) == handler
        for event in events:
            stick_device.write(make_event(event))
        for index, marker in enumerate(markers):
            if not marker.wait(1):
                assert False, 'Event handler %d failed to fire' % index


def test_stick_callbacks_no_queue(stick_device):
    with SenseStick() as stick:
        event = StickEvent(datetime.now(), 'left', True, False)
        marker = Event()
        stick.when_left = lambda evt: marker.set()
        stick_device.write(make_event(event))
        assert marker.wait(1)
        sleep(0.2)
        stick_device.write(make_event(event._replace(pressed=False)))
        assert marker.wait(1)


def test_stick_callbacks_unassigned(stick_device):
    with SenseStick() as stick:
        events = [
            StickEvent(datetime.now(), direction, pressed, False)
            for direction, pressed in [
                    ('right', True),
                    ('right', False),
                    ('down', True),
                    ('down', False),
                    ('left', True),
                    ('left', False),
            ]
        ]
        markers = [Event() for event in events]
        def handler(e):
            for event, marker in zip(events, markers):
                if e == event:
                    marker.set()
        stick.when_right = handler
        # Don't assign when_down handler
        stick.when_left = handler
        for event in events:
            stick_device.write(make_event(event))
        assert markers[0].wait(1)
        assert markers[1].wait(1)
        assert not markers[2].wait(0)
        assert not markers[3].wait(0)
        assert markers[4].wait(1)
        assert markers[5].wait(1)


def test_stick_callbacks_shutdown(stick_device):
    with SenseStick() as stick:
        assert stick._callbacks_thread is None
        directions = {'left', 'right', 'up', 'down', 'enter'}
        for d in directions:
            setattr(stick, 'when_' + d, lambda evt: None)
            assert stick._callbacks_thread is not None
        for d in directions:
            setattr(stick, 'when_' + d, None)
        assert stick._callbacks_thread is None


def test_stick_callbacks_warning(stick_device):
    with SenseStick() as stick:
        stick.when_right = lambda evt: None
        with warnings.catch_warnings(record=True) as w:
            stick.read(0)
            assert len(w) == 1
            assert w[0].category == SenseStickCallbackRead


def test_stick_attrs(stick_device):
    with SenseStick() as stick:
        directions = {'left', 'right', 'up', 'down', 'enter'}
        assert tuple(getattr(stick, d) for d in directions) == (
            False,) * len(directions)
        assert tuple(getattr(stick, d + '_held') for d in directions) == (
            False,) * len(directions)
        for d in directions:
            stick_device.write(
                make_event(StickEvent(datetime.now(), d, True, False)))
            stick.read()
            assert tuple(getattr(stick, d) for d in directions) == tuple(
                e == d for e in directions)
            assert tuple(getattr(stick, d + '_held') for d in directions) == (
                False,) * len(directions)
            stick_device.write(
                make_event(StickEvent(datetime.now(), d, True, True)))
            stick.read()
            assert tuple(getattr(stick, d) for d in directions) == tuple(
                e == d for e in directions)
            assert tuple(getattr(stick, d) for d in directions) == tuple(
                e == d for e in directions)
            stick_device.write(
                make_event(StickEvent(datetime.now(), d, False, True)))
            stick.read()
            assert tuple(getattr(stick, d) for d in directions) == (
                False,) * len(directions)
            assert tuple(getattr(stick, d + '_held') for d in directions) == (
                False,) * len(directions)
