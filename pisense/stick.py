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
Defines the :class:`SenseStick` class representing the Sense HAT's joystick.
"""

from __future__ import (
    unicode_literals,
    absolute_import,
    print_function,
    division,
)

import io
import os
import glob
import errno
import struct
import select
import warnings
import termios
from datetime import datetime
from collections import namedtuple
from threading import Thread, Event, Lock
try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty

from .exc import SenseStickBufferFull, SenseStickCallbackRead

# native_str represents the "native" str type (bytes in Py 2, unicode in Py 3)
# of the interpreter; str is then redefined to always represent unicode
native_str = str  # pylint: disable=invalid-name
str = type('')  # pylint: disable=redefined-builtin,invalid-name


class StickEvent(namedtuple('StickEvent',
                            ('timestamp', 'direction', 'pressed', 'held'))):
    """
    Represents a joystick event as a :func:`~collections.namedtuple`. The
    fields of the event are defined below:

    .. attribute:: timestamp

        A :class:`~datetime.datetime` object specifying when the event took
        place. This timestamp is derived from the kernel event so it should be
        accurate even when callbacks have taken time reacting to events. The
        timestamp is a naive :class:`~datetime.datetime` object in local time.

    .. attribute:: direction

        A string indicating which direction the event pertains to. Can be one
        of "up", "down", "leftʺ, "right", or "enter" (the last event refers to
        the joystick being pressed inward).

    .. attribute:: pressed

        A bool which is ``True`` when the event indicates that the joystick is
        being pressed *or held* in the specified direction. When this is
        ``False``, the event indicates that the joystick has been released from
        the specified direction.

    .. attribute:: held

        A bool which is ``True`` when the event indicates that the joystick
        direction is currently held down (when :attr:`pressed` is also
        ``True``) or that the direction was *previously* held down (when
        :attr:`pressed` is ``False`` buut :attr:`held` is still ``True``).
    """
    __slots__ = ()


class SenseStick(object):
    """
    The :class:`SenseStick` class represents the joystick on the Sense HAT.
    Users can either instantiate this class themselves, or can access an
    instance from :attr:`SenseHAT.stick`.

    The :meth:`read` method can be called to obtain :class:`StickEvent`
    instances, or the instance can be treated as an iterator in which case
    events will be yielded as they come in::

        hat = SenseHAT()
        for event in hat.stick:
            if event.pressed and not event.held:
                print('%s pressed!' % event.direction)

    Alternatively, handler functions can be assigned to the attributes
    :attr:`when_up`, :attr:`when_down`, :attr:`when_left`, :attr:`when_right`,
    :attr:`when_enter`. The assigned functions will be called when any matching
    event occurs.

    Finally, the attributes :attr:`up`, :attr:`down`, :attr:`left`,
    :attr:`right`, and attr:`enter` can be polled to determine the current
    state of the joystick.

    The :attr:`rotation` attribute can be modified to alter the orientation of
    events, and the aforementioned attributes.

    The *max_events* parameter controls the size of the internal queue used to
    buffer joystick events. This defaults to 100 which should be more than
    sufficient to ensure events are not lost. The *flush_input* parameter,
    which defaults to ``True`` controls whether, when the instance is closed,
    it attempts to flush the stdin of the owning terminal. This is useful as
    the joystick also acts as a keyboard. On the command line, this can mean
    that joystick movements (buffered during a script's execution) can
    inadvertently execute historical commands (e.g. Up a few times followed by
    Enter).

    Finally, if the *emulate* parameter is ``True``, the instance will connect
    to the joystick in the `desktop Sense HAT emulator`_ instead of the "real"
    Sense HAT joystick.

    .. _desktop Sense HAT emulator: https://sense-emu.readthedocs.io/
    """
    # pylint: disable=too-many-instance-attributes

    __slots__ = (
        '_flush',
        '_callbacks_lock',
        '_callbacks_close',
        '_callbacks',
        '_callbacks_thread',
        '_closing',
        '_stream',
        '_buffer',
        '_read_thread',
        '_pressed',
        '_held',
        '_rotation',
        '_rot_map',
    )

    SENSE_HAT_EVDEV_NAME = 'Raspberry Pi Sense HAT Joystick'
    EVENT_FORMAT = native_str('llHHI')
    EVENT_SIZE = struct.calcsize(EVENT_FORMAT)

    EV_KEY = 0x01

    STATE_RELEASE = 0
    STATE_PRESS = 1
    STATE_HOLD = 2

    KEY_UP = 103
    KEY_LEFT = 105
    KEY_RIGHT = 106
    KEY_DOWN = 108
    KEY_ENTER = 28

    def __init__(self, max_events=100, flush_input=True, emulate=False):
        self._flush = flush_input
        self._callbacks_lock = Lock()
        self._callbacks_close = Event()
        self._callbacks = {}
        self._callbacks_thread = None
        self._closing = Event()
        self._stream = False
        self._buffer = Queue(maxsize=max_events)
        if emulate:
            from sense_emu.stick import init_stick_client
            stick_file = init_stick_client()
        else:
            stick_file = io.open(self._stick_device(), 'rb', buffering=0)
        self._read_thread = Thread(target=self._read_stick, args=(stick_file,))
        self._read_thread.daemon = True
        self._read_thread.start()
        # This is just a guess; we can't know the actual joystick state at
        # initialization. However, if it's incorrect, future events should
        # correct this
        self._pressed = set()
        self._held = set()
        self._rotation = 0
        self._rot_map = {
            SenseStick.KEY_UP: SenseStick.KEY_RIGHT,
            SenseStick.KEY_LEFT: SenseStick.KEY_UP,
            SenseStick.KEY_DOWN: SenseStick.KEY_LEFT,
            SenseStick.KEY_RIGHT: SenseStick.KEY_DOWN,
            SenseStick.KEY_ENTER: SenseStick.KEY_ENTER,
        }

    def close(self):
        """
        Call the :meth:`close` method to close the joystick interface and free
        up any background resources. The method is idempotent (you can call it
        multiple times without error) and after it is called, any operations on
        the joystick may return an error (but are not guaranteed to do so).
        """
        if self._read_thread is not None:
            self._closing.set()
            self._read_thread.join()
            if self._callbacks_thread:
                self._callbacks_thread.join()
            self._read_thread = None
            self._callbacks_thread = None
            self._buffer = None
        if self._flush:
            self._flush = False
            try:
                termios.tcflush(0, termios.TCIFLUSH)
            except termios.error:
                # If stdin isn't a tty (or isn't available), ignore the error
                pass

    def __iter__(self):
        while True:
            yield self.read(0 if self._stream else None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

    @staticmethod
    def _stick_device():
        for evdev in glob.glob('/sys/class/input/event*'):
            try:
                with io.open(os.path.join(evdev, 'device', 'name'), 'r') as f:
                    if f.read().strip() == SenseStick.SENSE_HAT_EVDEV_NAME:
                        return os.path.join('/dev', 'input', os.path.basename(evdev))
            except IOError as exc:
                if exc.errno != errno.ENOENT:
                    raise
        raise RuntimeError('unable to locate SenseHAT joystick device')

    def _read_stick(self, stick_file):
        try:
            while not self._closing.wait(0):
                if select.select([stick_file], [], [], 0.1)[0]:
                    event = stick_file.read(SenseStick.EVENT_SIZE)
                    if event == b'':
                        # This is mostly to ease testing, but also deals with
                        # some edge cases
                        break
                    (
                        tv_sec,
                        tv_usec,
                        evt_type,
                        code,
                        value,
                    ) = struct.unpack(SenseStick.EVENT_FORMAT, event)
                    r = self._rotation
                    while r:
                        code = self._rot_map[code]
                        r -= 90
                    if evt_type == SenseStick.EV_KEY:
                        if self._buffer.full():
                            warnings.warn(SenseStickBufferFull(
                                "The internal SenseStick buffer is full; "
                                "try reading some events!"))
                            self._buffer.get()
                        evt = StickEvent(
                            timestamp=datetime.fromtimestamp(
                                tv_sec + (tv_usec / 1000000)
                            ),
                            direction={
                                SenseStick.KEY_UP:    'up',
                                SenseStick.KEY_DOWN:  'down',
                                SenseStick.KEY_LEFT:  'left',
                                SenseStick.KEY_RIGHT: 'right',
                                SenseStick.KEY_ENTER: 'enter',
                            }[code],
                            pressed=(value != SenseStick.STATE_RELEASE),
                            held=(value == SenseStick.STATE_HOLD or (
                                value == SenseStick.STATE_RELEASE and
                                code in self._held))
                        )
                        if not evt.pressed:
                            self._pressed -= {code}
                            self._held -= {code}
                        elif evt.held:
                            self._pressed |= {code}  # to correct state
                            self._held |= {code}
                        else: # pressed
                            self._pressed |= {code}
                            self._held -= {code}  # to correct state
                        # Only push event onto the queue once the internal
                        # state is updated; this ensures the various read-only
                        # properties will be accurate for event handlers that
                        # subsequently fire (although if they take too long the
                        # state may change again before the next handler fires)
                        self._buffer.put(evt)
        finally:
            stick_file.close()

    def _run_callbacks(self):
        while not self._callbacks_close.wait(0) and not self._closing.wait(0):
            try:
                event = self._buffer.get(timeout=0.1)
            except Empty:
                pass
            else:
                with self._callbacks_lock:
                    try:
                        callback = self._callbacks[event.direction]
                    except KeyError:
                        callback = None
                if callback is not None:
                    callback(event)

    def _start_stop_callbacks(self):
        with self._callbacks_lock:
            if self._callbacks and not self._callbacks_thread:
                self._callbacks_close.clear()
                self._callbacks_thread = Thread(target=self._run_callbacks)
                self._callbacks_thread.daemon = True
                self._callbacks_thread.start()
            elif not self._callbacks and self._callbacks_thread:
                self._callbacks_close.set()
                self._callbacks_thread.join()
                self._callbacks_thread = None

    @property
    def rotation(self):
        """
        Specifies the rotation (really, the orientation) of the joystick as a
        multiple of 90 degrees. When rotation is 0 (the default), "up" is
        toward the GPIO pins:

        .. image:: images/rotation_0.*

        When rotation is 90, "up" is towards the LEDs, and so on:

        .. image:: images/rotation_90.*

        The other two rotations are trivial to derive from this.

        .. note::

            This property is updated by the unifying :attr:`SenseHAT.rotation`
            attribute.
        """
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        # TODO If rotation is modified while _pressed and _held aren't empty
        # then we potentially have bad state (unless it's just ENTER);
        # technically we should anti-rotate their current values here...
        if value % 90:
            raise ValueError('rotation must be a multiple of 90')
        self._rotation = value % 360

    def read(self, timeout=None):
        """
        Wait up to *timeout* seconds for another joystick event. If one occurs,
        return it as a :class:`StickEvent`, otherwise return ``None``.

        .. note::

            Attempting to call this method when callbacks are assigned to
            attributes like :attr:`when_left` will trigger a
            :exc:`SenseStickCallbackRead` warning. This is because using the
            callback mechanism causes a background thread to continually read
            joystick events (removing them from the queue that :meth:`read`
            accesses). Mixing these programming styles can result in missing
            events.
        """
        if self._callbacks_thread is not None:
            warnings.warn(SenseStickCallbackRead(
                'read called while when_* callbacks are assigned'))
        try:
            return self._buffer.get(timeout=timeout)
        except Empty:
            return None

    @property
    def stream(self):
        """
        When ``True``, treating the joystick as an iterator will always yield
        immediately (yielding ``None`` if no event has occurred). When
        ``False`` (the default), the iterator will only yield when an event
        has occurred.

        .. note::

            This property can be set while an iterator is active, but if the
            current value is ``False``, the iterator will wait indefinitely
            for the next event *before* it will start returning ``None``. It
            is better to set this property *prior* to obtaining the iterator.
        """
        return self._stream

    @stream.setter
    def stream(self, value):
        self._stream = bool(value)

    @property
    def up(self):
        """
        Returns ``True`` if the joystick is currently pressed upward.
        """
        # pylint: disable=invalid-name
        return SenseStick.KEY_UP in self._pressed

    @property
    def up_held(self):
        """
        Returns ``True`` if the joystick is currently held upward.
        """
        return SenseStick.KEY_UP in self._held

    @property
    def when_up(self):
        """
        The function to call when the joystick is moved upward.
        """
        with self._callbacks_lock:
            return self._callbacks.get('up')

    @when_up.setter
    def when_up(self, value):
        with self._callbacks_lock:
            if value:
                self._callbacks['up'] = value
            else:
                self._callbacks.pop('up', None)
        self._start_stop_callbacks()

    @property
    def down(self):
        """
        Returns ``True`` if the joystick is currently pressed downward.
        """
        return SenseStick.KEY_DOWN in self._pressed

    @property
    def down_held(self):
        """
        Returns ``True`` if the joystick is currently held downward.
        """
        return SenseStick.KEY_DOWN in self._held

    @property
    def when_down(self):
        """
        The function to call when the joystick is moved downward.
        """
        with self._callbacks_lock:
            return self._callbacks.get('down')

    @when_down.setter
    def when_down(self, value):
        with self._callbacks_lock:
            if value:
                self._callbacks['down'] = value
            else:
                self._callbacks.pop('down', None)
        self._start_stop_callbacks()

    @property
    def left(self):
        """
        Returns ``True`` if the joystick is currently pressed leftward.
        """
        return SenseStick.KEY_LEFT in self._pressed

    @property
    def left_held(self):
        """
        Returns ``True`` if the joystick is currently held leftward.
        """
        return SenseStick.KEY_LEFT in self._held

    @property
    def when_left(self):
        """
        The function to call when the joystick is moved leftward.
        """
        with self._callbacks_lock:
            return self._callbacks.get('left')

    @when_left.setter
    def when_left(self, value):
        with self._callbacks_lock:
            if value:
                self._callbacks['left'] = value
            else:
                self._callbacks.pop('left', None)
        self._start_stop_callbacks()

    @property
    def right(self):
        """
        Returns ``True`` if the joystick is currently pressed rightward.
        """
        return SenseStick.KEY_RIGHT in self._pressed

    @property
    def right_held(self):
        """
        Returns ``True`` if the joystick is currently held rightward.
        """
        return SenseStick.KEY_RIGHT in self._held

    @property
    def when_right(self):
        """
        The function to call when the joystick is moved rightward.
        """
        with self._callbacks_lock:
            return self._callbacks.get('right')

    @when_right.setter
    def when_right(self, value):
        with self._callbacks_lock:
            if value:
                self._callbacks['right'] = value
            else:
                self._callbacks.pop('right', None)
        self._start_stop_callbacks()

    @property
    def enter(self):
        """
        Returns ``True`` if the joystick is currently pressed inward.
        """
        return SenseStick.KEY_ENTER in self._pressed

    @property
    def enter_held(self):
        """
        Returns ``True`` if the joystick is currently held inward.
        """
        return SenseStick.KEY_ENTER in self._held

    @property
    def when_enter(self):
        """
        The function to call when the joystick is pressed in or released.
        """
        with self._callbacks_lock:
            return self._callbacks.get('enter')

    @when_enter.setter
    def when_enter(self, value):
        with self._callbacks_lock:
            if value:
                self._callbacks['enter'] = value
            else:
                self._callbacks.pop('enter', None)
        self._start_stop_callbacks()
