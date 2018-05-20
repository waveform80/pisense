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
Defines the :class:`ScreenArray` and :class:`SenseScreen` classes for
controlling and manipulating the RGB pixel array on the Sense HAT.
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
import mmap
import errno
import time
import struct
import fcntl

import numpy as np
from colorzero import Color

from .easings import linear
from .anim import scroll_text, fade_to, slide_to, zoom_to
from .formats import (
    color,
    buf_to_image,
    buf_to_rgb,
    iter_to_rgb,
    image_to_rgb565,
    rgb565_to_image,
    rgb_to_rgb565,
    rgb565_to_rgb,
)

# Make Py2's str and range equivalent to Py3's
native_str = str  # pylint: disable=invalid-name
str = type('')  # pylint: disable=redefined-builtin,invalid-name

DEFAULT_GAMMA = [0, 0, 0, 0, 0, 0, 1, 1,
                 2, 2, 3, 3, 4, 5, 6, 7,
                 8, 9, 10, 11, 12, 14, 15, 17,
                 18, 20, 21, 23, 25, 27, 29, 31]

LOW_GAMMA = [0, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 2, 2, 2,
             3, 3, 3, 4, 4, 5, 5, 6,
             6, 7, 7, 8, 8, 9, 10, 10]


def array(data=None, shape=(8, 8)):
    """
    Use this function to construct a new :class:`ScreenArray` and fill it with
    an initial source of *data*, which can be:

    * A single :class:`Color`. The resulting array will have the specified
      *shape*.

    * A list of :class:`Color` values. The resulting array will have the
      specified *shape*.

    * An :class:`~PIL.Image.Image`. The resulting array will have the shape
      of the image (the *shape* parameter is ignored).

    * Any compatible :class:`~numpy.ndarray`. In this case the shape of the
      array is preserved (the *shape* parameter is ignored).
    """
    if data is None:
        result = ScreenArray(shape)
        result[...] = 0
    elif isinstance(data, Color):
        result = ScreenArray(shape)
        result[...] = data
    else:
        try:
            result = buf_to_rgb(data)
        except TypeError:
            result = iter_to_rgb(data, shape)
        result = result.view(color, ScreenArray)
    return result


class ScreenArray(np.ndarray):
    """
    The :class:`ScreenArray` class is a descendant of :class:`~numpy.ndarray`
    with customizations to make working with the Sense HAT screen a little
    easier.

    Instances of this class should *not* be created directly. Rather, obtain
    the current state of the screen from :attr:`SenseScreen.array` or use the
    :func:`array` function to create instances from a variety of sources (a PIL
    :class:`~PIL.Image.Image`, another array, a list of
    :class:`~colorzero.Color` instances, etc).

    In most respects, a :class:`ScreenArray` will act like any other numpy
    array but for the following exceptions:

    Display Association
    ~~~~~~~~~~~~~~~~~~~

    If the instance was obtained from :attr:`SenseScreen.array` it will be
    "associated" with the display. Manipulating the content of the array will
    manipulate the appearance of the display.

    Copying a screen array that is associated with a display (via the
    :meth:`~numpy.ndarray.copy` method) breaks the association. This is a
    convenient way to take a copy of the current display, fiddle around with it
    without intermediate states displaying, and then update the display by
    copying it back::

        >>> from pisense import *
        >>> hat = SenseHAT()
        >>> arr = hat.screen.array.copy()
        >>> # Mess around with arr here
        >>> hat.screen.array = arr

    Operations in numpy that create a new array will also break the display
    association (e.g. adding two arrays together to create a new array; the new
    array will not "derive" its display association from the original arrays).
    However, operations that *don't* create a new array (e.g.  slicing,
    flipping, etc.) will normally maintain the association. This is why you can
    update portions of the display using slices.

    Data Type
    ~~~~~~~~~

    The data-type is fixed and cannot be altered. Specifically the data-type is
    a 3-tuple of single-precision floating point values between 0.0 and 1.0,
    labelled "r", "g" and "b".

    The 0.0 to 1.0 range of color values is *not* enforced.  Hence if you add
    two screen arrays together you may wind up with values greater than 1.0 in
    one or more color planes. This is deliberate as intermediate values
    exceeding this range can be useful in some calculations.

    .. hint::

        The numpy :meth:`~numpy.ndarray.clip` method is a convenient way of
        limiting values to the 0.0 to 1.0 range before updating the display.

    Format Strings
    ~~~~~~~~~~~~~~

    Screen arrays can be used in format strings to preview the state of the
    display visually. The format string specification for screen arrays
    consists of a string used to represent each element optionally followed by
    a colon and a terminal type description derived from the `colorzero`_
    library's :ref:`format strings <~colorzero.format-strings>`. For example::

    .. attention:: TODO finalize format string spec
    """
    # pylint: disable=too-few-public-methods

    def __new__(cls, shape=(8, 8)):
        # pylint: disable=protected-access
        result = np.ndarray.__new__(cls, shape=shape, dtype=color)
        result._screen = None
        return result

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = [
            np.ascontiguousarray(v).view(np.float32, np.ndarray).reshape(v.shape + (3,))
            if isinstance(v, np.ndarray) and v.dtype == color else
            np.ascontiguousarray(v).view(v.dtype, np.ndarray).reshape(v.shape)
            if isinstance(v, np.ndarray) else
            v
            for v in inputs
        ]
        try:
            v, = kwargs['out']
        except (KeyError, ValueError):
            pass
        else:
            kwargs['out'] = (
                np.ascontiguousarray(v).view(np.float32, np.ndarray).reshape(v.shape + (3,)),
            )
        result = super(ScreenArray, self).__array_ufunc__(
            ufunc, method, *inputs, **kwargs)
        if (
                isinstance(result, np.ndarray) and
                result.dtype == np.float32 and
                len(result.shape) == 3 and
                result.shape[-1] == 3):
            result = result.view(color, self.__class__).squeeze()
        return result

    def __array_finalize__(self, obj):
        # pylint: disable=protected-access,attribute-defined-outside-init
        if obj is None:
            return
        self._screen = getattr(obj, '_screen', None)

    def __setitem__(self, index, value):
        # pylint: disable=protected-access
        super(ScreenArray, self).__setitem__(index, value)
        if self._screen:
            # If we're a slice of the original pixels value, find the parent
            # that contains the complete array and send that to the setter
            orig = self
            while orig.shape != (8, 8) and orig.base is not None:
                orig = orig.base
            self._screen.array = orig

    def __setslice__(self, i, j, sequence):
        # pylint: disable=protected-access
        super(ScreenArray, self).__setslice__(i, j, sequence)
        if self._screen:
            orig = self
            while orig.shape != (8, 8) and orig.base is not None:
                orig = orig.base
            self._screen.array = orig

    def __format__(self, format_spec):
        if format_spec.endswith('p'):
            char = format_spec[:-1]
            return '\n'.join(
                ''.join('{0:16m}{1}{0:0}'.format(Color(*elem), char)
                        for elem in row)
                for row in self
            )
        else:
            char = format_spec[:-1]
            space = ' ' * len(char)
            return '\n'.join(
                ''.join(char if Color(*elem).lightness >= 0.33333 else space
                        for elem in row)
                for row in self
            )

    def preview(self, element='\u2588\u2588', colors=None, width=None):
        """
        Print a preview of the screen to the console.

        The *element* parameter specifics the string used to represent each
        element of the display. This defaults to "██" (two Unicode full block
        drawing characters) which is usually sufficient to provide a fairly
        accurate representation of the screen.

        The *colors* parameter indicates the sort of ANSI coding (if any) that
        should be used to depict the colors of the display. The following
        values are accepted:

        +-------+-------------------------------------------------------+
        | Value | Description                                           |
        +=======+=======================================================+
        | 16m   | Use true-color ANSI codes capable of representing ≈16 |
        |       | million colors. This is the default if stdout is a    |
        |       | TTY. The default terminal in Raspbian supports this   |
        |       | style of ANSI code.                                   |
        +-------+-------------------------------------------------------+
        | 256   | Use 256-color ANSI codes. Most modern terminals       |
        |       | (including Raspbian's default terminal) support this  |
        |       | style of ANSI code.                                   |
        +-------+-------------------------------------------------------+
        | 8     | Use old-style DOS ANSI codes, only capable of         |
        |       | representing 8 colors. There is rarely a need to      |
        |       | resort to this setting.                               |
        +-------+-------------------------------------------------------+
        | 1     | Don't use ANSI color codes. Instead, any pixel values |
        |       | with a brightness >33% (an arbitrary cut-off) will be |
        |       | displayed, while darker pixels will be rendered as    |
        |       | spaces. This is the default if stdout is *not* a TTY. |
        +-------+-------------------------------------------------------+

        The *width* parameter specifies the maximum width for the output. This
        defaults to ``None`` which means the method will attempt to use the
        terminal's width (if this can be determined; if it cannot, then 80 will
        be used as a fallback). Pixels beyond the specified width will be
        excluded from the output.
        """
        # TODO detect console tty state for default color/no color
        # TODO detect console width for clipping
        pass

    def copy(self, order='C'):
        # pylint: disable=missing-docstring,protected-access
        result = super(ScreenArray, self).copy(order)
        result._screen = None
        return result


class SenseScreen(object):
    """
    The :class:`SenseScreen` class represents the LED matrix on the Sense HAT.
    Users can either instantiate this class themselves, or can access an
    instance from :attr:`SenseHAT.screen`.

    The two primary means of accessing and manipulating the screen are:

    * The :attr:`array` attribute which returns a :class:`ScreenArray` (a
      customized :class:`~numpy.ndarray`). If the array is manipulated, it
      will update the screen "live".

    * The :meth:`image` and :meth:`draw` methods. The former returns the
      current state of the display as an 8x8 PIL :class:`~PIL.Image.Image`,
      while the latter updates the screen to display the provided image.

    Attributes are provided to modify the :attr:`rotation` of the display, and
    the :attr:`gamma` table. The :attr:`hflip` and :attr:`vflip` attributes can
    be used to mirror the display horizontally and vertically. Finally, several
    methods are provided for playing animations: :meth:`slide_to`,
    :meth:`fade_to`, :meth:`zoom_to`, and :meth:`scroll_text` each of which
    accept either image or array representations of the screen.
    """
    # pylint: disable=too-many-instance-attributes

    __slots__ = (
        '_fb_file',
        '_fb_mmap',
        '_fb_array',
        '_array',
        '_hflip',
        '_vflip',
        '_rotation',
        'fps',
        'easing',
    )

    SENSE_HAT_FB_NAME = 'RPi-Sense FB'

    GET_GAMMA = 61696
    SET_GAMMA = 61697
    RESET_GAMMA = 61698
    GAMMA_DEFAULT = 0
    GAMMA_LOW = 1
    GAMMA_USER = 2

    def __init__(self, fps=15, easing=linear):
        self._fb_file = io.open(self._fb_device(), 'wb+')
        self._fb_mmap = mmap.mmap(self._fb_file.fileno(), 128)
        self._fb_array = np.frombuffer(self._fb_mmap, dtype=np.uint16).reshape((8, 8))
        self._array = ScreenArray()
        # pylint: disable=protected-access
        self._array._screen = self
        self._hflip = False
        self._vflip = False
        self._rotation = 0
        self.fps = fps
        self.easing = easing

    def close(self):
        """
        Call the :meth:`close` method to close the screen interface and free
        up any background resources. The method is idempotent (you can call it
        multiple times without error) and after it is called, any operations on
        the screen may return an error (but are not guaranteed to do so).
        """
        if self._fb_array is not None:
            self._fb_array = None
            self._fb_mmap.close()
            self._fb_file.close()
            # pylint: disable=protected-access
            self._array._screen = None
            self._array = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

    def _fb_device(self):
        for device in glob.glob('/sys/class/graphics/fb*'):
            try:
                with io.open(os.path.join(device, 'name'), 'r') as f:
                    if f.read().strip() == self.SENSE_HAT_FB_NAME:
                        return os.path.join('/dev', os.path.basename(device))
            except IOError as exc:
                if exc.errno != errno.ENOENT:
                    raise
        raise RuntimeError('unable to locate SenseHAT framebuffer device')

    @property
    def raw(self):
        """
        Provides direct access to the Sense HAT's RGB565 framebuffer.

        This attribute returns a numpy :class:`~numpy.ndarray` containing 8x8
        unsigned 16-bit integer elements, each of which represents a single
        pixel on the display in RGB565 format (5-bits for red, 6-bits for
        green, 5-bits for blue). Internally, the screen actually uses 5-bits
        for all colors (the LSB of green is dropped); see :attr:`gamma` for
        more information.

        The array that is returned is built upon the framebuffer's memory. In
        other words, manipulating the array directly mainpulates the
        framebuffer. As such, this property will *not* be affected by
        :attr:`hflip`, :attr:`vflip` or :attr:`rotation`.

        .. note::

            Generally you should have no need to use this property. The
            :attr:`array` attribute and :meth:`image` method are far simpler to
            work with.
        """
        return self._fb_array

    @raw.setter
    def raw(self, value):
        self._fb_array[:] = value

    @property
    def gamma(self):
        """
        Returns the gamma lookup table for the screen.

        This property returns a 32-element array of integer values each of
        which is in the range 0 to 31 (5-bits). This forms the "gamma table" of
        the Sense HAT's screen and is used to map intensities to their final
        values on the screen.

        Internally, the Sense HAT's framebuffer uses 5-bits (values 0 to 31) to
        represent each color. After a color's `Least significant bits` have
        been stripped to reduce it to 5-bits, the resulting value is then used
        as an index into this list. The value obtained from this lookup will be
        the final value used when lighting the corresponding LED.

        Two "standard" gamma tables are provided: :data:`DEFAULT_GAMMA` and
        :data:`LOW_GAMMA` which can be assigned directly to this property::

            >>> import pisense
            >>> hat = pisense.SenseHAT()
            >>> hat.screen.gamma = pisense.LOW_GAMMA

        .. note::

            This property is designed to be assigned to in its entirety. The
            list returned by it is not "live" (it is a copy of the actual gamma
            table) and changing individual elements in it will *not* change the
            gamma settings.

        .. _Least significant bits: https://en.wikipedia.org/wiki/Bit_numbering
        """
        buf = bytearray(32)
        fcntl.ioctl(self._fb_file, SenseScreen.GET_GAMMA, buf)
        return list(buf)

    @gamma.setter
    def gamma(self, value):
        if value is None:
            fcntl.ioctl(self._fb_file, SenseScreen.RESET_GAMMA, 0)
        else:
            if len(value) != 32:
                raise ValueError('gamma array must contain 32 entries')
            if not all(0 <= v < 32 for v in value):
                raise ValueError('gamma values must be in the range 0..31')
            buf = struct.pack(native_str('32B'), *value)
            fcntl.ioctl(self._fb_file, SenseScreen.SET_GAMMA, buf)

    @property
    def array(self):
        """
        Returns the screen as a :class:`ScreenArray`. The returned array is
        "live" and modifications to it will modify the state of the screen.
        See :class:`ScreenArray` for more information on the usage and
        facilities of this class.
        """
        arr = self._array
        # pylint: disable=protected-access
        arr._screen = None
        try:
            rgb565_to_rgb(self.raw, arr)
            arr = self._undo_transforms(arr)
        finally:
            arr._screen = self
        return arr

    @array.setter
    def array(self, value):
        if isinstance(value, np.ndarray):
            value = value.view(color).reshape((8, 8))
        else:
            value = np.array(value, dtype=color).reshape((8, 8))
        value = self._apply_transforms(value)
        rgb_to_rgb565(value, self.raw)

    @property
    def vflip(self):
        """
        When set to ``True`` the display will be mirrored vertically. Defaults
        to ``False``.
        """
        return self._vflip

    @vflip.setter
    def vflip(self, value):
        raw = self._undo_transforms(self.raw)
        self._vflip = bool(value)
        self.raw = self._apply_transforms(raw)

    @property
    def hflip(self):
        """
        When set to ``True`` the display will be mirrored horizontally.
        Defaults to ``False``.
        """
        return self._hflip

    @hflip.setter
    def hflip(self, value):
        raw = self._undo_transforms(self.raw)
        self._hflip = bool(value)
        self.raw = self._apply_transforms(raw)

    @property
    def rotation(self):
        """
        Specifies the rotation (really, the orientation) of the screen as a
        multiple of 90 degrees.

        When rotation is 0 (the default), Y is 0 near the GPIO pins and
        increases towards the Raspberry Pi logo, while X is 0 near the notch at
        the edge of the board and increases towards the joystick:

        .. image:: images/rotation_0.*

        When rotation is 90, Y is 0 near the notch at the edge of the board and
        increases towards the joystick, while X is 0 near the Raspberry Pi
        logo, and increases towards the GPIO pins:

        .. image:: images/rotation_90.*

        The other two rotations are trivial to derive from this.
        """
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        if value % 90:
            raise ValueError('rotation must be a multiple of 90')
        raw = self._undo_transforms(self.raw)
        self._rotation = value % 360
        self.raw = self._apply_transforms(raw)

    def _apply_transforms(self, arr):
        if self._vflip:
            arr = np.flipud(arr)
        if self._hflip:
            arr = np.fliplr(arr)
        arr = np.rot90(arr, self._rotation // 90)
        return arr

    def _undo_transforms(self, arr):
        arr = np.rot90(arr, (360 - self._rotation) // 90)
        if self._hflip:
            arr = np.fliplr(arr)
        if self._vflip:
            arr = np.flipud(arr)
        return arr

    def clear(self, fill=Color('black')):
        """
        Set all pixels in the display to the same *fill* color, which defaults
        to black (off). *fill* can be a :class:`~colorzero.Color` instance, or
        anything that could be used to construct a :class:`~colorzero.Color`
        instance.
        """
        if not isinstance(fill, Color):
            fill = Color(*fill)
        self.raw = fill.rgb565

    def image(self):
        """
        Return an 8x8 PIL :class:`~PIL.Image.Image` representing the current
        state of the display.

        The image returned is a copy of the display's state. Drawing on the
        image will *not* update the display. Instead, it is recommended that
        you perform whatever drawing you wish (e.g. with
        :class:`~PIL.ImageDraw.ImageDraw`), then call :meth:`draw` with the
        image to update the display.
        """
        arr = self._undo_transforms(self.raw)
        arr = rgb565_to_image(arr)
        return arr

    def draw(self, image):
        """
        Draw the provided image (or array) on the display.

        The *image* passed to this method can be anything accepted by
        :func:`buf_to_image`. The only restriction is that the result must be
        an 8x8 image.
        """
        img = buf_to_image(image)
        if img.size != (8, 8):
            raise ValueError('image must be an 8x8 RGB PIL Image')
        arr = image_to_rgb565(img)
        arr = self._apply_transforms(arr)
        self.raw = arr

    def play(self, frames):
        """
        Play an animation on the display.

        The *frames* provided to this method must be in one of the formats
        accepted by the :attr:`draw` method; *frames* itself can be any
        iterable, including a generator. Frames will be played back at a rate
        governed by the :attr:`fps` attribute.
        """
        delay = 1 / self.fps
        for frame in frames:
            if (isinstance(frame, np.ndarray) and
                    frame.shape == (8, 8) and
                    frame.dtype == np.uint16):
                # Fast-path
                self.raw = self._apply_transforms(frame)
            else:
                self.draw(frame)
            time.sleep(delay)

    def scroll_text(self, text, font='default.pil', size=8,
                    foreground=Color('white'), background=Color('black'),
                    direction='left', duration=None, fps=None):
        """
        Renders *text* in the specified *font* and *size*, and scrolls the
        result across the display.

        See the :func:`scroll_text` function for more information on the
        meaning of the parameters. This method simply calls that function with
        the provided parameters, and passes the result to :meth:`play`.
        """
        # pylint: disable=too-many-arguments
        frames = scroll_text(text, font, size, foreground, background,
                             direction, duration,
                             self.fps if fps is None else fps)
        # Pre-calc all the frames in the raw RGR565 format; doesn't take a huge
        # amount of memory and ensures a smooth playback even on tiny
        # platforms like the A+
        frames = [image_to_rgb565(frame) for frame in frames]
        self.play(frames)

    def fade_to(self, image, duration=1, fps=None, easing=None):
        """
        Smoothly fades the display from its current state to the provided
        *image* (which can be anything compatible with :meth:`draw`).

        See the :func:`fade_to` function for more information on the meaning of
        the parameters. This method simply calls that function with the current
        state of the display (via :meth:`image`) and the provided parameters,
        and passes the result to :meth:`play`.
        """
        frames = fade_to(self.image(), image, duration,
                         self.fps if fps is None else fps,
                         self.easing if easing is None else easing)
        frames = [image_to_rgb565(frame) for frame in frames]
        self.play(frames)

    def slide_to(self, image, direction='left', cover=False, duration=1,
                 fps=None, easing=None):
        """
        Slide *image* (which can be anything compatible with :meth:`draw`) over
        the display in the specified *direction*.

        See the :func:`slide_to` function for more information on the meaning
        of the parameters. This method simply calls that function with the
        current state of the display (via :meth:`image`) and the provided
        parameters, and passes the result to :meth:`play`.
        """
        # pylint: disable=too-many-arguments
        frames = slide_to(self.image(), image, direction, cover, duration,
                          self.fps if fps is None else fps,
                          self.easing if easing is None else easing)
        frames = [image_to_rgb565(frame) for frame in frames]
        self.play(frames)

    def zoom_to(self, image, center=(4, 4), direction='in', duration=1,
                fps=None, easing=None):
        """
        Zoom the display in or out (specified by *direction*) to *image* (which
        can be anything compatible with :meth:`draw`).

        See the :func:`zoom_to` function for more information on the meaning of
        the parameters. This method simply calls that function with the current
        state of the display (via :meth:`image`) and the provided parameters,
        and passes the result to :meth:`play`.
        """
        # pylint: disable=too-many-arguments
        frames = zoom_to(self.image(), image, center,
                         direction, duration,
                         self.fps if fps is None else fps,
                         self.easing if easing is None else easing)
        frames = [image_to_rgb565(frame) for frame in frames]
        self.play(frames)
