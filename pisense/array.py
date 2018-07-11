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
Defines the :class:`ScreenArray` class for representing the RGB pixel array
as a specialized :class:`~numpy.ndarray`.
"""

from __future__ import (
    unicode_literals,
    absolute_import,
    print_function,
    division,
)

import sys
import os
import time
import struct
import fcntl
import termios

from pkg_resources import require, VersionConflict
try:
    # Check whether we're dealing with an old numpy version which doesn't
    # implement ndarray.__array_ufunc__; in this case we have to do something
    # rather different to implement standard binary operations for ScreenArray
    require('numpy>=1.13')
    _has_array_ufunc = True
except VersionConflict:
    _has_array_ufunc = False
except DistributionNotFound:
    # This will occur on RTD where we've mocked numpy; pretend we've got a
    # recent version of numpy installed
    _has_array_ufunc = True
import numpy as np
from colorzero import Color

from .formats import (
    color_dtype,
    buf_to_rgb,
    iter_to_rgb,
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

    * A single :class:`~colorzero.Color`. The resulting array will have the
      specified *shape*.

    * A list of :class:`~colorzero.Color` values. The resulting array will have
      the specified *shape*.

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
        result = result.view(color_dtype, ScreenArray)
    return result


class ScreenArray(np.ndarray):
    """
    By some curious numpy magic, anything I write here disappears from the
    __doc__ property.
    """
    # pylint: disable=too-few-public-methods

    def __new__(cls, shape=(8, 8)):
        # pylint: disable=protected-access
        result = np.ndarray.__new__(cls, shape=shape, dtype=color_dtype)
        result._screen = None
        return result

    @staticmethod
    def _to_ndarray(v):
        return (
            np.ascontiguousarray(v).view(np.float32, np.ndarray).reshape(v.shape + (3,))
            if isinstance(v, np.ndarray) and v.dtype == color_dtype else
            np.ascontiguousarray(v).view(v.dtype, np.ndarray).reshape(v.shape)
            if isinstance(v, np.ndarray) else
            v
        )

    @classmethod
    def _from_ndarray(cls, v):
        if (
                isinstance(v, np.ndarray) and
                v.dtype == np.float32 and
                len(v.shape) == 3 and
                v.shape[-1] == 3):
            return v.view(color_dtype, cls).squeeze()
        return v

    if _has_array_ufunc:
        # XXX For numpy >=1.13; this ensures all ufuncs called against a
        # ScreenArray will treat the array as a 3-dimensional array of single
        # precision floats (compatible with just about everything) instead of
        # a 2-dimensional array of structures (which no ufunc can deal with)

        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            inputs = [self._to_ndarray(v) for v in inputs]
            try:
                v, = kwargs['out']
            except (KeyError, ValueError):
                pass
            else:
                kwargs['out'] = (self._to_ndarray(v),)
            result = super(ScreenArray, self).__array_ufunc__(
                ufunc, method, *inputs, **kwargs)
            return self._from_ndarray(result)

    else:
        # XXX For numpy <1.13; unfortunately there doesn't seem to be a
        # universal way of handling ufunc overrides prior to 1.13 (and Raspbian
        # Jessie and Stretch are both below this version). However, we can
        # override the standard operators and some common methods to provide
        # similar functionality which is probably good enough for the majority
        # of use-cases

        @classmethod
        def _call_ufunc(cls, ufunc, *inputs, **kwargs):
            inputs = [cls._to_ndarray(v) for v in inputs]
            try:
                v, = kwargs['out']
            except (KeyError, ValueError):
                pass
            else:
                kwargs['out'] = (cls._to_ndarray(v),)
            return cls._from_ndarray(ufunc(*inputs, **kwargs))

        __add__       = lambda self, other: self._call_ufunc(np.add, self, other)
        __sub__       = lambda self, other: self._call_ufunc(np.subtract, self, other)
        __mul__       = lambda self, other: self._call_ufunc(np.multiply, self, other)
        __truediv__   = lambda self, other: self._call_ufunc(np.true_divide, self, other)
        __floordiv__  = lambda self, other: self._call_ufunc(np.floor_divide, self, other)
        __mod__       = lambda self, other: self._call_ufunc(np.mod, self, other)
        __pow__       = lambda self, other: self._call_ufunc(np.power, self, other)

        __radd__      = lambda self, other: self._call_ufunc(np.add, other, self)
        __rsub__      = lambda self, other: self._call_ufunc(np.subtract, other, self)
        __rmul__      = lambda self, other: self._call_ufunc(np.multiply, other, self)
        __rtruediv__  = lambda self, other: self._call_ufunc(np.true_divide, other, self)
        __rfloordiv__ = lambda self, other: self._call_ufunc(np.floor_divide, other, self)
        __rmod__      = lambda self, other: self._call_ufunc(np.mod, other, self)
        __rpow__      = lambda self, other: self._call_ufunc(np.power, other, self)

        __iadd__      = lambda self, other: self._call_ufunc(np.add, self, other, out=(self,))
        __isub__      = lambda self, other: self._call_ufunc(np.subtract, self, other, out=(self,))
        __imul__      = lambda self, other: self._call_ufunc(np.multiply, self, other, out=(self,))
        __itruediv__  = lambda self, other: self._call_ufunc(np.true_divide, self, other, out=(self,))
        __ifloordiv__ = lambda self, other: self._call_ufunc(np.floor_divide, self, other, out=(self,))
        __imod__      = lambda self, other: self._call_ufunc(np.mod, self, other, out=(self,))
        __ipow__      = lambda self, other: self._call_ufunc(np.power, self, other, out=(self,))

        __neg__       = lambda self: self._call_ufunc(np.negative, self)
        __pos__       = lambda self: self
        __abs__       = lambda self: self._call_ufunc(np.abs, self)

        __lt__ = lambda self, other: self._call_ufunc(np.less, self, other)
        __le__ = lambda self, other: self._call_ufunc(np.less_equal, self, other)
        __eq__ = lambda self, other: self._call_ufunc(np.equal, self, other)
        __ne__ = lambda self, other: self._call_ufunc(np.not_equal, self, other)
        __gt__ = lambda self, other: self._call_ufunc(np.greater, self, other)
        __ge__ = lambda self, other: self._call_ufunc(np.greater_equal, self, other)

        # XXX Python 2.7 compat
        __div__ = __truediv__
        __rdiv__ = __rtruediv__
        __idiv__ = __itruediv__

        def clip(self, a_min, a_max, out=None):
            if out is not None:
                out = self._to_ndarray(out)
            result = self._to_ndarray(self).clip(a_min, a_max, out)
            return self._from_ndarray(result)

    # TODO implement matmul for image transforms?

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

    def __setslice__(self, i, j, sequence): # pragma: no cover
        # pylint: disable=protected-access
        super(ScreenArray, self).__setslice__(i, j, sequence)
        if self._screen:
            orig = self
            while orig.shape != (8, 8) and orig.base is not None:
                orig = orig.base
            self._screen.array = orig

    @staticmethod
    def _term_supports_color():
        try:
            stdout_fd = sys.stdout.fileno()
        except (AttributeError, IOError) as exc:
            return False
        else:
            is_a_tty = os.isatty(stdout_fd)
            is_windows = sys.platform.startswith('win')
            return is_a_tty and not is_windows

    @staticmethod
    def _term_size():
        "Returns the size (cols, rows) of the console"
        try:
            buf = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, '12345678')
            row, col = struct.unpack(native_str('hhhh'), buf)[0:2]
            return (col, row)
        except IOError:
            # Don't try and get clever with ctermid; this can work but gives
            # false readings under things like IDLE. Just try the environment
            # and fall back to a sensible default if that fails
            try:
                return (int(os.environ['COLUMNS']),
                        int(os.environ['LINES']))
            except KeyError:
                return (80, 24)

    def __format__(self, format_spec):
        # Parse the format_spec; we don't calculate defaults for colors and
        # width unless they're not explicitly specified
        elements = '\u2588\u2588'  # ██
        colors = None
        width = None
        overflow = '\u00BB'  # »
        for section in format_spec.split(':'):
            section = section.lstrip()
            if not section.rstrip():
                pass
            elif section.startswith('e'):
                elements = section[1:]
            elif section.startswith('c'):
                colors = section[1:].strip().lower()
            elif section.startswith('w'):
                width = int(section[1:].strip())
            elif section.startswith('o'):
                overflow = section[1:]
            else:
                raise ValueError('invalid section in array format spec: %s' %
                                 section)
        if colors is None:
            colors = '16m' if self._term_supports_color() else '0'
        if width is None:
            width = self._term_size()[0]
        if len(elements) * self.shape[1] > width:
            x_limit = (width - len(overflow)) // len(elements)
        else:
            x_limit = None
            overflow = ''
        if colors == '0':
            space = ' ' * len(elements)
            return '\n'.join(
                ''.join(
                    elements if Color(*c).lightness >= 1/4 else space
                    for c in row[:x_limit]
                ) + overflow
                for row in self
            )
        else:
            return '\n'.join(
                ''.join(
                    '{color_dtype:{colors}}{elements}{color_dtype:0}'.format(
                        color_dtype=Color(*c), colors=colors, elements=elements)
                    for c in row[:x_limit]
                ) + overflow
                for row in self
            )

    def show(self, element=None, colors=None, width=None,
             overflow=None):
        """
        By some curious numpy magic, anything I write here disappears from the
        __doc__ property.
        """
        specs = []
        if element is not None:
            specs.append('e' + element)
        if colors is not None:
            specs.append('c' + str(colors))
        if width is not None:
            specs.append('w' + str(width))
        if overflow is not None:
            specs.append('o' + overflow)
        print('{self:{spec}}'.format(self=self, spec=':'.join(specs)))

    def copy(self, order='C'):
        # pylint: disable=missing-docstring,protected-access
        result = super(ScreenArray, self).copy(order)
        result._screen = None
        return result
