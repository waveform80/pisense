===================
API - Screen Arrays
===================

.. currentmodule:: pisense

This chapter covers the :class:`ScreenArray` class, how it should be
constructed, how it can be used to manipulate the Sense HAT's display, and how
to convert it to various different formats.


ScreenArray Class
=================

.. For some reason numpy descendants blank their __doc__ strings so this chunk
.. has to be included manually...

.. class:: ScreenArray(shape=(8, 8))

    The :class:`ScreenArray` class is a descendant of :class:`~numpy.ndarray`
    with customizations to make working with the Sense HAT screen a little
    easier.

    In most respects, a :class:`ScreenArray` will act like any other numpy
    array. Exceptions to the normal behaviour are documented in the following
    sections.

    Instances of this class should *not* be created directly. Rather, obtain
    the current state of the screen from :attr:`SenseScreen.array` or use the
    :func:`array` function to create instances from a variety of sources (a PIL
    :class:`~PIL.Image.Image`, another array, a list of
    :class:`~colorzero.Color` instances, etc).

.. autofunction:: array

Display Association
===================

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
=========

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
==============

Screen arrays can be used in format strings to preview the state of the
display visually. The format string specification for screen arrays
consists of a string used to represent each element optionally followed by
a colon and a terminal type description derived from the `colorzero`_
library's :ref:`format strings <colorzero.format>`. For example::

.. attention:: TODO finalize format string spec

A method is also provided for convenient command line previewing:

.. method:: ScreenArray.show(element='\\u2588\\u2588', colors=None, width=None)

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
    | 2     | Don't use ANSI color codes. Instead, any pixel values |
    |       | with a brightness >33% (an arbitrary cut-off) will be |
    |       | displayed, while darker pixels will be rendered as    |
    |       | spaces. This is the default if stdout is *not* a TTY. |
    +-------+-------------------------------------------------------+

    The *width* parameter specifies the maximum width for the output. This
    defaults to ``None`` which means the method will attempt to use the
    terminal's width (if this can be determined; if it cannot, then 80 will
    be used as a fallback). Pixels beyond the specified width will be
    excluded from the output.


Format conversions
==================

The following conversion functions are provided to facilitate converting
various inputs into something either easy to manipulate or easy to display on
the screen.

.. autofunction:: buf_to_image

.. autofunction:: buf_to_rgb

.. autofunction:: buf_to_rgb888


Advanced conversions
====================

The following conversion functions are used internally by pisense, and are
generally not required unless you want to work with :attr:`SenseScreen.raw`
directly, or you know exactly what formats you are converting between and want
to skip the overhead of the ``buf_to_*`` routines figuring out the input type.

.. autofunction:: image_to_rgb565

.. autofunction:: rgb565_to_image

.. autofunction:: image_to_rgb888

.. autofunction:: rgb888_to_image

.. autofunction:: image_to_rgb

.. autofunction:: rgb_to_image

.. autofunction:: rgb888_to_rgb565

.. autofunction:: rgb565_to_rgb888

.. autofunction:: rgb_to_rgb888

.. autofunction:: rgb888_to_rgb

.. autofunction:: rgb_to_rgb565

.. autofunction:: rgb565_to_rgb


.. _colorzero: https://colorzero.readthedocs.io/
