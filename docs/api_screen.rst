============
API - Screen
============

.. currentmodule:: pisense

The screen interface is by far the most extensive and complex part of the
pisense library, comprising several classes and numerous functions to handle
representing the screen in a variety of conveniently manipulated formats, and
generation of slick animations. The two most important elements are the main
:class:`SenseScreen` class itself, and the :class:`ScreenArray` class which is
used to represent the contents of the display.


SenseScreen
===========

.. autoclass:: SenseScreen


ScreenArray
===========

.. For some reason numpy descendants blank their __doc__ strings so this chunk
.. has to be included manually...

.. class:: ScreenArray

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
    library's :ref:`format strings <colorzero.format>`. For example::

    .. attention:: TODO finalize format string spec

    .. _colorzero: https://colorzero.readthedocs.io/

    .. method:: preview(element='██', colors=None, width=Nnone)

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

.. autofunction:: array


Animation functions
===================

The following animation generator functions are used internally by the
animation methods of :class:`SenseScreen`. They are also provided as separate
generator functions to permit users to build up complex sequences of
animations, or to aid in generating other effects like interspersing frames
with other sequences.

Each function is a generator function which yields an :class:`~PIL.Image.Image`
for each frame of the animation.

.. autofunction:: scroll_text

.. autofunction:: fade_to

.. autofunction:: slide_to

.. autofunction:: zoom_to

.. autofunction:: draw_text


.. _easing:

Easing functions
================

The easing functions are used with the animation functions above for their
*easing* parameters.

An easing function must take a single integer parameter indicating the number
of frames in the resulting animation. It must return a sequence of (or
generator which yields) floating point values between 0.0 (which indicates the
start of the animation) and 1.0 (which indicates the end of the animation). How
fast the value moves from 0.0 to 1.0 dictates how fast the animation progresses
from frame to frame.

Several typical easing functions are provided by the library, but you are free
to use any function which complies which this interface. The default easing
function is always linear:

.. autofunction:: linear

.. autofunction:: ease_in

.. autofunction:: ease_out

.. autofunction:: ease_in_out


Gamma tables
============

Two built-in gamma tables are provided which can be assigned to
:attr:`SenseScreen.gamma`. However, you are free to use any compatible list of
32 values.


.. TODO graphs

.. data:: DEFAULT_GAMMA

    The default gamma table, which can be assigned directly to
    :attr:`~SenseScreen.gamma`. The default rises in a steady curve from 0
    (off) to 31 (full brightness).

.. data:: LOW_GAMMA

    The "low light" gamma table, which can be assigned directly to
    :attr:`~SenseScreen.gamma`. The low light table rises in a steady curve
    from 0 (off) to 10.


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
