============
API - Screen
============

.. currentmodule:: pisense

.. autoclass:: SenseScreen

.. autoclass:: ScreenArray

.. autofunction:: array


Animations
==========

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

Easing
======

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

.. data:: DEFAULT_GAMMA

    The default gamma table, which can be assigned directly to
    :attr:`~SenseScreen.gamma`. The default rises in a steady curve from 0
    (off) to 31 (full brightness).

.. data:: LOW_GAMMA

    The "low light" gamma table, which can be assigned directly to
    :attr:`~SenseScreen.gamma`. The low light table rises in a steady curve
    from 0 (off) to 10.


Conversions
===========

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
