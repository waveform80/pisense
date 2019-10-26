============
API - Screen
============

.. module:: pisense.screen

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

.. autofunction:: wipe_to

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
