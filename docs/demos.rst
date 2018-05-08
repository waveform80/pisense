============
Simple Demos
============


To get us warmed up before we attempt any real projects, here's some simple
demos that use the functionality of the Sense HAT.


Rainbow Scroller
================

There are many different color systems, and the `colorzero`_ library that
pisense relies upon implements several, including HSV (Hue, Saturation, Value).
In this scheme, hue is essentially cyclic. This makes it quite easy to produce
a scrolling rainbow display. We'll construct an 8x8 array in which the hue of a
color depends on the sum of its X and Y coordinates divided by 14 (as the
maximum sum is 7 + 7), which will give us a nice range of hues. You can try
this easily from the command line::

    >>> from pisense import SenseHAT, array
    >>> from colorzero import Color
    >>> hat = SenseHAT()
    >>> rainbow = array([
    ... Color(h=(x + y) / 14, s=1, v=1)
    ... for x in range(8)
    ... for y in range(8)
    ... ])
    >>> hat.screen.array = rainbow

At this point you should have a nice rainbow on your display. How do we make
this scroll? Simple: construct a loop that increments the hue a tiny amount
each time round. For example:

.. literalinclude:: examples/rainbow.py


Joystick Movement
=================

In this demo we'll move a dot around the screen in response to joystick moves.
The easiest way to interact with the joystick is to treat it as an iterator
(treating it as if it's a rather slow list that only provides another value
when something happens to the joystick). Most of the time you're not that
interested in the joystick events themselves, but rather on what they *mean* to
your application.

Hence our first step is to define a generator function that transforms joystick
events into relative X, Y movements. Then we'll define another simple generator
that transforms these into arrays for the display. Finally, we'll use that
output to drive the display:

.. literalinclude:: examples/joystick_basic.py

This pattern of programming, treating inputs as iterators and writing a series
of transforms to produce screen arrays, will become a common theme in much of
the rest of this manual.


Environment Sensing
===================

How about a simple thermometer? We'll treat the thermometer as an iterator, and
write a transform that produces a screen containing the temperature as both a
number (in a small font), and a very basic chart which lights more elements as
the temperature increases.

.. literalinclude:: examples/thermometer.py
