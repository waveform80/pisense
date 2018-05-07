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
