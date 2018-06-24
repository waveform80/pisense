============
Simple Demos
============

.. currentmodule:: pisense

To get us warmed up before we attempt some complete applications, here's some
simple demos that use the functionality of the Sense HAT. Along with some demos
there's a small exercise, which you might like to try if you want to hone your
skills with the library.


Rainbow Scroller
================

There are many different color systems, and the `colorzero`_ library that
pisense relies upon implements several, including HSV (Hue, Saturation, Value).
In this scheme, hue is essentially cyclic. This makes it quite easy to produce
a scrolling rainbow display. We'll construct an 8x8 array in which the hue of a
color depends on the sum of its X and Y coordinates divided by 14 (as the
maximum sum is 7 + 7), which will give us a nice range of hues. You can try
this easily from the command line:

.. code-block:: pycon

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
this scroll? We simply construct a loop that increments the hue a tiny amount
each time round. For example:

.. literalinclude:: examples/rainbow.py
    :caption:


.. _joystick_movement:

Joystick Movement
=================

In this demo we'll move a dot around the screen in response to joystick moves.
The easiest way to interact with the joystick is to treat it as an iterator
(treating it as if it's a rather slow list that only provides another value
when something happens to the joystick). Most of the time you're not that
interested in the joystick events themselves, but rather on what they *mean* to
your application.

Hence our first step is to define a generator function that transforms joystick
events into relative X, Y movements:

.. literalinclude:: examples/joystick_dot.py
    :pyobject: movements

You can try this out from the command line like so::

    >>> hat = SenseHAT()
    >>> for x, y in movements(hat.stick):
    ...     print('x:', x, 'y:', y)
    ...
    x: 1 y: 0
    x: 1 y: 0
    x: 0 y: 1
    x: 0 y: 1
    x: -1 y: 0

.. note::

    You may see several control characters like ``^[[C`` and ``^[[D`` appearing
    as you play with this. These are the raw characters that represent the
    cursor keys; this output can be ignored. Press the joystick in (generate
    an "enter" event) when you want to terminate the loop.

Now, we'll define another simple generator that transforms these into arrays
for the display. Finally, we'll use that output to drive the display:

.. literalinclude:: examples/joystick_dot.py
    :caption:

This pattern of programming, treating inputs as iterators and writing a series
of transforms to produce screen arrays, will become a common theme in much of
the rest of this manual.

.. admonition:: Exercise

    Can you convert the rainbow demo above to use an iterable for its display?
    Hint: the iterable doesn't need to take any input because it's not really
    transforming anything, just yielding outputs.


Orientation Sensing
===================

Could we adapt the joystick example to "roll" the dot around the screen using
the Inertial Measurement Unit (IMU)? Quite easily as it happens. The only thing
that needs to change is the transformation that yields the changes in the X and
Y positions. Instead of transforming joystick events, it needs to transform IMU
readings.

As it happens, the IMU's accelerometer is perfect for this task. When the HAT
is tilted to the right, the X-axis of the accelerometer winds up pointing
downward, which means it starts reading close to 1 (due to gravity). The same
happens for the Y-axis when the HAT is tilted toward you. So, the
transformation is quite trivial:

1. Grab the accelerometer's X and Y axes

2. Clamp the values to the range -1 to 1 (we don't want things moving too
   fast!)

3. Round the values to the nearest integer (so we stay still until the HAT is
   tilted quite a lot)

4. Don't bother yielding a movement unless one value is non-zero

5. Introduce a short delay (with :func:`~time.sleep`) because the IMU is
   capable of spitting out readings hundreds of times a second, and we don't
   want the dot shooting around *that* fast!

Here's the modified ``movements`` function:

.. literalinclude:: examples/imu_dot.py
    :pyobject: movements

Again, you can try this function out from the command line in the same manner
as the joystick; just pass the IMU component to it instead::

    >>> from pisense import SenseHAT
    >>> hat = SenseHAT()
    >>> for x, y in movements(hat.imu):
    ...     print('x:', x, 'y:', y)
    ...
    x: 1 y: 0
    x: 1 y: 0
    x: 0 y: 1
    x: 0 y: 1
    x: -1 y: 0

Here's the whole thing put together. Note that the only substantial change from
the joystick demo above is the ``movements`` function:

.. literalinclude:: examples/imu_dot.py
    :caption:

.. admonition:: Exercise

    Can you combine the orientation demo with the rainbow scroller and make the
    rainbow scroll in different directions based on the orientation of the
    board?


.. _thermometer:

Environment Sensing
===================

How about a simple thermometer? We'll treat the thermometer as an iterator, and
write a transform that produces a screen containing the temperature as both a
number (in a small font), and a very basic chart which lights more elements as
the temperature increases.

We'll start with a function that takes a *reading*, limits it to the range of
temperatures we're interested in (0°C to 50°C), and distributes that over the
range 0 <= n < 64 (representing all 64 elements of the HAT's display):

.. literalinclude:: examples/thermometer.py
    :lines: 1-7

Next, we need to construct the crude chart representing the temperature. For
this we call :func:`array` and pass it a list of 64 :class:`~colorzero.Color`
objects which will be solid red if the element is definitely below the current
temperature, a scaled red for the element at the current temperature, and black
(off) if the element is above the current temperature. We also flip the result
as we want the chart to start at the bottom and work its way up:

.. literalinclude:: examples/thermometer.py
    :lines: 8-14

Next, we call :func:`draw_text` which will return us a small
:class:`~PIL.Image.Image` object containing the rendered text (we've added some
padding at the bottom so the text is "top aligned"). We'll convert that to an
array, and "add" that to the chart we've drawn (a simple method of overlaying)
and then clip the result to the range 0 to 1 (because where the text overlays
the chart we'll probably exceed the bounds of the red channel):

.. literalinclude:: examples/thermometer.py
    :lines: 15-19

Finally, here's the whole thing put together:

.. literalinclude:: examples/thermometer.py
    :caption:

You can test this script by running it, then placing your finger on the
humidity sensor (which is the sensor we're using to read temperature). If the
ambient temperature is below about 24°C you should see the reading rise quite
quickly. Take your finger off the sensor and it should fall back down again.

Why, in this example, did we construct a function that took a single reading?
Why did we not pass the `environ` iterator to the `thermometer` function? Quite
simply because we didn't have to: making an array for the screen works from a
single reading. It doesn't have any need to know prior readings, or to keep
any state between frames, so it's simplest to make it a straight-forward
function. That said…

.. admonition:: Exercise

    Can you change the script to show whether the temperature is rising or
    falling? Hint: passing the iterator to the transform is one way to do this,
    but for a neater way (without passing the iterator), look up ``pairwise``
    in :mod:`itertools`.

.. _colorzero: https://colorzero.readthedocs.io/
