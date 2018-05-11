===================
Environment Monitor
===================

Here's a basic first project for the Sense HAT: make an environmental monitor
that can display the temperature, humidity, and pressure in a variety of forms.
We've already seen a demo thermometer in :ref:`thermometer`. First we'll
construct variants of this for the humidity and pressure sensors. Then we'll
combine all three into an interactive application using the joystick to select
the required functionality, and outputting data to the screen, and to an SQLite
database.

Hygrometer
==========

Firstly, let's adapt our thermometer script for sensing humidity. Here's the
thermometer script again:

.. literalinclude:: examples/thermometer.py
    :caption:

We'll use a very similar structure for our hygrometer. This time we don't need
to clamp the range (we'll use the full 0% to 100%, but we'll scale it to 0 <= n
< 64 again). We'll use a reasonably dark blue ("#000088" in HTML terms) but
everything else should look fairly familiar:

.. literalinclude:: examples/hygrometer.py
    :caption:

The one other subtle adjustment is in the caption. We can't fit "100" on our
display; it's too wide (this wasn't a problem for the thermometer where we
clamped the temperature range from 0° to 50°; you'd be right if you guessed
this was for simplicity!). Instead, whenever the humidity is >99% we display
"^^" to indicate the maximum value.

Test this script out by running it and then breathing gently on the humidity
sensor. You should see the humidity reading rise rapidly (possibly to the "^^"
value) then slowly fall back down.


Barometer
=========

Next we'll tackle the pressure sensor. This will have a very familiar structure
by now:

1. Clamp the pressure readings to a sensible range (in this case we'll use
   950mbar to 1050mbar).

2. Scale this to the range 0 <= n < 64.

3. Draw a rudimentary chart (this time we'll use green to distinguish it from
   our thermometer and hygrometer scripts).

4. Draw the pressure as a number superimposed on the chart.

Oh dear. That's a problem! All the valid pressure values are too large to fit
on the display, so we can't use an easy hack like displaying "^^" as we did in
the hygrometer above.

It'd be nice if the pressure reading could scroll back and forth on the
display, still superimposed on the chart. It turns out, using iterators again,
this is actually quite easy to achieve. What we want is a sliding window over
our rendered text, like so:

.. image:: images/sliding_window.*
    :align: center

Hence our first requirement is an infinite iterator which produces the
"bouncing" X offset for the sliding window:

.. literalinclude:: examples/barometer.py
    :pyobject: bounce

Well, that was simple!

The :func:`~itertools.cycle` and :func:`~itertools.chain` functions come from
the standard library's fantastic :mod:`itertools` module which I urge anyone
using iterators to check out. The :func:`reversed` function is a standard
Python built-in function.

How do we combine the offsets produced by ``bounce`` with the readings from the
sensor? We simply use the built-in :func:`zip` function:

.. literalinclude:: examples/barometer.py
    :caption:

.. note::

    This example will *only* work in Python 3 because it evaluates :func:`zip`
    lazily. In Python 2, this will crash as zip attempts to construct a list
    for an infinite iterator (use ``izip`` from :mod:`itertools` in Python 2).
