================================
Frequently Asked Questions (FAQ)
================================

Feel free to `ask the author`_, or add questions to the `issue tracker`_ on
GitHub, or even edit this document yourself and add frequently asked questions
you've seen on other forums!

.. _ask the author: mailto:dave@waveform.org.uk
.. _issue tracker: https://github.com/waveform80/pisense/issues


.. currentmodule:: pisense

Why?
====

To be rather blunt, I'm not a fan of the Sense HAT's `official API`_. This
probably sounds a bit strange coming from someone who played a small part in
making it (I wrote the joystick handling side of it, and later the desktop
Sense HAT emulator)!  Originally pisense was my attempt, back when the Sense
HAT was relatively new, to design an API the way *I* wanted. It was a rough
experiment and I didn't want to "pollute" the space by offering a competing API
to the official one, so I left it as just that: an experiment available from my
GitHub pages, but not properly documented, tested, or packaged.

Over the years, I've wanted to actually use the Sense HAT in a few applications
and each time I've tried, I've found myself frustrated by the inconsistencies
or short-comings in the official API. Eventually that came to a head and I
decided to pull pisense out of storage and polish it up for serious use (I
considered including it statically in applications I built, but that seemed
ugly).

To be clear: this is *not* an attempt to supplant the official API. If you're a
teacher in education you're almost certainly better off with the official API.
All the learning resources are built for it, the community support is there for
it, and it's the only API accepted for the fabulous `Astro Pi`_ mission. Stop
reading this and go learn that one.

.. _official API: https://pythonhosted.org/sense-hat
.. _Astro Pi: https://astro-pi.org/


You still haven't answered why…
===============================

All the teachers gone? Okay. I don't *want* to put you off using the `official
API`_, but here's what I *don't* like about it:

* It pulls in numpy as a dependency. So does pisense, but we actually *use* it
  for more than rotating the display (seriously, that's all the official API
  uses it for). Why pull in numpy (a huge dependency) and then not use its
  signature class (an n-dimensional array) for your *two dimensional* display?

* It pulls in PIL as a dependency. Again, so does pisense, but we use it for a
  little more than a single method which just loads images for display. How
  about presenting the display as a PIL image for manipulation? Or using the
  drawing and scaling capabilities for animation? Font support for text
  display? Oh, and our image conversions don't rely on nested lists …

* Fixed width fonts for scrolling text? Urgh.

* The stick interface (yes, the one I wrote …) isn't *bad*, but it's not
  *great*. The real stroke of genius in pisense (which sadly I can't take
  credit for: yet again, it was one of Ben Nuttall's fabulous notions) was
  separating ``held`` into its own value in the :class:`StickEvent` tuple so
  that ``released`` events can tell if the button was *previously* held.

* Everything is conflated into a single class (except the joystick) so if you
  don't want certain functionality: tough, you still have to deal with all the
  initialization and memory usage for it (okay, that's just a nitpick really).

* Tons of duplicated ways of doing things. I want the temperature; do I call
  the ``get_temperature()`` method, or the ``get_temperature_from_humidity()``
  method, or query the ``temperature`` property, or the ``temp`` property?
  Actually it doesn't matter; they all do the same thing (call
  ``get_temperature_from_humidity()``).

* Several limitations in the API. I want both the raw accelerometer readings
  (in g, because degrees really are useless for that) *and* the magnetometer
  readings. The only way to do this is to query ``accel_raw`` and
  ``compass_raw`` (or call their duplicated methods). However, under the covers
  this causes two *separate* IMU reads with all the attendant overhead that
  implies. There's no way to get this set of data from a single IMU read.

I'm not intending this to be the simplest interface to the Sense HAT. The
official API is probably easier to get going with. My feeling is that I'd
prefer an API that was a little harder to get started with if it allowed me
more scope to "get things done".


Why are you using *single* precision floats in the display?!
============================================================

Under the covers, the Sense HAT's display framebuffer stores pixel information
in RGB565 format. That's 5-bits for red and blue, and 6-bits for green. The
32-bit `single-precision floating point`_ format used in pisense still uses
23-bits for the mantissa; more than enough to represent the 5 or 6-bits of data
for each pixel.

Why not use RGB565 directly? We do: the :attr:`SenseScreen.raw` attribute
provides an array backed by the actual framebuffer in RGB565 format, if you
really want the fastest, lowest level access.

However, for ease of use I wanted the array format to be compatible with my
`colorzero`_ library, which meant using a floating point format. The smaller
the format, the more efficient the library as there's less data to chuck around
and crunch (ideally I wanted it to perform reasonably on the smallest Pi
platforms like the old A+). During development, this library used the rather
obscure half-precision floating point format which is only 16-bits in size
(and provides 11-bits for the mantissa). However, hardware support for this
floating point format is only present on some Pi models and as best as I can
tell isn't supported at all in Raspbian's 32-bit userland. In tests, the single
precision format turned out to be the fastest so that's what the library uses.

.. _single-precision floating point: https://en.wikipedia.org/wiki/Single-precision_floating-point_format
.. _colorzero: https://colorzero.readthedocs.io/
.. _numpy: https://numpy.org/


Why are orientation and gyroscopic values in radians, not degrees?
==================================================================

Firstly, there's routines built into Python's standard library for conversion
so this is trivial to achieve without the library duplicating it. However, the
more important reason is not to clutter the API with unnecessary attributes.

Degrees are probably simpler to look at as pure values, but they're
considerably less useful to *use* in practice. This is because almost every
routine you are likely to use these values with (all trigonometric routines for
instance), only accept radians. This is why the :func:`repr` of the orientation
includes degree values (because they're useful values to "eyeball") but the
actual class doesn't include such values.

If it did, I'd likely name them things like ``roll_degrees`` at which point
you're typing almost as much as ``degrees(roll)`` anyway!
