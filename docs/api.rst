===================
API - The Sense HAT
===================

.. currentmodule:: pisense

.. module:: pisense

The :mod:`pisense` module is the main namespace for the pisense package; it
imports (and exposes) all publically accessible classes, functions, and
constants from all the modules beneath it for convenience. It also defines
the top-level :class:`SenseHAT` class.


SenseHAT
========

.. autoclass:: SenseHAT(settings='/etc/RTIMULib.ini', \*, fps=15, easing=<function linear>, max_events=100, flush_input=True)


Warnings
========

.. autoexception:: SenseHATReinit
