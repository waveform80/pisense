=========================
API - Environment Sensors
=========================

.. module:: pisense.environ

.. currentmodule:: pisense

The Sense HAT has two environment sensors: a humidity sensor and a pressure
sensor, which are exposed in the combined :class:`SenseEnviron` class. This
provides readings as :class:`EnvironReadings` tuples.


SenseEnviron
============

.. autoclass:: SenseEnviron


EnvironReadings
===============

.. autoclass:: EnvironReadings(pressure, humidity, temperature)


Temperature Configuration
=========================

.. autofunction:: temp_pressure

.. autofunction:: temp_humidity

.. autofunction:: temp_average

.. autofunction:: temp_both
