=====================================
API - Inertial Measurement Unit (IMU)
=====================================

.. module:: pisense.imu

.. currentmodule:: pisense

The `Inertial Measurement Unit`_ (IMU) on the Sense HAT has myriad uses in all
sorts of projects from `High Altitude Balloon`_ (HAB) flights, robotics,
detecting magnetic fields, or making novel user interfaces. It is represented
in pisense by the :class:`SenseIMU` class, and provides readings as
:class:`IMUState`, :class:`IMUVector` and :class:`IMUOrient` tuples.


SenseIMU
========

.. autoclass:: SenseIMU


IMUState
========

.. autoclass:: IMUState(compass, gyro, accel, orient)


IMUVector
=========

.. autoclass:: IMUVector(x, y, z)


IMUOrient
=========

.. autoclass:: IMUOrient(roll, pitch, yaw)


SenseSettings
=============

.. autoclass:: SenseSettings


.. _Inertial Measurement Unit: https://en.wikipedia.org/wiki/Inertial_measurement_unit
.. _High Altitude Balloon: https://en.wikipedia.org/wiki/High-altitude_balloon
