==============
API - Joystick
==============

.. module:: pisense.stick

.. currentmodule:: pisense

The joystick on the Sense HAT is an excellent tool for providing a user
interface on Pis without an attached keyboard. The :class:`SenseStick` class
provides several different paradigms for programming such an interface:

* At its simplest, you can poll the state of the joystick with various
  attributes like :attr:`SenseStick.up`.

* You can use event-driven programming by assigning handlers to attributes
  like :attr:`SenseStick.when_up`.

* You can also treat the joystick like an iterable and write transformations
  the convert events into other useful outputs.


SenseStick
==========

.. autoclass:: SenseStick


StickEvent
==========

.. autoclass:: StickEvent(timestamp, direction, pressed, held)


Warnings
========

.. autoexception:: SenseStickBufferFull

.. autoexception:: SenseStickCallbackRead
