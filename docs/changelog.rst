==========
Change log
==========

.. currentmodule:: pisense


Release 0.2 (2018-12-22)
========================

Reasonably happy with the API now, so this will probably be the final 0.x
release before 1.0. Nonetheless, a few bugs fixed:

* Setting a non-zero rotation on the joystick failed on the next event that
  occurred (`#1`_)
* Multiline text rendering was broken (`#2`_)

.. _#1: https://github.com/waveform80/pisense/issues/1
.. _#2: https://github.com/waveform80/pisense/issues/2


Release 0.1 (2018-07-19)
========================

Initial release. Please note that as this is a pre-v1 release, API backwards
compatibility is not *yet* guaranteed. I'm *mostly* happy with the API but for
some subtle aspects of the :class:`ScreenArray` class. Hence if anything's
going to change it's probably going to be there. Feedback welcome!
