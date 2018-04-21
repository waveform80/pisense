.. -*- rst -*-

=======
pisense
=======

This package is an alternative interface to the Raspberry Pi Sense HAT. It is
still somwwhat experimental and as such you should not rely upon it for
production systems; it's largely a play ground for my interface ideas!

That said, if you want to propose enhancements or bug fixes, please feel free.

The major difference to the official API is that the various components of the
Sense HAT (the screen, the joystick, the environment sensors, etc.) are
each represented by separate classes which can be used individually or by the
main class which composes them together.

The screen has a few more tricks including multiple (very basic) fonts, and
a representation as a numpy array (which makes scrolling by assigning slices of
a larger image very simple). The joystick, and all sensors, have an iterable
interface too.

Links
=====

* The code is licensed under the `BSD license`_
* The `source code`_ can be obtained from GitHub, which also hosts the `bug
  tracker`_
* The `documentation`_ (which includes installation, quick-start examples, and
  lots of code recipes) can be read on ReadTheDocs
* Packages can be downloaded from `PyPI`_, but reading the installation
  instructions is more likely to be useful

.. _PyPI: http://pypi.org/pypi/pisense/
.. _documentation: http://pisense.readthedocs.io/
.. _source code: https://github.com/waveform80/pisense
.. _bug tracker: https://github.com/waveform80/pisense/issues
.. _BSD license: http://opensource.org/licenses/BSD-3-Clause
