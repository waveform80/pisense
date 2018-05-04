.. -*- rst -*-

=======
pisense
=======

This package is an alternative interface to the Raspberry Pi Sense HAT.  The
major difference to the `official API`_ is that the various components of the
Sense HAT (the screen, the joystick, the environment sensors, etc.) are each
represented by separate classes which can be used individually or by the main
class which composes them together.

The screen has a few more tricks including support for any fonts that `PIL`_
supports, representation as a `numpy`_ array (which makes scrolling by
assigning slices of a larger image very simple), and bunch of rudimentary
animation functions. The joystick, and all sensors, have an iterable interface
too.

Links
=====

* The code is licensed under the `BSD license`_
* The `source code`_ can be obtained from GitHub, which also hosts the `bug
  tracker`_
* The `documentation`_ (which includes installation, quick-start examples, and
  lots of code recipes) can be read on ReadTheDocs
* Packages can be downloaded from `PyPI`_, but reading the installation
  instructions is more likely to be useful

.. _official API: https://pythonhosted.org/sense-hat
.. _PIL: https://pillow.readthedocs.io/
.. _numpy: https://numpy.org/
.. _PyPI: http://pypi.org/pypi/pisense/
.. _documentation: http://pisense.readthedocs.io/
.. _source code: https://github.com/waveform80/pisense
.. _bug tracker: https://github.com/waveform80/pisense/issues
.. _BSD license: http://opensource.org/licenses/BSD-3-Clause
