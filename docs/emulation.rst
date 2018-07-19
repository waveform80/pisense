==================
Sense HAT Emulator
==================

.. currentmodule:: pisense

The pisense library is compatible with the desktop `Sense HAT emulator`_,
however it uses a slightly different method of specifying that the emulator
should be used instead of the "real" HAT. You can construct the
:class:`SenseHAT` class passing ``True`` as the value of the *emulate*
parameter::

    from pisense import SenseHAT

    hat = SenseHAT(emulate=True)

However, the default value of *emulate* is taken from an environment variable:
``PISENSE_EMULATE``. This means an even easier way (which doesn't require
modifying your script at all) is to simply run your script after setting that
variable. For example:

.. code-block:: console

    $ python my_script.py  # run on the "real" HAT
    $ PISENSE_EMULATE=1 python my_script.py  # run on the emulator

If you are going to be working with the emulator primarily (e.g. if you're
not working on a Pi), you may wish to add the following line to your
:file:`~/.bashrc` script so that all scripts default to using the emulator:

.. code-block:: bash

    export PISENSE_EMULATE=1

If the emulator is not detected when :class:`SenseHAT` is constructed, and
*emulate* is either ``True`` or defaults to ``True`` because of the environment
variable, the emulator will be launched.

.. note::

    The emulator referred to here is the desktop Sense HAT emulator, *not*
    the excellent `online emulator`_ developed by Trinket. Unfortunately as
    pisense relies on both numpy and PIL, it's unlikely pisense can be
    easily ported to this.

.. _Sense HAT emulator: https://sense-emu.readthedocs.io/
.. _online emulator: https://trinket.io/sense-hat
