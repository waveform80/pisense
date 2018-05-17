===============
Getting started
===============

.. currentmodule:: pisense

.. warning::

    Make sure your Pi is off while installing the Sense HAT.


Hardware
========

Remove the sense HAT from its packaging. You should have the following parts:

.. TODO package pictures

1. The Sense HAT itself

2. A 40-pin stand-off header. This usually comes attached to the Sense HAT and
   many people don't realize it's removable (until they try and unplug their
   Sense HAT and it comes off!)

3. Eight screws and four stand-off posts.

To install the Sense HAT:

.. TODO installation pictures

1. Screw the stand-off posts onto the Pi from the bottom.

   .. warning::

        On the Pi 3B, some people have noticed reduced performance from using a
        stand-off post next to the wireless antenna (the top-left position if
        looking at the top of the Pi with the HDMI port at the bottom).  You
        may wish to leave this position empty or simply skip using the
        stand-offs entirely (they are optional but make the joystick a little
        easier to use).

2. Push the Sense HAT onto Pi's GPIO pins ensuring all the pins are aligned.
   The Sense HAT should cover most of the Pi (other than the USB / Ethernet
   ports).

3. If using the stand-offs, secure them to the Sense HAT from the top with the
   remaining screws. If you find you cannot align the holes on the Sense HAT
   with the stand-offs this is a sure-fire sign that the pins are misaligned
   (you've missed a row / column of GPIO pins when installing the HAT). In
   this case, remove the Sense HAT from the GPIO pins and try again.

4. Finally, apply power to the Pi. If everything is installed correctly (and
   you have a sufficiently up to date version of Raspbian on your SD card) you
   should see a rainbow appear on the Sense HAT's LEDs as soon as power is
   applied. The rainbow should disappear at some point during boot-up. If the
   rainbow does *not* disappear this either means the HAT is not installed
   correctly or your copy of Raspbian is not sufficiently up to date.


First Steps
===========

Start a Python environment (this documentation assumes you use Python 3, though
the pisense library is compatible with both Python 2 and 3), and import the
pisense library, then construct an object to interface to the HAT:

.. code-block:: console

    $ python3
    Python 3.5.3 (default, Jan 19 2017, 14:11:04)
    [GCC 6.3.0 20170124] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import pisense
    >>> hat = pisense.SenseHAT()

The ``hat`` object represents the Sense HAT, and provides several attributes
which represent the different components on the HAT. Specifically:

* ``hat.screen`` represents the 8 x 8 grid of LEDs on the HAT.

* ``hat.stick`` represents the miniature joystick at the bottom right of the
  HAT.

* ``hat.environ`` represents the environmental (pressure, humidity and
  temperature) sensors on the HAT.

* ``hat.imu`` represents the sensors of the Internal Measurement Unit (IMU)
  on the HAT.


The Screen
==========

Let's try controlling the screen first of all. The screen's state is
represented as a `numpy`_ :class:`~numpy.ndarray` of ``(red, green, blue)``
values.  The structure of the values is compatible with
:class:`~colorzero.Color` class from the `colorzero`_ library which makes them
quite easy to work with:

.. code-block:: pycon

    >>> from colorzero import Color
    >>> hat.screen.array[0, 0] = Color('red')

You should see the top-left LED on the HAT light up red.  It's worth noting at
this point that the two dimensions of the numpy array are rows, *then* columns
so the first coordinate is the Y coordinate, and that 0 on the Y-axis is at
the top. If this seems confusing (because graphs are typically drawn with the
origin at the *bottom* left) consider that (in English at least) you start
reading from the *top* left of a page which is why the origin of computer
displays is there.

As for why the "X" coordinate comes second, this is due to the way image data
is laid out in memory. "Bigger" dimensions (by which we mean slower moving
dimensions) come first, followed by "smaller" dimensions. When dealing with a
graphical display (or reading text in English), we move *along* the display
first before moving *down* a line. Hence the "X" coordinate is "smaller"; it
moves "faster" than the Y coordinate, changing with every step along the
display whereas the Y coordinate only changes when we reach the end of a line.

Hence, just as we put "bigger" values first when writing out numbers
(thousands, then hundreds, then tens, then units), or the time (hours, minutes,
seconds), we write the "bigger" coordinate (the Y coordinate) first when
addressing pixels in the display:

.. code-block:: pycon

    >>> hat.screen.array[0, 1] = Color('green')
    >>> hat.screen.array[1, 0] = Color('blue')

Numpy's arrays allow us to address more than one value at once, by "slicing"
the array. We won't cover all the details of Python `slicing`_ (see the linked
manual page for full details), but here's some examples of what we can do with
slicing (and what bits are optional). We can turn four pixels along the top red
in a single command:

.. code-block:: pycon

    >>> hat.screen.array[0, 0:4] = Color('red')

If the start of a slice is zero it can be omitted (if the end of a slice is
unspecified it is the length of whatever you're slicing). Hence we can change
the entire upper left quadrant red with a single command:

.. code-block:: pycon

    >>> hat.screen.array[:4, :4] = Color('red')

We can omit both the start and end of a slice (by specifying ":") to indicate
we want the entire length of whatever we're slicing. For example, to draw a
couple of white lines next to our quadrant:

.. code-block:: pycon

    >>> hat.screen.array[:, 4] = Color('white')
    >>> hat.screen.array[4, :] = Color('white')

We can also *read* the display as well as write to it. We can read individual
elements or slices, just as with writing:

.. code-block:: pycon

    >>> hat.screen.array[0, 0]
    (1., 0., 0.)
    >>> hat.screen.array[4, :]
    ScreenArray([(1., 1., 1.), (1., 1., 1.), (1., 1., 1.), (1., 1., 1.),
                 (1., 1., 1.), (1., 1., 1.), (1., 1., 1.), (1., 1., 1.)],
                dtype=[('r', '<f4'), ('g', '<f4'), ('b', '<f4')])

This means we can *scroll* our display by assigning a slice to another
(similarly shaped) slice. First we'll take a copy of our display so we can get
it back later, then we'll use a loop with a delay to slide our display left:

.. code-block:: pycon

    >>> original = hat.screen.array.copy()
    >>> from time import sleep
    >>> for i in range(8):
    ...     hat.screen.array[:, :7] = hat.screen.array[:, 1:]
    ...     sleep(0.1)
    ...

Neat as that was, the screen object actually has several methods to make
animations like this easy. Let's slide our original back onto the display:

.. code-block:: pycon

    >>> hat.screen.slide_to(original, direction='right')

We can construct images for the display with the :func:`array` function. Let's
construct a blue screen (thankfully not of death!) and fade to it:

.. code-block:: pycon

    >>> blue_screen = pisense.array(Color('blue'))
    >>> hat.screen.fade_to(blue_screen)

The :func:`array` function can also be given a list of values to initialize
itself. This is particularly useful with :class:`~colorzero.Color` aliases a
single letter long. For example, to draw a French flag on our display:

.. code-block:: pycon

    >>> B = Color('black')
    >>> r = Color('red')
    >>> w = Color('white')
    >>> b = Color('blue')
    >>> black_line = [B, B, B, B, B, B, B, B]
    >>> flag_line = [B, b, b, w, w, r, r, B]
    >>> flag = pisense.array(black_line * 2 + flag_line * 4 + black_line * 2)
    >>> hat.screen.fade_to(flag)

Finally, if you're familiar with the `Pillow`_ library (formerly PIL, the
Python Imaging Library) you can obtain a representation of the screen with the
:meth:`~SenseScreen.image` method. You can draw on this with the facilities of
Pillow's :mod:`~PIL.ImageDraw` module then copy the result back to the Sense
HAT's screen with the :meth:`~SenseScreen.draw` method (the image returned
doesn't automatically update the screen when modified, unlike the array
representation).


The Joystick
============

The miniature joystick at the bottom right of the Sense HAT is exceedingly
useful as a basic interface for Raspberry Pis without a keyboard. The joystick
actually emulates a keyboard (which in some circumstances is very annoying) but
it's simpler, and more useful, to use the library's facilities to read the
joystick rather than trying to read the keyboard. The :meth:`~SenseStick.read`
method can be used to wait for an event from the joystick. Type the following
then briefly tap the joystick to the right:

.. code-block:: pycon

    >>> hat.stick.read()
    StickEvent(timestamp=datetime.datetime(2018, 5, 4, 22, 52, 35, 961776),
    direction='right', pressed=True, held=False)

As you've released the joystick there should be a "release" event waiting to be
retrieved. Notice that its timestamp is shortly after the former event (because
the timestamp is the time at which the event *occurred*, not when it was
retrieved):

.. code-block:: pycon

    >>> hat.stick.read()
    StickEvent(timestamp=datetime.datetime(2018, 5, 4, 22, 52, 36, 47511),
    direction='right', pressed=False, held=False)

The :meth:`~SenseStick.read` method can also take a timeout value (measured in
seconds). If an event has not occurred before the timeout elapses, it will
return ``None``:

.. code-block:: pycon

    >>> print(repr(hat.stick.read(1.0)))
    None

The event is returned as a :func:`~collections.namedtuple` with the following
fields:

* ``timestamp`` -- the timestamp at which the event occurred.

* ``direction`` -- the direction in which the joystick was pushed. If the
  joystick is pushed inwards this will be "enter" (as that's the key that it
  emulates).

* ``pressed`` -- this will be ``True`` if the event occurred due to the
  joystick being pressed *or held* in a particular direction. If this is
  ``False``, the joystick has been released from the specified direction.

* ``held`` -- when ``True`` the meaning of this field depends on the
  ``pressed`` field:

  - When ``pressed`` is also ``True`` this indicates that the event is a repeat
    event occurring because the joystick is being held in the specified
    direction.

  - When ``pressed`` is ``False`` this indicates that the joystick has been
    released but it *was* held down (this is useful for distinguishing between
    a press and a hold during the release event).

Hence a typical sequence of events when briefly pressing the joystick right
would be:

========= ======= =====
direction pressed held
========= ======= =====
right     True    False
right     False   False
========= ======= =====

However, when holding the joystick right, the sequence would be:

========= ======= =====
direction pressed held
========= ======= =====
right     True    False
right     True    True
right     True    True
right     True    True
right     True    True
right     False   True
========= ======= =====

Finally, the joystick can be treated as an iterator which yields events
whenever they occur. This is particularly useful for driving interfaces as
we'll see in later sections. For now, you can try this on the command line:

.. code-block:: pycon

    >>> for event in hat.stick:
    ...     print(repr(event))
    ...
    StickEvent(timestamp=datetime.datetime(2018, 5, 4, 20, 6, 10, 845258), direction='right', pressed=True, held=False)
    StickEvent(timestamp=datetime.datetime(2018, 5, 4, 20, 6, 11, 100073), direction='right', pressed=True, held=True)
    StickEvent(timestamp=datetime.datetime(2018, 5, 4, 20, 6, 11, 150078), direction='right', pressed=True, held=True)
    StickEvent(timestamp=datetime.datetime(2018, 5, 4, 20, 6, 11, 200125), direction='right', pressed=True, held=True)
    StickEvent(timestamp=datetime.datetime(2018, 5, 4, 20, 6, 11, 250146), direction='right', pressed=True, held=True)
    StickEvent(timestamp=datetime.datetime(2018, 5, 4, 20, 6, 11, 300088), direction='right', pressed=True, held=True)
    StickEvent(timestamp=datetime.datetime(2018, 5, 4, 20, 6, 11, 316964), direction='right', pressed=False, held=True)
    ^C

.. note::

    You'll probably see several strange sequences appear on the terminal when
    playing with this (like ``^[[A``, ``^[[B``, etc). These are the raw control
    codes for the cursor keys and can be ignored. Press :kbd:`Control-c` when
    you want to terminate the loop.


Environmental Sensors
=====================

The environmental sensors on the Sense HAT consist of two components: a
pressure sensor and a humidity sensor. Both of these components are also
capable of measuring temperature. For the sake of simplicity, both sensors are
wrapped in a single item in pisense which can be queried for pressure,
humidity, or temperature:

.. code-block:: pycon

    >>> hat.environ.pressure
    1025.3486328125
    >>> hat.environ.humidityy
    51.75486755371094
    >>> hat.environ.temperature
    29.045833587646484

The pressure is returned in `millibars`_ (which are equivalent to
`hectopascals`_). The humidity is given as a `relative humidity`_ percentage.
Finally, the temperature is returned in `celsius`_.

Despite there being effectively two temperature sensors there's only a single
``temperature`` property. By default it returns the reading from the humidity
sensor, but you configure this with the
:attr:`~SenseEnvironment.temperature_source` attribute:

.. code-block:: pycon

    >>> hat.environ.temperature_source
    <function temp_humidity at 0x7515b588>
    >>> hat.environ.temperature_source = pisense.temp_pressure
    >>> hat.environ.temperature
    29.149999618530273
    >>> hat.environ.temperature_source = pisense.temp_humidity
    >>> hat.environ.temperature
    25.24289321899414

Note that both temperature readings can be quite different!  You can also
configure it to take the average of the two readings:

.. code-block:: pycon

    >>> hat.environ.temperature_source = pisense.temp_average
    27.206080436706543

However, if you think this will give you more accuracy, `Dilbert`_ may take
exception to your notion!

Like the joystick, the environment sensor(s) can also be treated as an
iterator:

.. code-block:: pycon

    >>> for reading in hat.environ:
    ...     print(repr(reading))
    ...
    EnvironReadings(pressure=1025.41, humidity=51.1534, temperature=27.1774)
    EnvironReadings(pressure=1025.41, humidity=50.9851, temperature=27.2261)
    EnvironReadings(pressure=1025.41, humidity=50.9851, temperature=27.2271)
    EnvironReadings(pressure=1025.42, humidity=50.9851, temperature=27.2240)
    EnvironReadings(pressure=1025.42, humidity=50.9209, temperature=27.2240)
    EnvironReadings(pressure=1025.42, humidity=50.9209, temperature=27.2230)
    EnvironReadings(pressure=1025.42, humidity=50.9209, temperature=27.2261)
    EnvironReadings(pressure=1025.42, humidity=50.9209, temperature=27.2271)
    EnvironReadings(pressure=1025.42, humidity=51.0693, temperature=27.2331)
    ^C

A simple experiment you can run is to breathe near the humidity sensor and then
query its value. You should see the value rise quite rapidly before it slowly
falls back down as the vapour you exhaled evaporates from the surface of the
sensor.


Inertial Measurement Unit (IMU)
===============================

The `Inertial Measurement Unit`_ (IMU) on the Sense HAT actually consists of
three different sensors (an `accelerometer`_, a `gyroscope`_, and a
`magnetometer`_) each of which provide three readings (X, Y, and Z). This is
why you may also hear the sensor referred to as a 9-DoF (9 Degrees of Freedom)
sensor; it returns 9 independent values.

You can read values from the sensors independently:

.. code-block:: pycon

    >>> hat.imu.accel
    IMUReadings(x=0.0351547, y=0.052686, z=1.015)
    >>> hat.imu.gyro
    IMUReadings(x=0.0180783, 0.000277532, z=-0.0179323)
    >>> hat.imu.compass
    IMUReadings(x=-47.4917, y=-22.4575, z=38.9144)

.. TODO update these

The accelerometer returns values in g (`standard gravities`_, equivalent to
9.80665m/s²). Hence, with the Sense HAT lying flat on a table, the X and Y
values of the accelerometer should be close to zero, while the Z values should
be close to 1 (because gravity is pulling it straight down).

The gyroscope returns values in `radians per second`_. With the Sense HAT lying
stationary all values should be close to zero. If you wish to test the
gyroscope, set the console to continually print values and slowly rotate the
HAT:

.. code-block:: pycon

    >>> while True:
    ...     print(hat.imu.gyro)
    ...     sleep(0.1)
    ...

.. TODO output

Finally, the magnetometer returns values in µT (`micro-Teslas`_, where 1µT is
equal to 10mG or `milli-Gauss`_). The Earth's magnetic field is incredibly
weak, so if you wish to test the magnetometer it is easier to do so with a
permanent magnet, especially something strong like a small `neodymium magnet`_.
Bringing such a magnet within 10cm of the HAT should provoke an obvious
reaction in the readings.

The readings from these three components are combined by the underlying library
to form a composite "orientation" reading which provides the `roll, pitch, and
yaw`_ of the HAT in `radians`_:

.. code-block:: pycon

    >>> hat.imu.orient
    IMUOrient(roll=-0.461751, pitch=1.10459, yaw=-0.508346)

.. TODO correct this

Like all the other sensors on the HAT, the IMU can be treated as an iterator:

    >>> for state in hat.imu:
    ...     print(repr(state))
    ...

.. TODO output

This concludes the tour of the Raspberry Pi Sense HAT, and of the bare
functionality of the pisense library. The next sections will introduce some
simple projects to give you an idea of how the library can be used to combine
these facilities to useful or fun effect!


.. _numpy: https://numpy.org/
.. _Pillow: https://pillow.readthedocs.io/
.. _colorzero: https://colorzero.readthedocs.io/
.. _slicing: https://docs.scipy.org/doc/numpy/user/quickstart.html#indexing-slicing-and-iterating
.. _millibars: https://en.wikipedia.org/wiki/Bar_(unit)
.. _hectopascals: https://en.wikipedia.org/wiki/Pascal_(unit)
.. _relative humidity: https://en.wikipedia.org/wiki/Relative_humidity
.. _celsius: https://en.wikipedia.org/wiki/Celsius
.. _Inertial Measurement Unit: https://en.wikipedia.org/wiki/Inertial_measurement_unit
.. _accelerometer: https://en.wikipedia.org/wiki/Accelerometer
.. _gyroscope: https://en.wikipedia.org/wiki/Gyroscope
.. _magnetometer: https://en.wikipedia.org/wiki/Magnetometer
.. _standard gravities: https://en.wikipedia.org/wiki/Standard_gravity
.. _radians per second: https://en.wikipedia.org/wiki/Radian_per_second
.. _micro-Teslas: https://en.wikipedia.org/wiki/Tesla_(unit)
.. _milli-Gauss: https://en.wikipedia.org/wiki/Gauss_(unit)
.. _neodymium magnet: https://en.wikipedia.org/wiki/Neodymium_magnet
.. _roll, pitch, and yaw: https://en.wikipedia.org/wiki/Aircraft_principal_axes
.. _radians: https://en.wikipedia.org/wiki/Radian
.. _Dilbert: http://dilbert.com/strip/2008-05-07
