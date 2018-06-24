===============
Getting started
===============

.. currentmodule:: pisense

.. warning::

    Make sure your Pi is off while installing the Sense HAT.


Hardware
========

Remove the sense HAT from its packaging. You should have the following parts:

.. attention:: TODO package pictures

1. The Sense HAT itself

2. A 40-pin stand-off header. This usually comes attached to the Sense HAT and
   many people don't realize it's removable (until they try and unplug their
   Sense HAT and it comes off!)

3. Eight screws and four stand-off posts.

To install the Sense HAT:

.. attention:: TODO installation pictures

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

.. code-block:: pycon

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

.. image:: images/highlight_screen.*
    :align: center

Let's try controlling the screen first of all. The screen's state is
represented as a two-dimensional :class:`~numpy.ndarray` of ``(red, green,
blue)`` values.  The structure of the values is compatible with
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
representation):

.. code-block:: pycon

    >>> flag_img = hat.screen.image()
    >>> from PIL import Image, ImageFilter
    >>> blur_img = flag_img.filter(ImageFilter.GaussianBlur(1))
    >>> hat.screen.draw(blur_img)


The Joystick
============

.. image:: images/highlight_stick.*
    :align: center

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
    codes for the cursor keys and can be ignored. Press :kbd:`Ctrl-c` when you
    want to terminate the loop.


Environmental Sensors
=====================

.. image:: images/highlight_environ.*
    :align: center

The environmental sensors on the Sense HAT consist of two components: a
pressure sensor and a humidity sensor. Both of these components are also
capable of measuring temperature. For the sake of simplicity, both sensors are
wrapped in a single item in pisense which can be queried for pressure,
humidity, or temperature:

.. code-block:: pycon

    >>> hat.environ.pressure
    1025.3486328125
    >>> hat.environ.humidity
    51.75486755371094
    >>> hat.environ.temperature
    29.045833587646484

The pressure is returned in `millibars`_ (which are equivalent to
`hectopascals`_). The humidity is given as a `relative humidity`_ percentage.
Finally, the temperature is returned in `celsius`_.

Despite there being effectively two temperature sensors there's only a single
``temperature`` property. By default it returns the reading from the humidity
sensor, but you change this with the :attr:`~SenseEnviron.temp_source`
attribute:

.. code-block:: pycon

    >>> hat.environ.temp_source
    <function temp_humidity at 0x7515b588>
    >>> hat.environ.temp_source = pisense.temp_pressure
    >>> hat.environ.temperature
    29.149999618530273
    >>> hat.environ.temp_source = pisense.temp_humidity
    >>> hat.environ.temperature
    25.24289321899414

Note that both temperature readings can be quite different!  You can also
configure it to take the average of the two readings:

.. code-block:: pycon

    >>> hat.environ.temperature_source = pisense.temp_average
    27.206080436706543

However, if you think this will give you more accuracy, `Dilbert`_ may have a
pithy word or two!

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

.. note::

    As above, press :kbd:`Ctrl-c` when you want to terminate the loop.

A simple experiment you can run is to breathe near the humidity sensor and then
query its value. You should see the value rise quite rapidly before it slowly
falls back down as the vapour you exhaled evaporates from the surface of the
sensor.


Inertial Measurement Unit (IMU)
===============================

.. image:: images/highlight_imu.*
    :align: center

The `Inertial Measurement Unit`_ (IMU) on the Sense HAT actually consists of
three different sensors (an `accelerometer`_, a `gyroscope`_, and a
`magnetometer`_) each of which provide three readings (X, Y, and Z). This is
why you may also hear the sensor referred to as a 9-DoF (9 Degrees of Freedom)
sensor; it returns 9 independent values.

You can read values from the sensors independently:

.. code-block:: pycon

    >>> hat.imu.accel
    IMUVector(x=0.0404885, y=0.0551139, z=1.01719)
    >>> hat.imu.gyro
    IMUVector(x=0.044841, y=0.00200727, z=-0.0528594)
    >>> hat.imu.compass
    IMUVector(x=-21.1644, y=-12.2358, z=18.4494)

The accelerometer returns values in g (`standard gravities`_, equivalent to
9.80665m/s²). Hence, with the Sense HAT lying flat on a table, the X and Y
values of the accelerometer should be close to zero, while the Z value should
be close to 1 (because gravity is a constant acceleration force toward the
center of the Earth … assuming that you're on Earth, that is).

The gyroscope returns values in `radians per second`_. With the Sense HAT lying
stationary all values should be close to zero. If you wish to test the
gyroscope, set the console to continually print values and slowly rotate the
HAT:

.. code-block:: pycon

    >>> while True:
    ...     print(hat.imu.gyro)
    ...     sleep(0.1)
    ...
    IMUVector(x=0.0437177, y=0.00241541, z=-0.0463548)
    IMUVector(x=0.0408809, y=0.00207451, z=-0.0443745)
    IMUVector(x=0.0428965, y=0.00294054, z=-0.0448299)
    IMUVector(x=0.0376711, y=0.00259082, z=-0.0440765)
    IMUVector(x=0.0376385, y=0.00705177, z=-0.0457381)
    IMUVector(x=0.0276967, y=-0.00117483, z=-0.0446691)
    IMUVector(x=-0.206876, y=-0.0201117, z=-0.128358)
    IMUVector(x=-0.0773721, y=-0.523465, z=-0.318948)
    IMUVector(x=-0.429841, y=-0.663047, z=0.0814746)
    IMUVector(x=0.288231, y=-1.13005, z=-0.0245105)
    IMUVector(x=-0.450611, y=-1.86431, z=-0.382783)
    IMUVector(x=-0.173889, y=-1.05461, z=-0.238619)
    IMUVector(x=-0.225202, y=-2.61934, z=-0.0840699)
    IMUVector(x=-0.00529005, y=-1.86309, z=-0.000686785)
    IMUVector(x=-0.00254116, y=-1.85271, z=0.115072)
    IMUVector(x=-0.0382768, y=-0.26965, z=-0.374536)

.. note::

    As above, press :kbd:`Ctrl-c` when you want to terminate the loop.

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
    IMUOrient(roll=0.868906 (49.8°), pitch=1.2295 (70.4°), yaw=0.818843 (46.9°))

Note that while the representation of the reading includes degree conversions
for the sake of convenience, the reading returned by querying the properties is
always in radians (you can convert to degrees with the built-in function
:func:`math.degrees`).

.. code-block:: pycon

    >>> for state in hat.imu:
    ...     print(repr(state))
    ...
    IMUState(compass=IMUVector(x=-13.9255, y=-30.4649, z=-18.815), gyro=IMUVector(x=0.0393031, y=0.00371209, z=-0.0437528), accel=IMUVector(x=0.0409734, y=0.0517148, z=1.00427), orient=IMUOrient(roll=2.17333 (124.5°), pitch=-1.18527 (-67.9°), yaw=2.81119 (161.1°)))
    IMUState(compass=IMUVector(x=-19.879, y=-29.4562, z=-7.37771), gyro=IMUVector(x=0.040144, y=-0.00145538, z=-0.0430174), accel=IMUVector(x=0.0431554, y=0.0495297, z=1.00939), orient=IMUOrient(roll=2.09063 (119.8°), pitch=-1.15771 (-66.3°), yaw=2.85458 (163.6°)))
    IMUState(compass=IMUVector(x=-19.879, y=-29.4562, z=-7.37771), gyro=IMUVector(x=0.040144, y=-0.00145538, z=-0.0430174), accel=IMUVector(x=0.0431554, y=0.0495297, z=1.00939), orient=IMUOrient(roll=2.09063 (119.8°), pitch=-1.15771 (-66.3°), yaw=2.85458 (163.6°)))
    IMUState(compass=IMUVector(x=-19.879, y=-29.4562, z=-7.37771), gyro=IMUVector(x=0.040144, y=-0.00145538, z=-0.0430174), accel=IMUVector(x=0.0431554, y=0.0495297, z=1.00939), orient=IMUOrient(roll=2.09063 (119.8°), pitch=-1.15771 (-66.3°), yaw=2.85458 (163.6°)))
    IMUState(compass=IMUVector(x=-19.879, y=-29.4562, z=-7.37771), gyro=IMUVector(x=0.040144, y=-0.00145538, z=-0.0430174), accel=IMUVector(x=0.0431554, y=0.0495297, z=1.00939), orient=IMUOrient(roll=2.09063 (119.8°), pitch=-1.15771 (-66.3°), yaw=2.85458 (163.6°)))
    IMUState(compass=IMUVector(x=-24.5605, y=-28.5779, z=1.99134), gyro=IMUVector(x=0.0379679, y=0.00247297, z=-0.0392915), accel=IMUVector(x=0.0421856, y=0.0500153, z=1.01597), orient=IMUOrient(roll=2.01459 (115.4°), pitch=-1.13169 (-64.8°), yaw=2.89324 (165.8°)))
    IMUState(compass=IMUVector(x=-24.5605, y=-28.5779, z=1.99134), gyro=IMUVector(x=0.0379679, y=0.00247297, z=-0.0392915), accel=IMUVector(x=0.0421856, y=0.0500153, z=1.01597), orient=IMUOrient(roll=2.01459 (115.4°), pitch=-1.13169 (-64.8°), yaw=2.89324 (165.8°)))
    IMUState(compass=IMUVector(x=-24.5605, y=-28.5779, z=1.99134), gyro=IMUVector(x=0.0379679, y=0.00247297, z=-0.0392915), accel=IMUVector(x=0.0421856, y=0.0500153, z=1.01597), orient=IMUOrient(roll=2.01459 (115.4°), pitch=-1.13169 (-64.8°), yaw=2.89324 (165.8°)))
    IMUState(compass=IMUVector(x=-24.5605, y=-28.5779, z=1.99134), gyro=IMUVector(x=0.0379679, y=0.00247297, z=-0.0392915), accel=IMUVector(x=0.0421856, y=0.0500153, z=1.01597), orient=IMUOrient(roll=2.01459 (115.4°), pitch=-1.13169 (-64.8°), yaw=2.89324 (165.8°)))


Further Reading
===============

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
