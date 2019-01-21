============================
Project: Environment Monitor
============================

.. currentmodule:: pisense

Here's our first "full" project for the Sense HAT: make an environmental
monitor that can display the temperature, humidity, and pressure in a variety
of forms. We've already seen a demo thermometer in :ref:`thermometer`. First
we'll construct variants of this for the humidity and pressure sensors. Then
we'll combine all three into an application. Finally, we'll add interactivity
using the joystick to select the required functionality, recording the data to
a database, and a trivial web interface.


Hygrometer
==========

Firstly, let's adapt our thermometer script for sensing humidity. Here's the
thermometer script again:

.. literalinclude:: examples/thermometer.py
    :caption:

We'll use a very similar structure for our hygrometer. This time we don't need
to clamp the range (we'll use the full 0% to 100%, but we'll scale it to 0 <= n
< 64 again). We'll use a reasonably dark blue ("#000088" in HTML terms) for the
chart, but everything else should look fairly familiar:

.. literalinclude:: examples/hygrometer.py
    :caption:

The one other subtle change is in the caption. We can't fit "100" on our
display; it's too wide (this wasn't a problem for the thermometer where we
clamped the temperature range from 0°C to 50°C; if you guessed this was for
simplicity, you were right!). Instead, whenever the humidity is >99% we display
"^^" to indicate the maximum value.

Test this script out by running it and then breathing gently on the humidity
sensor. You should see the humidity reading rise rapidly (possibly to "^^")
then slowly fall back down.


Barometer
=========

Next we'll tackle the pressure sensor. This will have a very familiar structure
by now:

1. Clamp the pressure readings to a sensible range (in this case we'll use
   950mbar to 1050mbar).

2. Scale this to the range 0 <= n < 64.

3. Draw a rudimentary chart (we'll use green to distinguish it from our
   thermometer and hygrometer scripts).

4. Draw the pressure as a number superimposed on the chart.

Oh dear, there's a problem! All the valid pressure values are too large to fit
on the display, so we can't use an easy hack like displaying "^^" as we did in
the hygrometer above.

It'd be nice if the pressure reading could scroll back and forth on the
display, still superimposed on the chart. It turns out, using iterators again,
this is actually quite easy to achieve. What we want is a sliding window over
our rendered text, like so:

.. image:: images/sliding_window.*
    :align: center

Hence our first requirement is an infinite iterator which produces the
"bouncing" X offset for the sliding window:

.. literalinclude:: examples/barometer.py
    :pyobject: bounce

Well, that was simple!

The :func:`~itertools.cycle` and :func:`~itertools.chain` functions come from
the standard library's fantastic :mod:`itertools` module which I urge anyone
using iterators to check out. The :func:`reversed` function is a standard
built-in function in Python.

How do we combine the offsets produced by ``bounce`` with the readings from the
sensor? We simply use the built-in :func:`zip` function:

.. literalinclude:: examples/barometer.py
    :caption:

.. note::

    This example will *only* work in Python 3 because it evaluates :func:`zip`
    lazily. In Python 2, this will crash as zip attempts to construct a list
    for an infinite iterator (use ``izip`` from :mod:`itertools` in Python 2).

.. admonition:: Exercise

    Can you adjust the hygrometer script so that it scrolls "100" when that is
    the reading, but smaller values stay static on the display?


Combining Screens
=================

We now have the three scripts that we want for our environmental monitor, but
how do we combine them into a single application? Our first step will be a
simple one: to make a function that will rotate between each of our
transformations periodically, first showing the thermometer for a few seconds,
then the hygrometer, then the barometer.

The easiest way to do this is to modify our thermometer and hygrometer
transforms to take a (useless) *offset* parameter just like the barometer
transform. Then (because our functions all now have a common prototype, and
functions are first class objects in Python) we can construct a
:func:`~itertools.cycle` of transforms and just loop around them. The result
looks like this:

.. literalinclude:: examples/monitor_auto.py
    :caption:


Interactivity!
==============

Switching automatically between things is okay, but it would be nicer if we
could *control* the switching with the joystick. For example, we could lay out
our screens side-by-side with thermometer at the far left, then hygrometer,
then pressure at the far right, and when the user presses left or right we
could scroll between the displays.

To do this we just need to refine our ``switcher`` function so that it depends
on both the readings (which it will pass to whatever the current transformation
is), *and* events from the joystick.

.. literalinclude:: examples/monitor_manual.py
    :pyobject: switcher

However, we have a problem: the joystick only yields events when something
happens so if we use this, our display will only update when the joystick emits
an event (because :func:`zip` will only yield a tuple of values when *all*
iterators it covers have each yielded a value).

Thankfully, there's a simple solution: the :attr:`SenseStick.stream` attribute.
When this is set to ``True`` the joystick will immediately yield a value
whenever one is requested. If no event has occurred it will simply yield
``None``. So all our script needs to do is remember to set
:attr:`SenseStick.stream` to ``True`` at the start and everything will work
happily. Just to make the exit a bit prettier we'll fade the screen to black
too:

.. literalinclude:: examples/monitor_manual.py
    :pyobject: main


Finishing Touches
=================

The fade is a nice touch, but it would be nicer if the screens would "slide"
between each other. And we've still got to add the database output too!

Thankfully this is all pretty easy to arrange. The ``main`` procedure is the
ideal place to handle transitions like fading and sliding; it just needs to be
told when to perform them. The ``switcher`` function can tell it when to do
this by yielding *two* values: the array to copy to the display, and the
transition animation to perform (if any). While we're at it, we may as well
move the fade to black to the end of the loop in ``switcher``.

.. literalinclude:: examples/monitor_fancy.py
    :pyobject: switcher

Now we enhance the ``main`` function to perform various transitions:

.. literalinclude:: examples/monitor_fancy.py
    :pyobject: main

Finally, we did promise that we're going to store the data in a database.
Ideally, we want a `round-robin database`_ for which we can use the excellent
`rrdtool`_ project (if you wish to understand the rrdtool calls below, I'd
strongly recommend reading its documentation). This provides all sorts of
facilities beyond just recording the data, including averaging it over
convenient time periods and producing good-looking charts of the data.

.. note::

    Unfortunately, the Python 3 bindings for rrdtool don't appear to be
    packaged at the moment so we'll need to install them manually. On Raspbian
    you can do this like so:

    .. code-block:: console

        $ sudo apt install rrdtool librrd-dev python3-pip
        $ sudo pip3 install rrdtool

    On other platforms the ``pip`` command will likely be similar, but the
    pre-requisites installed with ``apt`` may well differ.

We'll add a little code to construct the round-robin database if it doesn't
already exist, then add a tiny amount of code to record readings into the
database.  The final result (with the lines we've added highlighted) is as
follows:

.. literalinclude:: examples/monitor_final.py
    :caption:
    :emphasize-lines: 65-89,93,101,111-115

.. admonition:: Exercise

    At the moment, it's too easy to accidentally exit the script.  Can you make
    the application rotate around the screens (i.e. moving right from the
    barometer screen takes the user back to the thermometer screen, and
    vice-versa) and pressing the joystick is required to exit the application?

Finally, let's whip up a little web-server that we can run alongside the Sense
HAT script to allow remote clients to query our environmental data and see some
pretty graphs of the history:

.. literalinclude:: examples/monitor_server.py
    :caption:

Run this alongside the monitor script, make sure your Pi is accessible on your
local network and then visit http://your-pis-address-here:8000/ in a
web-browser.

.. note::

    We could have added this to the monitor script, but frankly there's no
    point as rrdtool includes all the locking we need to have something reading
    the database while something else writes to it. This also ensures that a
    bug in one script doesn't affect the operation of the other, and means web
    requests are far less likely to affect the operation of the Sense HAT
    interface.


Auto-start
==========

This is the sort of application it would be nice to start automatically upon
boot up. Thankfully, this is easy to arrange with a few `systemd`_ files.
Create the following under :file:`/etc/systemd/system/monitor_app.service`:

.. literalinclude:: examples/monitor_app.service
    :caption:

.. note::

    You'll need to modify the path for ``ExecStart`` to point to the location
    of your :file:`monitor_final.py` script. You may want to modify
    ``WorkingDirectory`` too if you want the database to be stored in another
    location.

Then for the web-service (if you want it), create the following under
:file:`/etc/systemd/system/monitor_web.service`:

.. literalinclude:: examples/monitor_web.service
    :caption:

.. note::

    Remember to modify ``ExecStart`` (and optionally ``WorkingDirectory``) as
    above.

Finally, inform systemd of the changes and tell it we want to start these new
services on boot-up as follows. For example, the following commands might be
used to achieve all of this:

.. code-block:: console

    $ cd /home/pi
    $ nano monitor_app.service
    $ nano monitor_web.service
    $ sudo cp monitor_*.service /etc/systemd/system/
    $ sudo systemctl daemon-reload
    $ sudo systemctl enable monitor_app
    $ sudo systemctl enable monitor_web

To start the services immediately:

.. code-block:: console

    $ sudo systemctl start monitor_app
    $ sudo systemctl start monitor_web

To stop the services immediately:

.. code-block:: console

    $ sudo systemctl stop monitor_app
    $ sudo systemctl stop monitor_web

If you want to disable these from starting at boot time you can simply run the
following commands:

.. code-block:: console

    $ sudo systemctl disable monitor_app
    $ sudo systemctl disable monitor_web

Naturally, you could disable the web service but leave the main application
running too.

.. _round-robin database: https://en.wikipedia.org/wiki/RRDtool
.. _rrdtool: https://oss.oetiker.ch/rrdtool/
.. _systemd: https://www.raspberrypi.org/documentation/linux/usage/systemd.md
