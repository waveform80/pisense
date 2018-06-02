==================
Project: Maze Game
==================

Here's another project for the Sense HAT that involves building a full maze
solving game. Initially this will be controlled with the joystick (because it's
easier for debugging), but at the end we'll switch to use the IMU to roll the
ball through the maze.

Let's start at a high level and work our way down. We'll construct the
application in the same manner as our earlier demos: a transformation of
inputs (initially from the joystick, later from the IMU) into a series of
screens to be shown on the display.

First some design points:

* The state of our maze can be represented as a large numpy array (larger than
  the screen anyway) which we'll slice to show on the display.

* We'll need:

  - a color to represent walls (white)

  - a color to represent unvisited spaces (black)

  - a color to represent visited spaces (green)

  - a color to represent the player's position (red)

  - a color to represent the goal (yellow)

* We'll also need:

  - a function to generate the maze

  - (possibly) a function to draw the generated maze as a numpy array

  - a transformation to convert joystick events / IMU readings into X+Y motion

  - a transformation to convert motions into new display states (essentially
    this is the "game logic")

  - a function to render the display states including any requested animations
    (just like in the final monitor script previously)

Let's start from the "top level" and work our way down. First, the imports:

.. literalinclude:: examples/maze_stick.py
    :lines: 1-5

Our "main" function will define the colors we need, call a function to generate
the maze, set up the motion transformation, the game transformation, and feed
all this to the display renderer:

.. literalinclude:: examples/maze_stick.py
    :pyobject: main

You may recall from our earlier demos (specifically :ref:`joystick_movement`)
that we had a neat little function that converted joystick events into X and Y
delta values. Let's copy that in next:

.. literalinclude:: examples/maze_stick.py
    :pyobject: moves

So far, this may look rather strange! What does it mean to call a generator
function like "moves" without a for loop? Quite simply: this creates an
instance of the generator but doesn't start evaluating it until it's used in a
loop. In other words nothing in the generator function will run *yet*. The same
goes for the "game" function which will also be a generator, looping over the
movements yielded from "moves" and yielding screens for "display" to deal
with.

Speaking of "display", that should be easy enough to deal with. It'll be a
slightly expanded version of what we used in the previous monitor example with
additional cases for zooming and scrolling text:

.. literalinclude:: examples/maze_stick.py
    :pyobject: display

Now onto the game logic itself. Let's assume that the player always starts at
the top left (which will be (1, 1) given that (0, 0) will be an external wall)
and must finish at the bottom right. We'll assume the maze generator handles
drawing the maze, including the goal, for us and we just need to handle drawing
the player's position and updating where the player has been.

We'll handle reacting to motion from the "moves" generator, preventing the
player from crossing walls (by checking the position they want to move to
doesn't have the "wall" color), and noticing when they've reached the goal
(likewise by checking the color of the position they want to move to):

.. literalinclude:: examples/maze_stick.py
    :pyobject: game

In the function above we've assumed the existence of two extra functions:

* "clamp" which, given a position (either the user's current X or Y coordinate)
  and a limit (the width or height of the maze), returns the lower and upper
  bounds we should display (on the fixed 8x8 LEDs).

* "winners_cup" which will provide some fancy "You've won!" sort of animation.
  This is called with ``yield from`` which is equivalent to iterating over it
  and yielding each result.

Let's construct "clamp" first as it's pretty easy:

.. literalinclude:: examples/maze_stick.py
    :pyobject: clamp

Now let's code some fancy animation for a user that's won. We'll zoom in to a
golden cup on a red background, fade to red, and scroll "You win!" across the
display:

.. literalinclude:: examples/maze_stick.py
    :pyobject: winners_cup

.. note::

    Not all generator functions need a loop in them!

Nearly there … now we've just got to generate the maze. There's lots of ways of
doing this but about the simplest is `Kruskal's Algorithm`_. Roughly speaking,
it works like this:

1. Start off assuming the maze has walls between every cell on every side
   (actually we'll only model the "north" and "west" walls between cells as
   that's sufficient to represent all interior walls in the maze).

   .. image:: images/maze_init.*

2. Construct a set of sets each of which represents an individual cell:

   S := {{1}, {2}, {3}, {4}}

3. Knock down a random wall and union together the sets containing the cells
   that have just been joined.

   S := {{1, 3}, {2}, {4}}

   .. image:: images/maze_during.*

4. Continue doing this until a single set remains containing all cells. At this
   point any cell must be reachable from any other cell and the maze is
   complete.

   S := {{1, 2, 3, 4}}

   .. image:: images/maze_final.*

Here's the implementation, with the actual drawing of the maze split out into
its own function:

.. literalinclude:: examples/maze_stick.py
    :pyobject: generate_maze

.. literalinclude:: examples/maze_stick.py
    :pyobject: generate_walls

At this point we should have a fully functioning maze game that looks quite
pretty. You can play it simply by running ``main()``. Once you've verified it
works, it's a simple matter to switch out the joystick for the IMU (in exactly
the same manner as in :doc:`demos`).  Here's the updated ``moves`` function
which queries the IMU instead of the joystick:

.. literalinclude:: examples/maze_imu.py
    :pyobject: moves

Finally, it would be nice to have the game run in a loop so that after the
winners screen it resets with a new maze. It would also be nice to launch the
script on boot so we can turn the Pi into a hand-held game. This is also simple
to arrange:

* We need to put an infinite loop in ``main`` to restart the game when it
  finishes

* We need to add a signal handler to shut down the game nicely when systemd
  tells it to stop (which it does by sending the SIGTERM signal; we can handle
  this with some simple routines from the built-in :mod:`signal` module).

* Just to keep things looking good, we'll add a ``try..finally`` block to
  ``display`` to ensure the screen always fades to black when the loop is
  terminated.

Here's the final listing with the updated lines highlighted:

.. literalinclude:: examples/maze_final.py
    :caption:
    :emphasize-lines:

Now to launch the game on boot, we'll create a systemd service to execute it
under the unprivileged "pi" user. Copy the following into
:file:`/etc/systemd/system/maze.service` ensuring you update the location of
the :file:`maze_final.py` script to wherever you've created it:

.. literalincluded:: examples/maze.service
    :caption:

Finally, run the following command line to enable the service on boot:

.. console:: console

    $ sudo systemctl enable maze

If you ever wish to stop the script running on boot:

.. console:: console

    $ sudo systemctl disable maze

.. _Kruskal's Algorithm: https://en.wikipedia.org/wiki/Maze_generation_algorithm#Randomized_Kruskal's_algorithm
