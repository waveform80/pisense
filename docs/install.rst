============
Installation
============

.. currentmodule:: pisense


Raspbian installation
=====================

On `Raspbian`_, it is best to obtain colorzero via the ``apt`` utility:

.. code-block:: console

    $ sudo apt update
    $ sudo apt install python-pisense python3-pisense

The usual apt upgrade method can be used to keep your installation up to date:

.. code-block:: console

    $ sudo apt update
    $ sudo apt upgrade

To remove your installation:

.. code-block:: console

    $ sudo apt remove python-pisense python3-pisense


Other platforms
===============

On other platforms, it is probably easiest to obtain colorzero via the ``pip``
utility:

.. code-block:: console

    $ sudo pip install pisense
    $ sudo pip3 install pisense

To upgrade your installation:

.. code-block:: console

    $ sudo pip install -U pisense
    $ sudo pip3 install -U pisense

To remove your installation:

.. code-block:: console

    $ sudo pip remove pisense
    $ sudo pip3 remove pisense


.. _Raspbian: https://www.raspberrypi.org/downloads/raspbian/
