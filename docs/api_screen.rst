============
API - Screen
============

.. currentmodule:: pisense

.. autoclass:: SenseScreen

.. autoclass:: ScreenArray

.. autofunction:: array


Animations
==========

.. autofunction:: scroll_text

.. autofunction:: fade_to

.. autofunction:: slide_to

.. autofunction:: zoom_to


Easing
======

.. autofunction:: linear

.. autofunction:: ease_in

.. autofunction:: ease_out

.. autofunction:: ease_in_out


Gamma tables
============

.. data:: default_gamma

    The default gamma table, which can be assigned directly to
    :attr:`~SenseScreen.gamma`. The default rises in a steady curve from 0
    (off) to 31 (full brightness).

.. data:: low_gamma

    The "low light" gamma table, which can be assigned directly to
    :attr:`~SenseScreen.gamma`. The low light table rises in a steady curve
    from 0 (off) to 10.
