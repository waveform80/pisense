# vim: set et sw=4 sts=4 fileencoding=utf-8:
#
# Alternative API for the Sense HAT
# Copyright (c) 2016-2018 Dave Jones <dave@waveform.org.uk>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the copyright holder nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Defines a set of routines for generating screen animations. These routines
are mirrored as (simpler) methods on the :class:`SenseScreen` class.
"""

from __future__ import (
    unicode_literals,
    absolute_import,
    print_function,
    division,
)

import os
import atexit

import numpy as np
from pkg_resources import resource_filename, cleanup_resources
from PIL import Image, ImageDraw, ImageFont
from colorzero import Color

from .easings import linear
from .formats import buf_to_image


_FONT_CACHE = {}

def _load_font(font, size):
    if isinstance(font, ImageFont.ImageFont):
        return font
    try:
        f = _FONT_CACHE[font]
    except KeyError:
        if font.endswith('.pil') and not os.path.exists(font):
            # PIL's internal font format is rather annoying in that it
            # requires *two* files (of which only one is specified in the
            # load() method). As a result, we can't use resource_stream
            # and have to (potentially) extract the resources to the
            # file-system (and register a manual clean-up routine).
            atexit.register(cleanup_resources)
            pil_file = resource_filename(__name__, font)
            # pylint: disable=unused-variable
            pbm_file = resource_filename(__name__, font[:-4] + '.pbm')
            f = ImageFont.load(pil_file)
        else:
            try:
                f = ImageFont.load(font)
            except OSError:
                f = ImageFont.truetype(font, size)
        _FONT_CACHE[font] = f
    return f


def draw_text(text, font='default.pil', size=8, foreground=Color('white'),
              background=Color('black'), padding=(0, 0, 0, 0), min_height=8):
    """
    Renders the string *text* in the specified *font* and *size*, returning the
    result as an :class:`~PIL.Image.Image`.

    The :func:`draw_text` function is useful for generating an
    :class:`~PIL.Image.Image` containing the specified *text* rendered in the
    given *font* and *size*. The default *font* (``default.pil``) is a fixed
    height, variable width font particularly suited to low resolution displays
    like the Sense HAT (the font is limited to 5x7 grid).

    One other specially made font (``small.pil``) is also provided which
    limits itself to a 3x5 grid. It is less readable than ``default.pil`` but
    can fit more on the display which can be useful in certain circumstances.
    Other valid values for *font* are any TrueType or OpenType font installed
    on the system. If the font is within the font search path, only the base
    filename needs to be specified. For example::

        >>> from pisense import *
        >>> img = draw_text('Hello!', font='Piboto-Light.ttf')
        >>> img.size
        # TODO insert size
        >>> arr = array(img)
        >>> arr.show()
        # TODO insert preview

    As can be seen, when rendered small most TrueType and OpenType fonts don't
    look very good (although there are some exceptions) although they do look
    "smoother" due to the anti-aliasing used. Here's the default font for
    comparison:

        >>> img = draw_text('Hello!')
        >>> img.size
        # TODO insert size
        >>> arr = array(img)
        >>> arr.show()
        # TODO insert preview

    The *foreground* and *background* parameters specify
    :class:`~colorzero.Color` instances for the text and background colors
    respectively, which default to white text on a black background.

    The *padding* parameter specifies the number of pixels of padding that
    should be included in the resulting image. This is specified as a 4-tuple
    of values representing the left, top, right, and bottom padding
    respectively. The default is no padding.

    Finally, the *min_height* parameter ensures the resulting image (including
    padding) is guaranteed to be at least *min_height* pixels high.  This
    defaults to 8 and is a convenience for when you know you are working with a
    smaller font (like ``default.pil`` or ``small.pil``). It ensures that
    horizontal slices of the result can be assigned to the display without
    worrying about the vertical slicing.
    """
    # pylint: disable=too-many-arguments,too-many-locals
    if not isinstance(foreground, Color):
        foreground = Color(*foreground)
    if not isinstance(background, Color):
        background = Color(*background)
    f = _load_font(font, size)
    width, height = f.getsize(text)
    pad_left, pad_top, pad_right, pad_bottom = padding
    pad_top = max(pad_top, min_height - (pad_top + height + pad_bottom))
    img = Image.new('RGB', (
        pad_left + width + pad_right,
        pad_top + height + pad_bottom
    ))
    draw = ImageDraw.Draw(img)
    draw.rectangle(((0, 0), img.size), background.rgb_bytes)
    draw.text((pad_left, pad_top), text, foreground.rgb_bytes, f)
    return img


def scroll_text(text, font='default.pil', size=8, foreground=Color('white'),
                background=Color('black'), direction='left',
                duration=None, fps=15):
    """
    Generator function which yields a series of frames depicting *text*
    scrolling in *direction* across the display. Each frame will be a
    :class:`~PIL.Image.Image` 8x8 pixels in size.

    The *text*, *font*, *size*, *foreground*, and *background* parameters are
    all equivalent to those in :func:`draw_text` (which is called to handle
    rendering the text).

    The *direction* parameter defaults to 'left' which results in the text
    scrolling from the right-hand side of the display towards the left (the
    typical direction for left-to-right languages). The value 'right' can also
    be specified to reverse the scrolling direction.

    The *duration* and *fps* parameters control how many frames will be yielded
    by the function. The *duration* parameter measures the length of the
    animation in seconds, while *fps* controls how many frames should be shown
    per second. Hence, if *duration* is 2 and *fps* is 15 (the default), the
    generator will yield 30 frames. The default for *duration* is ``None``
    indicating that the function should determine the duration based on the
    length of the rendered text. In this case the generator will produce frames
    which scroll 1 pixel horizontally per frame.

    The resulting animation will start with a full frame of *background* color;
    the text will appear to scroll onto the display, and off again with the
    final frame guaranteed to be another full frame of *background* color.
    """
    # pylint: disable=too-many-arguments
    #
    # +8 for blank screens either side (to let the text scroll onto and
    # off of the display) and +1 to compensate for spillage due to anti-
    # aliasing
    img = draw_text(text, font, size, foreground, background,
                    padding=(9, 0, 9, 0))
    if duration is None:
        steps = img.size[0] - 8
    else:
        steps = int(duration * fps)
    x_inc = (img.size[0] - 8) / steps
    try:
        x_steps = {
            'left': range(steps),
            'right': range(steps, -1, -1),
        }[direction]
    except KeyError:
        raise ValueError('invalid direction')
    for frame in (
            img.crop((x, 0, x + 8, img.size[1]))
            for x_step in x_steps
            for x in (int(x_step * x_inc),)
        ):
        yield frame
    # Guarantee the final frame is solid background color
    yield Image.new('RGB', (8, 8), background.rgb_bytes)


def fade_to(start, finish, duration=1, fps=15, easing=linear):
    """
    Generator function which yields a series of frames fading from the *start*
    frame to the *finish* frame. Each frame will be a :class:`~PIL.Image.Image`
    with the same size as the *start* and *finish* frames (which must be the
    same size).

    The *duration* and *fps* parameters control how many frames will be yielded
    by the function. The *duration* parameter measures the length of the
    animation in seconds, while *fps* controls how many frames should be shown
    per second. Hence, if *duration* is 1 (the default) and *fps* is 15 (the
    default), the generator will yield 15 frames.

    The *easing* parameter specifies a function which controls the progression
    of the fade. See :ref:`easing` for more information.
    """
    start = buf_to_image(start)
    finish = buf_to_image(finish)
    if start.size != finish.size:
        raise ValueError("start and finish frames must be the same size")
    mask = np.empty((8, 8), np.uint8)
    mask_img = Image.frombuffer('L', (8, 8), mask, 'raw', 'L', 0, 1)
    for f in easing(int(duration * fps)):
        mask[...] = int(255 * f)
        frame = start.copy()
        frame.paste(finish, (0, 0), mask_img)
        yield frame


def slide_to(start, finish, direction='left', cover=False, duration=1, fps=15,
             easing=linear):
    """
    Generator function which yields a series of frames depicting the *finish*
    sliding onto the display, covering or displacing the *start* frame.  Each
    frame will be a :class:`~PIL.Image.Image` with the same size as the *start*
    and *finish* frames (which must be the same size).

    The *direction* parameter controls which way the *finish* frame appears to
    slide onto the display. It defaults to 'left' but can also be 'right',
    'up', or 'down'. If the *cover* parameter is ``False`` (the default), then
    the *start* frame will appear to slide off the display in the same
    direction. If *cover* is ``True``, then the *finish* frame will slide over
    the *start* frame appearing to cover it.

    The *duration* and *fps* parameters control how many frames will be yielded
    by the function. The *duration* parameter measures the length of the
    animation in seconds, while *fps* controls how many frames should be shown
    per second. Hence, if *duration* is 1 (the default) and *fps* is 15 (the
    default), the generator will yield 15 frames.

    The *easing* parameter specifies a function which controls the progression
    of the fade. See :ref:`easing` for more information.
    """
    # pylint: disable=too-many-arguments
    try:
        delta_x, delta_y = {
            'left':  (-1, 0),
            'right': (1, 0),
            'up':    (0, -1),
            'down':  (0, 1),
        }[direction]
    except KeyError:
        raise ValueError('invalid direction: ' % direction)
    start = buf_to_image(start)
    finish_small = buf_to_image(finish)
    if start.size != finish_small.size:
        raise ValueError("start and finish frames must be the same size")
    start = start.resize((64, 64))
    finish = finish_small.resize((64, 64))
    if not cover:
        canvas = Image.new('RGB', (64, 64))
    for f in easing(int(duration * fps)):
        x = int(delta_x * f * 64)
        y = int(delta_y * f * 64)
        if cover:
            canvas = start.copy()
        else:
            canvas.paste(start, (x, y))
        canvas.paste(finish, (64 * -delta_x + x, 64 * -delta_y + y))
        yield canvas.resize((8, 8), Image.BOX)
    # Ensure the final frame is the finish image (without resizing blur)
    yield finish_small


def zoom_to(start, finish, center=(4, 4), direction='in', duration=1, fps=15,
            easing=linear):
    """
    Generator function which yields a series of frames depicting the *finish*
    zooming to fill the display, with the *start* frame ballooning out of the
    display or shrinking to a point. Each frame will be a
    :class:`~PIL.Image.Image` with the same size as the *start* and *finish*
    frames (which must be the same size).

    The *direction* parameter defaults to 'in' which means the *finish* frame
    will start as a single point at the (x, y) coordinates given by *center*,
    and will expand to fill the display. The *direction* can also be 'out' in
    which case the *start* frame will shrink towards to the *center* point with
    the *finish* frame appearing around the edges.

    The *duration* and *fps* parameters control how many frames will be yielded
    by the function. The *duration* parameter measures the length of the
    animation in seconds, while *fps* controls how many frames should be shown
    per second. Hence, if *duration* is 1 (the default) and *fps* is 15 (the
    default), the generator will yield 15 frames.

    The *easing* parameter specifies a function which controls the progression
    of the fade. See :ref:`easing` for more information.
    """
    # pylint: disable=too-many-arguments
    if direction == 'in':
        base = buf_to_image(start)
        top = buf_to_image(finish)
        final = top
    elif direction == 'out':
        final = buf_to_image(finish)
        base = final
        top = buf_to_image(start).copy()
    else:
        raise ValueError('invalid direction: %s' % direction)
    if base.size != top.size:
        raise ValueError("start and finish frames must be the same size")
    base = base.resize((64, 64))
    mask = np.empty((8, 8), np.uint8)
    mask_img = Image.frombuffer('L', (8, 8), mask, 'raw', 'L', 0, 1)
    for f in easing(int(duration * fps)):
        if direction == 'out':
            f = 1 - f
        mask[...] = int(255 * f)
        frame = base.copy()
        frame.paste(top, (center[0] * 8, center[1] * 8), mask_img)
        frame = frame.crop((
            int(center[0] * f * 8),
            int(center[1] * f * 8),
            int(64 - f * 8 * (8 - (center[0] + 1))),
            int(64 - f * 8 * (8 - (center[1] + 1))),
        ))
        yield frame.resize((8, 8), Image.BOX)
    # Ensure the final frame is the finish image (without resizing blur)
    yield final
