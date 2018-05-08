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
native_str = str
str = type('')


import os
import atexit

import numpy as np
from pkg_resources import resource_filename, cleanup_resources
from PIL import Image, ImageDraw, ImageFont
from colorzero import Color

from .easings import linear
from .images import (
    color,
    buf_to_image,
    image_to_rgb565,
    rgb565_to_image,
    rgb_to_rgb565,
    rgb565_to_rgb,
)


_font_cache = {}

def _load_font(font, size):
    if isinstance(font, ImageFont.ImageFont):
        return font
    try:
        f = _font_cache[font]
    except KeyError:
        if font.endswith('.pil') and not os.path.exists(font):
            # PIL's internal font format is rather annoying in that it
            # requires *two* files (of which only one is specified in the
            # load() method). As a result, we can't use resource_stream
            # and have to (potentially) extract the resources to the
            # file-system (and register a manual clean-up routine).
            atexit.register(cleanup_resources)
            pil_file = resource_filename(__name__, font)
            pbm_file = resource_filename(__name__, font[:-4] + '.pbm')
            f = ImageFont.load(pil_file)
        else:
            try:
                f = ImageFont.load(font)
            except OSError:
                f = ImageFont.truetype(font, size)
        _font_cache[font] = f
    return f


def draw_text(text, font='default.pil', size=8, foreground=Color('white'),
              background=Color('black'), padding=(0, 0, 0, 0), min_height=8):
    if not isinstance(foreground, Color):
        foreground = Color(*foreground)
    if not isinstance(background, Color):
        background = Color(*background)
    f = _load_font(font, size)
    w, h = f.getsize(text)
    h = max(h, min_height)
    pl, pt, pr, pb = padding
    img = Image.new('RGB', (pl + w + pr, pt + h + pb))
    draw = ImageDraw.Draw(img)
    draw.rectangle(((0, 0), img.size), background.rgb_bytes)
    draw.text((pl, pt), text, foreground.rgb_bytes, f)
    return img


def scroll_text(text, font='default.pil', size=8, foreground=Color('white'),
                background=Color('black'), direction='left',
                duration=None, fps=15):
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
    start = buf_to_image(start)
    finish = buf_to_image(finish)
    mask = np.empty((8, 8), np.uint8)
    mask_img = Image.frombuffer('L', (8, 8), mask, 'raw', 'L', 0, 1)
    frames = []
    for f in easing(int(duration * fps)):
        mask[...] = int(255 * f)
        frame = start.copy()
        frame.paste(finish, (0, 0), mask_img)
        yield frame


def slide_to(start, finish, direction='left', cover=False, duration=1, fps=15,
             easing=linear):
    try:
        delta_x, delta_y = {
            'left':  (-1, 0),
            'right': (1, 0),
            'up':    (0, -1),
            'down':  (0, 1),
        }[direction]
    except KeyError:
        raise ValueError('invalid direction: ' % direction)
    start = buf_to_image(start).resize((64, 64))
    finish_small = buf_to_image(finish)
    finish = finish_small.resize((64, 64))
    if not cover:
        canvas = Image.new('RGB', (64, 64))
    frames = []
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
    if direction == 'in':
        base = buf_to_image(start).resize((64, 64))
        top = buf_to_image(finish)
        final = top
    elif direction == 'out':
        final = buf_to_image(finish)
        base = final.resize((64, 64))
        top = buf_to_image(start).copy()
    else:
        raise ValueError('invalid direction: %s' % direction)
    frames = []
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
