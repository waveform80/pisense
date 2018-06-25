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

from __future__ import (
    unicode_literals,
    absolute_import,
    print_function,
    division,
)

import mock
import pytest
from PIL import Image, ImageFont
from pisense import *
from pisense.anim import _load_font, _FONT_CACHE


def teardown_function(function):
    _FONT_CACHE.clear()


def compare_images(a, b):
    assert a.size == b.size
    assert a.mode == b.mode
    for i, (ai, bi) in enumerate(zip(a.getdata(), b.getdata())):
        if ai != bi:
            assert False, 'Pixels at (%d, %d) differ: %r != %r' % (
                i % a.size[0], i // a.size[0], ai, bi)


def test_load_font():
    font = mock.Mock(spec=ImageFont.ImageFont)
    assert _load_font(font, None) is font


def test_load_default_font():
    with mock.patch('os.path.exists') as exists, \
            mock.patch('PIL.ImageFont.load') as load, \
            mock.patch('PIL.ImageFont.truetype') as truetype:
        exists.return_value = True
        _load_font('default.pil', 8)
        assert load.call_args == mock.call('default.pil')
        assert truetype.call_args is None
        # XXX set() around .keys is 2.7 compat
        assert {('default.pil', None)} == set(_FONT_CACHE.keys())


def test_load_small_font():
    with mock.patch('os.path.exists') as exists, \
            mock.patch('pisense.anim.resource_filename') as resource_filename, \
            mock.patch('PIL.ImageFont.load') as load, \
            mock.patch('PIL.ImageFont.truetype') as truetype:
        exists.return_value = False
        _load_font('small.pil', 8)
        resource_filename.assert_has_calls([
            mock.call('pisense.anim', 'small.pil'),
            mock.call('pisense.anim', 'small.pbm'),
        ], any_order=True)
        assert load.call_args == mock.call(resource_filename('pisense.anim', 'small.pil'))
        assert truetype.call_args is None
        # XXX set() around .keys is 2.7 compat
        assert {('small.pil', None)} == set(_FONT_CACHE.keys())


def test_load_truetype_font():
    with mock.patch('os.path.exists') as exists, \
            mock.patch('PIL.ImageFont.load') as load, \
            mock.patch('PIL.ImageFont.truetype') as truetype:
        load.side_effect = OSError(2, 'File not found')
        _load_font('Arial.ttf', 8)
        assert truetype.call_args == mock.call('Arial.ttf', 8)
        # XXX set() around .keys is 2.7 compat
        assert {('Arial.ttf', 8)} == (_FONT_CACHE.keys())


_ = 0
HELLO_DEFAULT = [
    1, _, _, _, 1, _, _, _, _, _, _, _, 1, 1, _, _, 1, 1, _, _, _, _, _, _, _, _,
    1, _, _, _, 1, _, _, _, _, _, _, _, _, 1, _, _, _, 1, _, _, _, _, _, _, _, _,
    1, _, _, _, 1, _, _, 1, 1, 1, _, _, _, 1, _, _, _, 1, _, _, _, 1, 1, 1, _, _,
    1, 1, 1, 1, 1, _, 1, _, _, _, 1, _, _, 1, _, _, _, 1, _, _, 1, _, _, _, 1, _,
    1, _, _, _, 1, _, 1, 1, 1, 1, 1, _, _, 1, _, _, _, 1, _, _, 1, _, _, _, 1, _,
    1, _, _, _, 1, _, 1, _, _, _, _, _, _, 1, _, _, _, 1, _, _, 1, _, _, _, 1, _,
    1, _, _, _, 1, _, _, 1, 1, 1, _, _, 1, 1, 1, _, 1, 1, 1, _, _, 1, 1, 1, _, _,
]
HELLO_SMALL = [
    1, _, 1, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    1, _, 1, _, _, 1, 1, _, 1, _, _, 1, _, _, _, 1, _, _,
    1, 1, 1, _, 1, _, 1, _, 1, _, _, 1, _, _, 1, _, 1, _,
    1, _, 1, _, 1, 1, _, _, 1, _, _, 1, _, _, 1, _, 1, _,
    1, _, 1, _, _, 1, 1, _, _, 1, _, _, 1, _, _, 1, _, _,
]
del _


def test_draw_text_Hello():
    img = Image.new('1', (len(HELLO_DEFAULT) // 7, 7))
    img.putdata(HELLO_DEFAULT)
    img = img.convert('RGB')
    compare_images(img, draw_text('Hello', min_height=0))


def test_draw_text_small_Hello():
    img = Image.new('1', (len(HELLO_SMALL) // 5, 5))
    img.putdata(HELLO_SMALL)
    img = img.convert('RGB')
    compare_images(img, draw_text('Hello', font='small.pil', min_height=0))


def test_draw_text_min_height():
    w = len(HELLO_SMALL) // 5
    s = [0] * w * 3 + HELLO_SMALL
    img = Image.new('1', (len(s) // 8, 8))
    img.putdata(s)
    img = img.convert('RGB')
    compare_images(img, draw_text('Hello', font='small.pil', min_height=8))


def test_draw_text_colors():
    img = Image.new('RGB', (len(HELLO_SMALL) // 5, 5))
    img.putdata([
        (255, 0, 0) if c else (255, 255, 255)
        for c in HELLO_SMALL
    ])
    compare_images(img, draw_text('Hello', font='small.pil',
                                  min_height=0,
                                  foreground=(255, 0, 0),
                                  background=(255, 255, 255)))


def test_scroll_text():
    w = len(HELLO_SMALL) // 5
    s = [0] * w * 3 + HELLO_SMALL
    s = [ # break s into a list of rows
        s[i:i + len(s) // 8]
        for i in range(0, len(s), len(s) // 8)
    ]
    s = [ # add left / right padding
        [0] * 9 + i + [0] * 9
        for i in s
    ]
    for i, frame in enumerate(scroll_text('Hello', font='small.pil')):
        img = Image.new('1', (8, 8))
        img.putdata([
            elem
            for row in s
            for elem in row[i: i + 8]
        ])
        img = img.convert('RGB')
        compare_images(img, frame)


def test_scroll_text_fixed_duration():
    w = len(HELLO_SMALL) // 5
    s = [0] * w * 3 + HELLO_SMALL
    s = [ # break s into a list of rows
        s[i:i + len(s) // 8]
        for i in range(0, len(s), len(s) // 8)
    ]
    s = [ # add left / right padding
        [0] * 9 + i + [0] * 9
        for i in s
    ]
    for i, frame in enumerate(scroll_text('Hello', font='small.pil',
                                          duration=len(s[0]) - 8, fps=0.5)):
        img = Image.new('1', (8, 8))
        img.putdata([
            elem
            for row in s
            for elem in row[i * 2: i * 2 + 8]
        ])
        img = img.convert('RGB')
        compare_images(img, frame)


def test_scroll_text_bad_direction():
    with pytest.raises(ValueError):
        list(scroll_text('Hello', direction='foo'))


def test_fade_to():
    start = Image.new('RGB', (8, 8), (0, 0, 0))
    middle = Image.new('RGB', (8, 8), (127, 0, 0))
    finish = Image.new('RGB', (8, 8), (255, 0, 0))
    frames = list(fade_to(start, finish, duration=3, fps=1))
    assert len(frames) == 3
    compare_images(start, frames[0])
    compare_images(middle, frames[1])
    compare_images(finish, frames[2])


def test_fade_bad():
    start = Image.new('RGB', (8, 8), (0, 0, 0))
    finish = Image.new('RGB', (8, 8), (255, 0, 0))
    with pytest.raises(ValueError):
        list(fade_to(start, finish.crop((0, 0, 4, 4))))


def test_slide_to():
    r = (255, 0, 0)
    g = (0, 255, 0)
    b = (0, 0, 255)
    W = (255, 255, 255)
    start = Image.new('RGB', (4, 4))
    start.putdata([
        r, g, b, W,
        r, g, b, W,
        r, g, b, W,
        r, g, b, W,
    ])
    middle = Image.new('RGB', (4, 4))
    middle.putdata([
        b, W, r, r,
        b, W, r, r,
        b, W, r, r,
        b, W, r, r,
    ])
    finish = Image.new('RGB', (4, 4), r)
    frames = list(slide_to(start, finish, direction='left', duration=3, fps=1))
    assert len(frames) == 3
    compare_images(start, frames[0])
    compare_images(middle, frames[1])
    compare_images(finish, frames[2])


def test_slide_over():
    r = (255, 0, 0)
    g = (0, 255, 0)
    b = (0, 0, 255)
    W = (255, 255, 255)
    start = Image.new('RGB', (4, 4))
    start.putdata([
        r, g, b, W,
        r, g, b, W,
        r, g, b, W,
        r, g, b, W,
    ])
    middle = Image.new('RGB', (4, 4))
    middle.putdata([
        r, g, r, r,
        r, g, r, r,
        r, g, r, r,
        r, g, r, r,
    ])
    finish = Image.new('RGB', (4, 4), r)
    frames = list(slide_to(start, finish, direction='left', cover=True,
                           duration=3, fps=1))
    assert len(frames) == 3
    compare_images(start, frames[0])
    compare_images(middle, frames[1])
    compare_images(finish, frames[2])


def test_slide_bad():
    start = Image.new('RGB', (4, 4))
    finish = Image.new('RGB', (4, 4))
    with pytest.raises(ValueError):
        list(slide_to(start, finish, direction='foo'))
    with pytest.raises(ValueError):
        list(slide_to(start, finish.crop((0, 0, 2, 2))))


def test_zoom_in():
    r = (255, 0, 0)
    g = (0, 255, 0)
    start = Image.new('RGB', (5, 5), r)
    finish = Image.new('RGB', (5, 5), g)
    frames = list(zoom_to(start, finish, center=(2, 2), duration=4, fps=1))
    assert len(frames) == 4
    compare_images(start, frames[0])
    # No good test for the middle frames
    compare_images(finish, frames[3])


def test_zoom_out():
    r = (255, 0, 0)
    g = (0, 255, 0)
    start = Image.new('RGB', (5, 5), r)
    finish = Image.new('RGB', (5, 5), g)
    frames = list(zoom_to(start, finish, center=(2, 2), direction='out',
                          duration=4, fps=1))
    assert len(frames) == 4
    compare_images(start, frames[0])
    # No good test for the middle frames
    compare_images(finish, frames[3])


def test_zoom_bad():
    start = Image.new('RGB', (5, 5))
    finish = Image.new('RGB', (5, 5))
    with pytest.raises(ValueError):
        list(zoom_to(start, finish, direction='foo'))
    with pytest.raises(ValueError):
        list(zoom_to(start, finish.crop((0, 0, 2, 2))))
