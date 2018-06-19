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

import pytest
import numpy as np
from PIL import Image
from colorzero import Color
from pisense import *


@pytest.fixture()
def img_rgb(request):
    return Image.new('RGB', (8, 8), (255, 0, 0))


@pytest.fixture()
def img_bool(request):
    return Image.new('1', (8, 8), 0)


@pytest.fixture()
def arr_black(request):
    arr = np.empty((8, 8), dtype=np.uint8)
    arr[...] = 0
    return arr


@pytest.fixture()
def arr_rgb888(request):
    arr = np.empty((8, 8, 3), dtype=np.uint8)
    arr[..., 0] = 255
    arr[..., 1] = 0
    arr[..., 2] = 0
    return arr


@pytest.fixture()
def arr_rgb565(request):
    arr = np.empty((8, 8), dtype=np.uint16)
    arr[...] = 0xF800
    return arr


@pytest.fixture()
def arr_rgb(request):
    arr = np.empty((8, 8), dtype=color)
    arr[...] = Color('red')
    return arr


def compare_images(a, b):
    assert a.size == b.size
    assert a.mode == b.mode
    for i, (ai, bi) in enumerate(zip(a.getdata(), b.getdata())):
        if ai != bi:
            assert False, 'Pixels at (%d, %d) differ: %r != %r' % (
                i % a.size[0], i // a.size[0], ai, bi)


def test_image_to_rgb888(arr_rgb888, img_rgb, img_bool):
    assert (image_to_rgb888(img_rgb) == arr_rgb888).all()
    arr_rgb888[..., 0] = 0
    assert (image_to_rgb888(img_bool) == arr_rgb888).all()


def test_image_to_rgb565(arr_rgb565, img_rgb):
    assert (image_to_rgb565(img_rgb) == arr_rgb565).all()


def test_image_to_rgb(arr_rgb, img_rgb):
    assert (image_to_rgb(img_rgb) == arr_rgb).all()


def test_rgb888_to_rgb565(arr_rgb888, arr_rgb565):
    assert (rgb888_to_rgb565(arr_rgb888) == arr_rgb565).all()
    out = np.empty((8, 8), dtype=np.uint16)
    rgb888_to_rgb565(arr_rgb888, out)
    assert (out == arr_rgb565).all()
    with pytest.raises(ValueError):
        out = np.empty((5, 5), dtype=np.uint16)
        rgb888_to_rgb565(arr_rgb888, out)


def test_rgb888_to_rgb(arr_rgb888, arr_rgb):
    assert (rgb888_to_rgb(arr_rgb888) == arr_rgb).all()
    out = np.empty((8, 8), dtype=color)
    rgb888_to_rgb(arr_rgb888, out)
    assert (out == arr_rgb).all()
    with pytest.raises(ValueError):
        out = np.empty((5, 5), dtype=color)
        rgb888_to_rgb(arr_rgb888, out)


def test_rgb565_to_rgb888(arr_rgb888, arr_rgb565):
    assert (rgb565_to_rgb888(arr_rgb565) == arr_rgb888).all()
    out = np.empty((8, 8, 3), dtype=np.uint8)
    rgb565_to_rgb888(arr_rgb565, out)
    assert (out == arr_rgb888).all()
    with pytest.raises(ValueError):
        out = np.empty((5, 5, 3), dtype=np.uint8)
        rgb565_to_rgb888(arr_rgb565, out)


def test_rgb565_to_rgb(arr_rgb, arr_rgb565):
    assert (rgb565_to_rgb(arr_rgb565) == arr_rgb).all()
    out = np.empty((8, 8), dtype=color)
    rgb565_to_rgb(arr_rgb565, out)
    assert (out == arr_rgb).all()
    with pytest.raises(ValueError):
        out = np.empty((5, 5), dtype=color)
        rgb565_to_rgb(arr_rgb565, out)


def test_rgb_to_rgb888(arr_rgb888, arr_rgb):
    assert (rgb_to_rgb888(arr_rgb) == arr_rgb888).all()
    out = np.empty((8, 8, 3), dtype=np.uint8)
    rgb_to_rgb888(arr_rgb, out)
    assert (out == arr_rgb888).all()
    with pytest.raises(ValueError):
        out = np.empty((5, 5, 3), dtype=np.uint8)
        rgb_to_rgb888(arr_rgb, out)


def test_rgb_to_rgb565(arr_rgb565, arr_rgb):
    assert (rgb_to_rgb565(arr_rgb) == arr_rgb565).all()
    out = np.empty((8, 8), dtype=np.uint16)
    rgb_to_rgb565(arr_rgb, out)
    assert (out == arr_rgb565).all()
    with pytest.raises(ValueError):
        out = np.empty((5, 5), dtype=np.uint16)
        rgb_to_rgb565(arr_rgb, out)


def test_rgb_to_image_roundtrip(arr_rgb, arr_rgb888):
    assert (image_to_rgb(rgb_to_image(arr_rgb)) == arr_rgb).all()
    with pytest.raises(ValueError):
        rgb_to_image(arr_rgb888)


def test_rgb565_to_image_roundtrip(arr_rgb565, arr_rgb):
    assert (image_to_rgb565(rgb565_to_image(arr_rgb565)) == arr_rgb565).all()
    with pytest.raises(ValueError):
        rgb565_to_image(arr_rgb)


def test_rgb888_to_image_roundtrip(arr_rgb888, arr_rgb):
    assert (image_to_rgb888(rgb888_to_image(arr_rgb888)) == arr_rgb888).all()
    with pytest.raises(ValueError):
        rgb888_to_image(arr_rgb)


def test_buf_to_rgb888(img_rgb, arr_rgb888, arr_rgb565, arr_rgb, arr_black):
    assert (buf_to_rgb888(img_rgb) == arr_rgb888).all()
    assert (buf_to_rgb888(arr_rgb888) == arr_rgb888).all()
    assert (buf_to_rgb888(arr_rgb) == arr_rgb888).all()
    arr = arr_rgb888.copy()
    arr[...] = 0
    assert (buf_to_rgb888(arr_black) == arr).all()
    assert (buf_to_rgb888(b'\xFF\x00\x00' * 64) == arr_rgb888).all()
    with pytest.raises(ValueError):
        buf_to_rgb888(arr_rgb888.astype(np.uint16))
    with pytest.raises(ValueError):
        buf_to_rgb888(arr_rgb565)
    with pytest.raises(ValueError):
        buf_to_rgb888(b'\xFF\x00\x00' * 32)
    with pytest.raises(TypeError):
        buf_to_rgb888([0] * 192)


def test_buf_to_image(img_rgb, arr_rgb888):
    compare_images(buf_to_image(img_rgb), img_rgb)
    compare_images(buf_to_image(img_rgb.convert('P')), img_rgb)
    compare_images(buf_to_image(arr_rgb888), img_rgb)


def test_buf_to_rgb(img_rgb, arr_rgb):
    assert (buf_to_rgb(arr_rgb) == arr_rgb).all()
    assert (buf_to_rgb(img_rgb) == arr_rgb).all()


def test_iter_to_rgb(arr_rgb):
    assert (iter_to_rgb([Color('red')] * 64) == arr_rgb).all()
