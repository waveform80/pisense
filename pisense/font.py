from __future__ import (
    unicode_literals,
    absolute_import,
    print_function,
    division,
    )
str = type('')

import io

import numpy as np

from .common import color_dtype


class SenseFont(object):
    def __init__(self, filename_or_obj):
        if isinstance(filename_or_obj, bytes):
            filename_or_obj = filename_or_obj.decode('utf-8')
        if isinstance(filename_or_obj, str):
            with io.open(filename_or_obj, 'r') as font_file:
                self._parse_font(font_file)
        else:
            self._parse_font(font_file)
        self._max_height = max(c.shape[0] for c in self._chars.values())

    @property
    def max_height(self):
        return self._max_height

    def _parse_font(self, f):
        self._chars = {}
        char = None
        lines = []
        for line in f:
            line = line.rstrip()
            if line.endswith(':'):
                if char is not None:
                    self._chars[char] = self._make_array(char, lines)
                char = line[:-1]
                lines = []
            elif line:
                lines.append(line)

    def _make_array(self, char, lines):
        rows = len(lines)
        cols = [len(line) for line in lines]
        if cols != [cols[0]] * rows:
            raise ValueError(
                'irregular number of columns in definition of char "%s"' % char)
        cols = cols[0]
        return np.fromiter(
            (c == '#' for line in lines for c in line),
            dtype=np.bool).reshape((rows, cols))

    def __getitem__(self, key):
        return self._chars[key]

    def render_line(
            self, text, foreground=(255, 255, 255), background=(0, 0, 0),
            letter_space=1, padding=(0, 0, 0, 0)):
        w = 0
        h = self.max_height
        for c in text:
            try:
                w += self[c].shape[1] + letter_space
            except KeyError:
                raise ValueError('Character "%s" does not exist in font' % c)
        w += padding[0] + padding[2]
        h += padding[1] + padding[3]
        result = np.empty((h, w), dtype=color_dtype)
        result[:] = background
        x = padding[0]
        for c in text:
            c_h, c_w = self._chars[c].shape
            result[padding[1]:padding[1] + c_h,
                   padding[0] + x:padding[0] + x + c_w][self[c]] = foreground
            x += c_w + letter_space
        return result

    def render_text(
            self, text, foreground=(255, 255, 255), background=(0, 0, 0),
            line_space=2, letter_space=1, padding=(0, 0, 0, 0)):
        lines = [
            self.render_line(line, foreground, background, letter_space)
            for line in text.splitlines()
            ]
        height = (
                sum(line.shape[0] for line in lines) +
                line_space * (len(lines) - 1) +
                padding[1] + padding[3])
        width = (
                max(line.shape[1] for line in lines) +
                padding[0] + padding[2])
        image = np.zeros((height, width), dtype=color_dtype)
        image[:] = background
        y = padding[1]
        for line in lines:
            image[y:y + line.shape[0], 0:line.shape[1]] = line
            y += line.shape[0] + line_space
        return image

