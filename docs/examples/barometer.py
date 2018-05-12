from pisense import SenseHAT, array, draw_text, image_to_rgb
from colorzero import Color, Green
from time import sleep
from itertools import cycle, chain
import numpy as np

def bounce(it):
    # bounce('ABC') --> A B C C B A A B C ...
    return cycle(chain(it, reversed(it)))

def barometer(offset, reading):
    p = (max(950, min(1050, reading.pressure)) - 950) / 100 * 64
    screen = array([
        Color('green') if i < int(p) else
        Color('green') * Green(p - int(p)) if i < p else
        Color('black')
        for i in range(64)
    ])
    screen = np.flipud(screen)
    text = image_to_rgb(draw_text(str(round(reading.pressure)),
                                  'small.pil', foreground=Color('gray'),
                                  padding=(0, 0, 8, 0)))
    screen[:text.shape[0], :] += text[:, offset:offset + 8]
    return screen.clip(0, 1)

with SenseHAT() as hat:
    for offset, reading in zip(bounce(range(8)), hat.environ):
        hat.screen.array = barometer(offset, reading)
        sleep(0.2)
