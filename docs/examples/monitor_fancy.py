from pisense import SenseHAT, array, draw_text, image_to_rgb
from colorzero import Color, Red, Green, Blue
from time import sleep
from itertools import cycle, chain
import numpy as np


def thermometer(offset, reading):
    t = max(0, min(50, reading.temperature)) / 50 * 64
    screen = array([
        Color('red') if i < int(t) else
        Color('red') * Red(t - int(t)) if i < t else
        Color('black')
        for i in range(64)
    ])
    screen = np.flipud(screen)
    text = image_to_rgb(draw_text(str(round(reading.temperature)),
                                  'small.pil', foreground=Color('gray'),
                                  padding=(0, 0, 0, 3)))
    screen[:text.shape[0], :text.shape[1]] += text
    return screen.clip(0, 1)


def hygrometer(offset, reading):
    h = reading.humidity / 100 * 64
    screen = array([
        Color('#008') if i < int(h) else
        Color('#008') * Blue(h - int(h)) if i < h else
        Color('black')
        for i in range(64)
    ])
    screen = np.flipud(screen)
    text = image_to_rgb(draw_text('^^' if reading.humidity > 99 else
                                  str(round(reading.humidity)),
                                  'small.pil', foreground=Color('gray'),
                                  padding=(0, 0, 0, 3)))
    screen[:text.shape[0], :text.shape[1]] += text
    return screen.clip(0, 1)


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
                                  padding=(0, 0, 8, 3)))
    screen[:text.shape[0], :] += text[:, offset:offset + 8]
    return screen.clip(0, 1)


def bounce(it):
    # bounce('ABC') --> A B C C B A A B C ...
    return cycle(chain(it, reversed(it)))


def switcher(events, readings):
    screens = {
        (thermometer, 'right'): hygrometer,
        (hygrometer, 'left'): thermometer,
        (hygrometer, 'right'): barometer,
        (barometer, 'left'): hygrometer,
    }
    screen = thermometer
    for event, offset, reading in zip(events, bounce(range(8)), readings):
        anim = 'draw'
        if event is not None and event.pressed:
            try:
                screen = screens[screen, event.direction]
                anim = event.direction
            except KeyError:
                yield array(Color('black')), 'fade'
                break
        yield screen(offset, reading), anim
        sleep(0.2)


def main():
    with SenseHAT() as hat:
        hat.stick.stream = True
        for a, anim in switcher(hat.stick, hat.environ):
            if anim == 'fade':
                hat.screen.fade_to(a, duration=0.5)
            elif anim == 'right':
                hat.screen.slide_to(a, direction='left', duration=0.5)
            elif anim == 'left':
                hat.screen.slide_to(a, direction='right', duration=0.5)
            else:
                hat.screen.array = a


if __name__ == '__main__':
    main()
