from __future__ import division  # for py2.x compatibility
from pisense import SenseHAT, array
from colorzero import Color
from time import sleep

def movements(imu):
    for reading in imu:
        delta_x = int(round(max(-1, min(1, imu.accel.x))))
        delta_y = int(round(max(-1, min(1, imu.accel.y))))
        if delta_x != 0 or delta_y != 0:
            yield delta_x, delta_y
        sleep(1/10)

def arrays(moves):
    a = array(Color('black'))  # blank screen
    x = y = 3
    a[y, x] = Color('white')
    yield a  # initial position
    for dx, dy in moves:
        a[y, x] = Color('black')
        x = max(0, min(7, x + dx))
        y = max(0, min(7, y + dy))
        a[y, x] = Color('white')
        yield a
    a[y, x] = Color('black')
    yield a  # end with a blank display

with SenseHAT() as hat:
    for a in arrays(movements(hat.imu)):
        hat.screen.array = a
