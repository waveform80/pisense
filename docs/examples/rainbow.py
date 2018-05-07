from pisense import SenseHAT, array
from colorzero import Color
from time import sleep

hat = SenseHAT()
offset = 0.0
while True:
    rainbow = array([
        Color(h=(x + y) / 14 + offset, s=1, v=1)
        for x in range(8)
        for y in range(8)
    ])
    hat.screen.array = rainbow
    offset += 0.01
    sleep(0.1)
