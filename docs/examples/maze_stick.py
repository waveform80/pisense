import numpy as np
import pisense as zs
from random import sample
from colorzero import Color
from time import sleep


def main(width=8, height=8):
    walls = generate_maze(width, height)
    with zs.SenseHAT() as hat:
        inputs = moves(hat.stick)
        outputs = game(walls, width, height, inputs)
        display(hat.screen, outputs)


def moves(stick):
    for event in stick:
        if event.pressed:
            try:
                delta_y, delta_x = {
                    'left': (0, -1),
                    'right': (0, 1),
                    'up': (-1, 0),
                    'down': (1, 0),
                }[event.direction]
                yield delta_y, delta_x
            except KeyError:
                break


def display(screen, states):
    for anim, data in states:
        if anim == 'fade':
            screen.fade_to(data)
        elif anim == 'zoom':
            screen.zoom_to(data, center=(6, 6))
        elif anim == 'show':
            screen.array = data
        elif anim == 'scroll':
            screen.scroll_text(data, background=Color('red'))
        else:
            assert False


def game(walls, width, height, moves):
    wall = Color('white')
    unvisited = Color('black')
    visited = Color('green')
    ball = Color('red')
    maze = draw_maze(width, height, walls, wall, unvisited)
    height, width = maze.shape
    y, x = (1, 1)
    maze[y, x] = ball
    left, right = clamp(x, width)
    top, bottom = clamp(y, height)
    yield 'fade', maze[top:bottom, left:right]
    for delta_y, delta_x in moves:
        if Color(*maze[y + delta_y, x + delta_x]) != wall:
            maze[y, x] = visited
            y += delta_y
            x += delta_x
            maze[y, x] = ball
            left, right = clamp(x, width)
            top, bottom = clamp(y, height)
            if y == height - 2 and x == width - 2:
                for state in winners_cup():
                    yield state
                break
            else:
                yield 'show', maze[top:bottom, left:right]
    yield 'fade', zs.array(Color('black'))


def generate_maze(width, height):
    # Generate the maze with Kruskal's algorithm (there's better choices,
    # but this is a simple demo!)
    sets = {frozenset({(y, x)}) for y in range(height) for x in range(width)}
    walls = set()
    for y in range(height):
        for x in range(width):
            if x > 0:
                # Add west wall
                walls.add(((y, x - 1), (y, x)))
            if y > 0:
                # Add north wall
                walls.add(((y - 1, x), (y, x)))
    for wall in sample(list(walls), k=len(walls)):
        # For a random wall, find the sets containing the adjacent cells
        a, b = wall
        set_a = set_b = None
        for s in sets:
            if {a, b} <= s:
                set_a = set_b = s
            elif a in s:
                set_a = s
            elif b in s:
                set_b = s
            if set_a is not None and set_b is not None:
                break
        # If the sets aren't the same, the cells aren't reachable; remove the
        # wall between them
        if set_a is not set_b:
            sets.add(set_a | set_b)
            sets.remove(set_a)
            sets.remove(set_b)
            walls.remove(wall)
        if len(sets) == 1:
            break
    assert len(sets) == 1
    assert sets.pop() == {(y, x) for y in range(height) for x in range(width)}
    return walls


def draw_maze(width, height, walls, wall_color, gap_color):
    result = zs.array(shape=(2 * height + 1, 2 * width + 1))
    result[...] = gap_color
    result[::2, ::2] = wall_color
    for a, b in walls:
        ay, ax = a
        by, bx = b
        y = 2 * by + 1
        x = 2 * bx + 1
        if ay == by:
            result[y, x - 1] = wall_color
        else:
            result[y - 1, x] = wall_color
    result[0, :] = result[:, 0] = wall_color
    result[-1, :] = result[:, -1] = wall_color
    return result


def clamp(pos, limit, window=8):
    low, high = pos - window // 2, pos + window // 2
    if low < 0:
        high += -low
        low = 0
    elif high > limit:
        low -= high - limit
        high = limit
    return low, high


def winners_cup():
    r = Color('red')
    y = Color('yellow')
    W = Color('white')
    yield 'zoom', zs.array([
        r, r, W, y, y, y, r, r,
        r, r, W, y, y, y, r, r,
        r, r, W, y, y, y, r, r,
        r, r, r, W, y, r, r, r,
        r, r, r, W, y, r, r, r,
        r, r, r, W, y, r, r, r,
        r, r, r, W, y, r, r, r,
        r, r, W, y, y, y, r, r,
    ])
    sleep(2)
    yield 'fade', zs.array(r)
    yield 'scroll', 'You win!'


if __name__ == '__main__':
    main()
