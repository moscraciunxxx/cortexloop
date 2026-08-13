from __future__ import annotations

import math

from .world import FLOOR, WALL, World, tile_color


def render_camera(world: World, width: int = 96, height: int = 64, fov: float = 1.15) -> bytes:
    """Wolfenstein-style 2.5D RGB view from the robot's pose."""
    r = world.robot
    out = bytearray(width * height * 3)
    for col in range(width):
        ang = r.theta - fov / 2 + fov * (col + 0.5) / width
        dx, dy = math.cos(ang), math.sin(ang)
        dist = 0.0
        hit = WALL
        step = 0.04
        hx = r.x
        hy = r.y
        while dist < 14.0:
            hx += dx * step
            hy += dy * step
            dist += step
            t = world.tile(hx, hy)
            if t != FLOOR:
                hit = t
                break
        shade = max(0.18, min(1.0, 1.4 / (0.35 + dist)))
        cr, cg, cb = tile_color(hit)
        wall_h = int(height * 1.15 / (0.35 + dist))
        y0 = max(0, height // 2 - wall_h // 2)
        y1 = min(height, height // 2 + wall_h // 2)
        for row in range(height):
            i = (row * width + col) * 3
            if row < y0:
                # ceiling
                k = 18 + int(22 * (1 - row / max(1, y0)))
                out[i : i + 3] = bytes((k, k + 2, k + 6))
            elif row >= y1:
                # floor
                f = 28 + int(18 * ((row - y1) / max(1, height - y1)))
                out[i : i + 3] = bytes((f, f + 2, f + 4))
            else:
                out[i] = min(255, int(cr * shade))
                out[i + 1] = min(255, int(cg * shade))
                out[i + 2] = min(255, int(cb * shade))
                # center reticle column
                if abs(col - width // 2) <= 1 and abs(row - height // 2) <= 1:
                    out[i : i + 3] = bytes((255, 255, 255))
    return bytes(out)


def render_centered_class(class_id: int, width: int = 32, height: int = 32, rng=None) -> bytes:
    """Synthetic close-up used for training, matching tile palette + noise."""
    import random as _random

    rng = rng or _random.Random()
    cr, cg, cb = tile_color(class_id)
    out = bytearray(width * height * 3)
    bg = (36 + rng.randint(0, 10), 38 + rng.randint(0, 10), 42 + rng.randint(0, 12))
    for y in range(height):
        for x in range(width):
            i = (y * width + x) * 3
            out[i : i + 3] = bytes(bg)
    x0 = width // 6 + rng.randint(-1, 1)
    y0 = height // 6 + rng.randint(-1, 1)
    x1 = width - x0
    y1 = height - y0
    for y in range(y0, y1):
        for x in range(x0, x1):
            i = (y * width + x) * 3
            n = rng.randint(-12, 12)
            out[i] = max(0, min(255, cr + n))
            out[i + 1] = max(0, min(255, cg + n))
            out[i + 2] = max(0, min(255, cb + n))
    if class_id == WALL:
        for y in range(y0, y1, 4):
            for x in range(x0, x1):
                i = (y * width + x) * 3
                out[i : i + 3] = bytes((70, 72, 78))
    if class_id == 5:  # hazard stripes
        for y in range(y0, y1):
            for x in range(x0, x1):
                if ((x + y) // 3) % 2 == 0:
                    i = (y * width + x) * 3
                    out[i : i + 3] = bytes((20, 20, 20))
    return bytes(out)
