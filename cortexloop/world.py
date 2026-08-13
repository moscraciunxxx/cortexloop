from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from . import CLASS_COLORS

FLOOR, WALL, BOX_RED, BOX_BLUE, BOX_GREEN, HAZARD, PERSON, DOCK = range(8)
BOXES = {BOX_RED, BOX_BLUE, BOX_GREEN}


@dataclass
class Robot:
    x: float = 2.5
    y: float = 2.5
    theta: float = 0.0
    carrying: int = 0  # 0 = empty gripper
    radius: float = 0.32
    speed: float = 0.0
    yaw_rate: float = 0.0
    energy_j: float = 0.0
    collisions: int = 0
    delivered: int = 0


@dataclass
class World:
    w: int = 22
    h: int = 14
    tiles: list[list[int]] = field(default_factory=list)
    robot: Robot = field(default_factory=Robot)
    t: float = 0.0
    mission: str = "collect_red"
    status: str = "idle"
    log: list[str] = field(default_factory=list)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h

    def tile(self, x: float, y: float) -> int:
        ix, iy = int(x), int(y)
        if not self.in_bounds(ix, iy):
            return WALL
        return self.tiles[iy][ix]

    def blocked(self, x: float, y: float) -> bool:
        t = self.tile(x, y)
        return t in (WALL, HAZARD, PERSON, BOX_RED, BOX_BLUE, BOX_GREEN, DOCK) and t != DOCK

    def solid(self, t: int) -> bool:
        return t in (WALL, HAZARD, PERSON, BOX_RED, BOX_BLUE, BOX_GREEN)

    def note(self, msg: str) -> None:
        self.log.append(msg)
        if len(self.log) > 40:
            self.log = self.log[-40:]


def make_warehouse(seed: int = 7, mission: str = "collect_red") -> World:
    rng = random.Random(seed)
    w, h = 22, 14
    tiles = [[FLOOR for _ in range(w)] for _ in range(h)]
    for x in range(w):
        tiles[0][x] = WALL
        tiles[h - 1][x] = WALL
    for y in range(h):
        tiles[y][0] = WALL
        tiles[y][w - 1] = WALL
    for y in range(3, h - 3, 4):
        for x in range(3, w - 3):
            if x % 5 != 0:
                tiles[y][x] = WALL
    tiles[h - 2][w - 3] = DOCK
    tiles[h - 2][w - 2] = DOCK
    tiles[2][w - 3] = DOCK

    palette = [BOX_RED, BOX_BLUE, BOX_GREEN]
    placed = 0
    while placed < 10:
        x, y = rng.randint(2, w - 3), rng.randint(2, h - 3)
        if tiles[y][x] == FLOOR:
            tiles[y][x] = palette[placed % 3]
            placed += 1
    for _ in range(4):
        x, y = rng.randint(2, w - 3), rng.randint(2, h - 3)
        if tiles[y][x] == FLOOR:
            tiles[y][x] = HAZARD
    for _ in range(2):
        x, y = rng.randint(2, w - 3), rng.randint(2, h - 3)
        if tiles[y][x] == FLOOR:
            tiles[y][x] = PERSON

    world = World(w=w, h=h, tiles=tiles, mission=mission)
    world.robot = Robot(x=2.5, y=h - 2.5, theta=0.0)
    world.note(f"mission start: {mission}")
    return world


def target_box(mission: str) -> int:
    return {"collect_red": BOX_RED, "collect_blue": BOX_BLUE, "collect_green": BOX_GREEN}.get(
        mission, BOX_RED
    )


def remaining_targets(world: World) -> list[tuple[int, int]]:
    want = target_box(world.mission)
    out = []
    for y in range(world.h):
        for x in range(world.w):
            if world.tiles[y][x] == want:
                out.append((x, y))
    return out


def dock_cells(world: World) -> list[tuple[int, int]]:
    return [(x, y) for y in range(world.h) for x in range(world.w) if world.tiles[y][x] == DOCK]


def try_move(world: World, dt: float) -> None:
    r = world.robot
    r.theta = (r.theta + r.yaw_rate * dt + math.pi) % (2 * math.pi) - math.pi
    nx = r.x + math.cos(r.theta) * r.speed * dt
    ny = r.y + math.sin(r.theta) * r.speed * dt
    if world.solid(world.tile(nx, ny)) or world.solid(world.tile(nx + 0.2 * math.cos(r.theta), ny + 0.2 * math.sin(r.theta))):
        r.collisions += 1
        r.speed = 0.0
        world.status = "collision"
        return
    r.x, r.y = nx, ny
    r.energy_j += (abs(r.speed) * 12.0 + abs(r.yaw_rate) * 4.0) * dt
    world.t += dt


def facing_cell(world: World) -> tuple[int, int]:
    r = world.robot
    return int(r.x + math.cos(r.theta) * 0.85), int(r.y + math.sin(r.theta) * 0.85)


def try_pick(world: World) -> bool:
    r = world.robot
    if r.carrying:
        return False
    fx, fy = facing_cell(world)
    if not world.in_bounds(fx, fy):
        return False
    t = world.tiles[fy][fx]
    if t in BOXES:
        r.carrying = t
        world.tiles[fy][fx] = FLOOR
        world.note(f"picked {t}")
        return True
    return False


def try_place(world: World) -> bool:
    r = world.robot
    if not r.carrying:
        return False
    fx, fy = facing_cell(world)
    if world.in_bounds(fx, fy) and world.tiles[fy][fx] == DOCK:
        r.delivered += 1
        world.note(f"delivered box -> dock ({r.delivered})")
        r.carrying = 0
        return True
    return False


def tile_color(t: int) -> tuple[int, int, int]:
    return CLASS_COLORS.get(t, (20, 20, 20))
