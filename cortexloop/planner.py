from __future__ import annotations

import heapq
import math

from .world import (
    BOXES,
    DOCK,
    FLOOR,
    HAZARD,
    PERSON,
    WALL,
    World,
    dock_cells,
    facing_cell,
    remaining_targets,
    target_box,
    try_pick,
    try_place,
)


def _passable(world: World, x: int, y: int, ignore_boxes: bool) -> bool:
    if not world.in_bounds(x, y):
        return False
    t = world.tiles[y][x]
    if t in (WALL, HAZARD, PERSON):
        return False
    if t in BOXES and not ignore_boxes:
        return False
    return True


def astar(world: World, start: tuple[int, int], goal: tuple[int, int], ignore_boxes: bool = False):
    if start == goal:
        return [start]
    w, h = world.w, world.h

    def hcost(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    openh = [(hcost(start, goal), 0, start)]
    came = {start: None}
    g = {start: 0}
    while openh:
        _, cost, cur = heapq.heappop(openh)
        if cur == goal:
            path = [cur]
            while came[cur] is not None:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path
        x, y = cur
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if not _passable(world, nx, ny, ignore_boxes) and (nx, ny) != goal:
                continue
            ng = cost + 1
            if ng < g.get((nx, ny), 1e9):
                g[(nx, ny)] = ng
                came[(nx, ny)] = cur
                heapq.heappush(openh, (ng + hcost((nx, ny), goal), ng, (nx, ny)))
    return []


def _steer_toward(world: World, tx: float, ty: float) -> None:
    r = world.robot
    ang = math.atan2(ty - r.y, tx - r.x)
    err = (ang - r.theta + math.pi) % (2 * math.pi) - math.pi
    r.yaw_rate = max(-2.6, min(2.6, 3.4 * err))
    if abs(err) < 0.35:
        r.speed = 1.55
    elif abs(err) < 0.8:
        r.speed = 0.55
    else:
        r.speed = 0.0


def plan(world: World, perception: dict) -> str:
    """Sense → decide → act. Perception can veto motion when a hazard is in view."""
    r = world.robot
    cls = perception.get("class_name", "empty")
    conf = float(perception.get("confidence", 0.0))

    if cls in ("hazard", "person") and conf > 0.45:
        r.speed = 0.0
        r.yaw_rate = 1.8
        world.status = f"avoid:{cls}"
        return world.status

    if r.carrying:
        if try_place(world):
            r.speed = 0.0
            r.yaw_rate = 0.0
            world.status = "placed"
            return world.status
        docks = dock_cells(world)
        if docks:
            path = astar(world, (int(r.x), int(r.y)), docks[0], ignore_boxes=True)
            if len(path) >= 2:
                _steer_toward(world, path[1][0] + 0.5, path[1][1] + 0.5)
                world.status = "haul_to_dock"
                return world.status
        world.status = "seek_dock"
        return world.status

    if cls in ("box_red", "box_blue", "box_green") and conf > 0.4:
        want = target_box(world.mission)
        names = {2: "box_red", 3: "box_blue", 4: "box_green"}
        if names.get(want) == cls:
            if try_pick(world):
                r.speed = 0.0
                world.status = "picked"
                return world.status
            r.speed = 0.7
            r.yaw_rate = 0.0
            world.status = "approach_box"
            return world.status

    targets = remaining_targets(world)
    if not targets:
        r.speed = 0.0
        r.yaw_rate = 0.0
        world.status = "mission_complete"
        return world.status

    goal = min(targets, key=lambda p: abs(p[0] - r.x) + abs(p[1] - r.y))
    path = astar(world, (int(r.x), int(r.y)), goal, ignore_boxes=False)
    if len(path) >= 2:
        _steer_toward(world, path[1][0] + 0.5, path[1][1] + 0.5)
        world.status = "navigate"
        return world.status
    # blocked: rotate to replan
    r.speed = 0.0
    r.yaw_rate = 1.2
    world.status = "replan"
    return world.status
