"""An occupancy map of the room from the depth camera, as a lidar: each frame's object
returns mark cells occupied and the rays to them (and to the floor) mark cells free,
in a world frame anchored where the crab woke up. The heading is the gyro's (integrated
about the world's up); the position is dead-reckoned from the pad's sticks and pulled
back onto the map each frame by matching the scan against what is already there
(a small search over offsets, the heading trusted). The map answers what the rules
could not: is the way behind clear before backing, has this pillar been passed, and
which way round the obstacles to a goal (A* on the inflated grid).

Measured on the room walk (w72, 2026-10-14): see `step`'s docstring for the speed
model the dead reckoning starts from; the scan match takes it the rest of the way.
"""
from __future__ import annotations

import heapq
import math

import numpy as np
from sense import COLS, FLOOR_CLEAR, ROWS, TOF_ORIGIN

RES = 0.05                 # m per cell
SIZE = 14.0                # m across; the wake pose sits at the centre
N = int(round(SIZE / RES))
HIT, MISS = 0.9, 0.25      # log-odds steps per frame
CLAMP = 6.0
OCC, FREE = 1.0, -1.0      # thresholds on the log-odds
MAP_RANGE = 3.0            # returns past this are not mapped (the picture's edge is noisy)
MAP_NEAR = 0.45            # nor inside this: the carried mug hangs 0.25-0.35 m ahead and, mapped, it
                           # smeared a trail of "obstacles" that stopped the retreat 9 cm behind (w82)
MAP_SELF = 0.35            # cells within this of the base are freed every step: the body stands there, and
                           # the specks the legs and the swinging mug left round the crab at the tray read
                           # as "0.30 m behind" and stopped the retreat (film58)
INFLATE = 0.33             # m: half the body plus clearance, for planning and corridors (0.28 -> 0.33 with the
                           # walk allowed 17 deg off the path, film42)
SPEED_FWD = 0.42           # m/s per unit of walk stick (0.5 -> 0.21 measured, CREEP_SPEED)
SPEED_SIDE = 0.35          # m/s per unit of strafe stick (0.5 -> ~0.17 m/s, the sidestep tests)
DEAD_BAND = 0.40           # a stick under this moves nothing
MATCH_WIN, MATCH_STEP = 0.15, 0.05   # the scan match's search, m
MATCH_GAIN = 0.5           # how much of the best offset is applied per frame


class Mapper:
    def __init__(self):
        self.grid = np.zeros((N, N), dtype=np.float32)      # log-odds, row = y cell, col = x cell
        self.x, self.y = 0.0, 0.0                            # the base in the wake frame
        self.yaw = 0.0
        self.last_t = None
        self.last_frame_t = None
        self.marks = {}                                      # named world points (the flag, the mug)
        self.path = []                                       # the last plan, world xy

    # -- frames -------------------------------------------------------------------
    def cell(self, wx, wy):
        return int(round(wy / RES)) + N // 2, int(round(wx / RES)) + N // 2

    def world(self, r, c):
        return (c - N // 2) * RES, (r - N // 2) * RES

    def to_world(self, pts_b):
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        x = self.x + cy * pts_b[:, 0] - sy * pts_b[:, 1]
        y = self.y + sy * pts_b[:, 0] + cy * pts_b[:, 1]
        return x, y

    def to_body(self, wx, wy):
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = wx - self.x, wy - self.y
        return cy * dx + sy * dy, -sy * dx + cy * dy

    # -- the step ----------------------------------------------------------------
    def step(self, senses, t, ly, lx):
        """Once per control step. `ly`/`lx` are the pad's sticks (ly < 0 walks forward,
        lx > 0 strafes right). Dead-reckons the pose, then on a fresh exposure matches
        the scan to the map and folds the frame in."""
        if self.last_t is not None:
            dt = t - self.last_t
            v = -SPEED_FWD * ly if abs(ly) >= DEAD_BAND else 0.0
            vy = -SPEED_SIDE * lx if abs(lx) >= DEAD_BAND else 0.0
            cy, sy = math.cos(self.yaw), math.sin(self.yaw)
            self.x += (cy * v - sy * vy) * dt
            self.y += (sy * v + cy * vy) * dt
        self.last_t = t
        self.yaw = senses.yaw
        # the body's own cells, freed every step (before the frame test: a step without a
        # fresh exposure still moves the base)
        k = int(math.ceil(MAP_SELF / RES))
        r0, c0 = self.cell(self.x, self.y)
        rr, cc = np.ogrid[-k:k + 1, -k:k + 1]
        disk = (rr * rr + cc * cc) * RES * RES <= MAP_SELF * MAP_SELF
        rs, cs = slice(max(0, r0 - k), min(N, r0 + k + 1)), slice(max(0, c0 - k), min(N, c0 + k + 1))
        sub = self.grid[rs, cs]
        d = disk[k - (r0 - rs.start): k + (rs.stop - r0), k - (c0 - cs.start): k + (cs.stop - c0)]
        sub[d] = np.minimum(sub[d], FREE - 0.5)
        self.yaw = senses.yaw
        f = senses.last
        if f is None or f["t"] == self.last_frame_t:
            return
        self.last_frame_t = f["t"]
        r = f["range"].ravel()
        pts, h = f["pts"], f["h"]
        valid = (r > 0) & ~f["self"].ravel() & (pts[:, 0] > 0.10) & (r < MAP_RANGE) & (r > MAP_NEAR)
        obj = valid & (h > FLOOR_CLEAR)
        floor = valid & (h <= FLOOR_CLEAR)
        # the scan match: object points against the map's occupied cells
        if obj.sum() >= 20 and (self.grid > OCC).sum() >= 40:
            ox, oy = self.to_world(pts[obj])
            best, best_s = (0.0, 0.0), -1
            base = None
            for dx in np.arange(-MATCH_WIN, MATCH_WIN + 1e-6, MATCH_STEP):
                for dy in np.arange(-MATCH_WIN, MATCH_WIN + 1e-6, MATCH_STEP):
                    rr = np.clip(np.round((oy + dy) / RES).astype(int) + N // 2, 0, N - 1)
                    cc = np.clip(np.round((ox + dx) / RES).astype(int) + N // 2, 0, N - 1)
                    sc = int((self.grid[rr, cc] > OCC).sum())
                    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                        base = sc
                    if sc > best_s or (sc == best_s and dx * dx + dy * dy < best[0] ** 2 + best[1] ** 2):
                        best, best_s = (float(dx), float(dy)), sc
            if base is not None and best_s >= base + max(4, 0.15 * obj.sum()):
                self.x += MATCH_GAIN * best[0]
                self.y += MATCH_GAIN * best[1]
        # fold the frame in: free rays to each column's nearest object (or farthest floor), hits at the objects
        ex, ey = self.to_world(TOF_ORIGIN[None, :2])
        ex, ey = float(ex[0]), float(ey[0])
        x_img = np.where(obj, pts[:, 0], np.inf).reshape(ROWS, COLS)
        near_i = x_img.argmin(axis=0)
        near_x = x_img[near_i, np.arange(COLS)]
        fl_img = np.where(floor, pts[:, 0], -np.inf).reshape(ROWS, COLS)
        far_i = fl_img.argmax(axis=0)
        far_x = fl_img[far_i, np.arange(COLS)]
        ends = []
        for c in range(COLS):
            if np.isfinite(near_x[c]):
                ends.append(near_i[c] * COLS + c)
            elif np.isfinite(far_x[c]):
                ends.append(far_i[c] * COLS + c)
        if ends:
            wx, wy = self.to_world(pts[ends])
            for px, py in zip(wx, wy):
                n = max(2, int(math.hypot(px - ex, py - ey) / (RES * 0.7)))
                xs = np.linspace(ex, px, n, endpoint=False)[1:]
                ys = np.linspace(ey, py, n, endpoint=False)[1:]
                rr = np.clip(np.round(ys / RES).astype(int) + N // 2, 0, N - 1)
                cc = np.clip(np.round(xs / RES).astype(int) + N // 2, 0, N - 1)
                self.grid[rr, cc] -= MISS
        if obj.any():
            ox, oy = self.to_world(pts[obj])
            rr = np.clip(np.round(oy / RES).astype(int) + N // 2, 0, N - 1)
            cc = np.clip(np.round(ox / RES).astype(int) + N // 2, 0, N - 1)
            np.add.at(self.grid, (rr, cc), HIT)
        np.clip(self.grid, -CLAMP, CLAMP, out=self.grid)

    # -- queries -----------------------------------------------------------------
    def mark(self, name, bx, by):
        """Remember a point seen in the body frame under a name, in world coordinates."""
        wx, wy = self.to_world(np.array([[bx, by]]))
        self.marks[name] = (float(wx[0]), float(wy[0]))

    def mark_body(self, name):
        """A remembered point, in the body frame now (or None)."""
        if name not in self.marks:
            return None
        return self.to_body(*self.marks[name])

    def occupied_cells(self):
        return self.grid > OCC

    def inflated(self):
        from scipy import ndimage
        k = int(math.ceil(INFLATE / RES))
        yy, xx = np.ogrid[-k:k + 1, -k:k + 1]
        disk = (xx * xx + yy * yy) * RES * RES <= INFLATE * INFLATE
        return ndimage.binary_dilation(self.grid > OCC, structure=disk)

    def clearance(self, heading_b, length, half_width=0.25):
        """The nearest occupied cell inside a corridor from the base along the body-frame
        bearing `heading_b` (0 ahead, pi behind), in metres, or `length` if clear."""
        occ = self.grid > OCC
        rr, cc = np.nonzero(occ)
        if len(rr) == 0:
            return length
        wx, wy = self.world(rr, cc)
        bx, by = self.to_body(wx, wy)
        ch, sh = math.cos(heading_b), math.sin(heading_b)
        along = bx * ch + by * sh
        across = -bx * sh + by * ch
        sel = (along > 0.0) & (along < length) & (np.abs(across) < half_width)
        return float(along[sel].min()) if sel.any() else length

    def plan(self, goal_w, inflate=None):
        """A* over the inflated grid from the base to `goal_w` (world xy); unknown cells
        count as free. Returns the path as world xy (empty when unreachable)."""
        blocked = self.inflated() if inflate is None else inflate
        sr, sc = self.cell(self.x, self.y)
        gr, gc = self.cell(*goal_w)
        sr, sc, gr, gc = (min(max(v, 0), N - 1) for v in (sr, sc, gr, gc))
        blocked[sr, sc] = False
        if blocked[gr, gc]:
            # the goal inside an obstacle's inflation (the flag's pole): take the nearest free cell
            rr, cc = np.nonzero(~blocked)
            k = int(np.argmin((rr - gr) ** 2 + (cc - gc) ** 2))
            gr, gc = int(rr[k]), int(cc[k])
        came = {}
        g = {(sr, sc): 0.0}
        pq = [(0.0, (sr, sc))]
        steps = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))
        found = False
        n_pop = 0
        while pq and n_pop < 60000:
            _, cur = heapq.heappop(pq)
            n_pop += 1
            if cur == (gr, gc):
                found = True
                break
            for dr, dc, w in steps:
                nr, nc = cur[0] + dr, cur[1] + dc
                if not (0 <= nr < N and 0 <= nc < N) or blocked[nr, nc]:
                    continue
                ng = g[cur] + w
                if ng < g.get((nr, nc), 1e9):
                    g[(nr, nc)] = ng
                    came[(nr, nc)] = cur
                    heapq.heappush(pq, (ng + math.hypot(nr - gr, nc - gc), (nr, nc)))
        if not found:
            self.path = []
            return []
        cells = [(gr, gc)]
        while cells[-1] != (sr, sc):
            cells.append(came[cells[-1]])
        cells.reverse()
        self.path = [self.world(r, c) for r, c in cells]
        return self.path

    def next_point(self, goal_w, lookahead=0.5):
        """Plan to `goal_w` and return the world point about `lookahead` metres along the
        path (None when no path). Planning costs ~20 ms; call it a few times a second
        and steer at the returned point in between (`bearing_to`)."""
        path = self.plan(goal_w)
        if not path:
            return None
        d = 0.0
        tx, ty = path[-1]
        for (x0, y0), (x1, y1) in zip(path, path[1:]):
            d += math.hypot(x1 - x0, y1 - y0)
            if d >= lookahead:
                tx, ty = x1, y1
                break
        return tx, ty

    def bearing_to(self, wx, wy):
        bx, by = self.to_body(wx, wy)
        return math.atan2(by, bx)

    def next_bearing(self, goal_w, lookahead=0.5):
        pt = self.next_point(goal_w, lookahead)
        return None if pt is None else self.bearing_to(*pt)

    # -- pictures ----------------------------------------------------------------
    def picture(self, path, span=4.0):
        """A PNG of the map around the base: dark = occupied, light = free, grey = unknown,
        the base as a red arrow, marks in yellow, the plan in cyan."""
        from PIL import Image, ImageDraw
        k = int(span / RES)
        r0, c0 = self.cell(self.x, self.y)
        r0, c0 = min(max(r0, k), N - k - 1), min(max(c0, k), N - k - 1)
        sub = self.grid[r0 - k: r0 + k, c0 - k: c0 + k]
        img = np.full(sub.shape + (3,), 128, dtype=np.uint8)
        img[sub > OCC] = (30, 30, 30)
        img[sub < FREE] = (220, 215, 200)
        im = Image.fromarray(img[::-1]).resize((sub.shape[1] * 3, sub.shape[0] * 3), Image.NEAREST)
        dr = ImageDraw.Draw(im)
        def px(wx, wy):
            r, c = self.cell(wx, wy)
            return (c - (c0 - k)) * 3 + 1, (2 * k - 1 - (r - (r0 - k))) * 3 + 1
        if len(self.path) > 1:
            dr.line([px(*p) for p in self.path], fill=(0, 200, 220), width=2)
        for name, (wx, wy) in self.marks.items():
            x, y = px(wx, wy)
            dr.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(255, 220, 0), width=2)
            dr.text((x + 6, y - 6), name, fill=(255, 220, 0))
        x, y = px(self.x, self.y)
        hx, hy = px(self.x + 0.3 * math.cos(self.yaw), self.y + 0.3 * math.sin(self.yaw))
        dr.line([(x, y), (hx, hy)], fill=(230, 40, 40), width=3)
        dr.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(230, 40, 40))
        im.save(path)
