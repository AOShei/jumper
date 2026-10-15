"""One crab's view of the room, from its own senses: the objects on the floor.

`tools/fetch/sense.py` reads the dToF range image, the IMU and the encoders and
finds the mug in them; `tools/battle/eyes.py` is the same reading for a crab
looking for a crab. This is the reading for a crab looking for things on the
floor: every cluster of returns that is not the crab itself and not the floor,
gated to the sizes of the race's objects (`room.KINDS`: 5-12 cm across, 5-11 cm
tall), so the other crab (0.3-0.5 m across) and the kerb (wall to wall) are not
objects, with each one's position in the base frame. Everything else -- the
pixel-to-ray map, the self-mask, the floor plane along the IMU's up, the
depth-connected clustering -- is copied from those two, for the reasons their
docstrings give. The frame it keeps (`last`) is what the readout draws and what
`mapper.Mapper.step` folds into the map.
"""
from __future__ import annotations

import copy
import math

import mujoco
import numpy as np
from arena import TOF
from scipy import ndimage

ROWS, COLS = 42, 54
HALF_X, HALF_Y = math.tan(math.radians(27.5)), math.tan(math.radians(21.0))
TOF_ORIGIN = np.array([0.090678, 0.0, 0.0168455])
TOF_EYE = TOF_ORIGIN + np.array([0.004, 0.0, 0.0])
FOOT_R = 0.012
FLOOR_CLEAR = 0.03
MAX_RANGE = 3.0
OBJECT_HEIGHT = (0.035, 0.16)     # m above the floor, the cluster's tallest point
OBJECT_WIDTH = (0.03, 0.26)       # m across (a crab is 0.3-0.5, the kerb is the picture's width)
CRAB_SKIRT = 0.45                 # m round a crab-sized cluster: the small clusters there are its legs
CRAB_HEIGHT = (0.12, 0.27)        # a crab's shell is 0.15-0.20 up; the kerb starts where a crab ends
KERB_MIN = 0.27                   # the kerb is 0.35 m tall: nothing else in the room reaches this


def ray_dirs() -> np.ndarray:
    u = ((np.arange(COLS) + 0.5) / COLS * 2 - 1) * HALF_X
    v = (1 - (np.arange(ROWS) + 0.5) / ROWS * 2) * HALF_Y
    uu, vv = np.meshgrid(u, v, indexing="xy")
    d = np.stack([np.ones_like(uu), -uu, vv], axis=-1).reshape(-1, 3)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def is_crab(cluster: dict) -> bool:
    """A cluster the size of the other crab: between the objects and the kerb in height, wide."""
    return CRAB_HEIGHT[0] < cluster["height"] < CRAB_HEIGHT[1] and cluster["width"] > 0.2


def _named(model, objtype, name: str, entity: str) -> int:
    i = mujoco.mj_name2id(model, objtype, f"{entity}/{name}")
    if i < 0:
        raise KeyError(f"{entity}/{name} is not in the model")
    return i


class Sight:
    def __init__(self, env, entity: str, yaw0: float = 0.0):
        self.env = env
        self.entity = entity
        self.robot = env.scene[entity]
        self.tof = env.scene[TOF[entity]]
        self.gyro = env.scene[f"{entity}/imu_ang_vel"]
        self.upvec = env.scene[f"{entity}/imu_upvector"]
        self.dt = float(env.step_dt)
        self.m = copy.copy(env.sim.mj_model)
        self.d = mujoco.MjData(self.m)
        base = _named(self.m, mujoco.mjtObj.mjOBJ_BODY, "base_link", entity)
        root = int(self.m.body_rootid[base])
        self.own_bodies = np.flatnonzero(self.m.body_rootid == root)
        own = np.isin(self.m.geom_bodyid, self.own_bodies)
        self.m.geom_group[~own] = 5
        self.qadr = np.array([self.m.jnt_qposadr[_named(self.m, mujoco.mjtObj.mjOBJ_JOINT, n, entity)]
                              for n in self.robot.joint_names])
        self.tof_body = _named(self.m, mujoco.mjtObj.mjOBJ_BODY, "tof_sensor_link", entity)
        self.dirs = ray_dirs()
        self._geomid = np.zeros(ROWS * COLS, dtype=np.int32)
        self._dist = np.zeros(ROWS * COLS, dtype=np.float64)
        self._group = np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8)
        self.yaw = yaw0                    # gyro-integrated from the heading the crab woke up at
        self.frames = 0
        self.last = None

    def _pose_self(self) -> None:
        q = self.robot.data.joint_pos_biased[0].cpu().numpy()
        self.d.qpos[:] = 0.0
        for j in range(self.m.njnt):
            if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.d.qpos[self.m.jnt_qposadr[j] + 3] = 1.0
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)

    def up(self) -> np.ndarray:
        u = self.upvec.data[0].cpu().numpy().astype(np.float64)
        return u / max(1e-6, np.linalg.norm(u))

    def _self_mask(self) -> np.ndarray:
        mujoco.mj_multiRay(self.m, self.d, TOF_EYE, self.dirs.ravel(), self._group, 1,
                           self.tof_body, self._geomid, self._dist, None, ROWS * COLS, mujoco.mjMAXVAL)
        hit = (self._geomid >= 0).reshape(ROWS, COLS)
        return ndimage.binary_dilation(hit, iterations=1).ravel()

    @staticmethod
    def _label(obj: np.ndarray, pts: np.ndarray, step: float = 0.10):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        idx = np.arange(ROWS * COLS).reshape(ROWS, COLS)
        x = pts[:, 0].reshape(ROWS, COLS)
        rows, cols = [], []
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = idx[max(0, -dr):ROWS - max(0, dr), max(0, -dc):COLS - max(0, dc)]
            b = idx[max(0, dr):ROWS - max(0, -dr), max(0, dc):COLS - max(0, -dc)]
            ok = obj.ravel()[a] & obj.ravel()[b] & (np.abs(x.ravel()[a] - x.ravel()[b]) < step)
            rows.append(a[ok]); cols.append(b[ok])
        r_, c_ = np.concatenate(rows), np.concatenate(cols)
        g = coo_matrix((np.ones(len(r_), dtype=np.int8), (r_, c_)), shape=(ROWS * COLS, ROWS * COLS))
        _, comp = connected_components(g, directed=False)
        labels = np.where(obj.ravel(), comp + 1, 0)
        uniq = np.unique(labels[labels > 0])
        remap = np.zeros(labels.max() + 1, dtype=np.int64)
        remap[uniq] = np.arange(1, len(uniq) + 1)
        return remap[labels].reshape(ROWS, COLS), len(uniq)

    def update(self, t: float) -> bool:
        """Once per control step: integrate the gyro; on a fresh exposure, read the
        image. Returns whether a frame was read."""
        g = self.gyro.data[0].cpu().numpy().astype(np.float64)
        self.yaw += float(np.dot(g[:3], self.up())) * self.dt
        if not bool(self.tof.data.fresh[0]):
            return False
        r = self.tof.data.range[0].cpu().numpy().astype(np.float64)
        self._pose_self()
        up = self.up()
        floor = float((self.d.xpos[self.own_bodies] @ up).min()) - FOOT_R
        self_mask = self._self_mask()
        valid = r > 0
        pts = TOF_ORIGIN + r[:, None] * self.dirs
        h = pts @ up - floor
        obj = valid & ~self_mask & (h > FLOOR_CLEAR) & (r < MAX_RANGE) & (pts[:, 0] > 0.10)
        self.frames += 1
        things, big, kerb, masks = self._things(obj.reshape(ROWS, COLS), pts, h)
        self.last = {"t": t, "range": r.reshape(ROWS, COLS), "self": self_mask.reshape(ROWS, COLS),
                     "obj": obj.reshape(ROWS, COLS), "floor": floor, "pts": pts, "h": h,
                     "things": things, "big": big, "kerb": kerb,
                     # for the readout: which zones are things, the kerb, the other crab
                     "thing_px": masks[0].reshape(ROWS, COLS), "kerb_px": masks[1].reshape(ROWS, COLS),
                     "crab_px": masks[2].reshape(ROWS, COLS)}
        return True

    def _things(self, obj: np.ndarray, pts: np.ndarray, h: np.ndarray):
        """The clusters that are object-sized, each with its position in the base
        frame (x ahead, y left: the nearest face's median), and the ones that are not
        (the other crab, the kerb), which the map still wants as obstacles."""
        labels, n = self._label(obj, pts)
        things, big, kerb = [], [], []
        thing_px, kerb_px, crab_px = (np.zeros(ROWS * COLS, dtype=bool) for _ in range(3))
        for k in range(1, n + 1):
            sel = (labels == k).ravel()
            if sel.sum() < 3:
                continue
            p = pts[sel]
            top, width = float(h[sel].max()), float(np.ptp(p[:, 1]))
            cols_ = np.nonzero(sel.reshape(ROWS, COLS).any(axis=0))[0]
            edge = bool(cols_.min() == 0 or cols_.max() == COLS - 1)
            near = float(p[:, 0].min())
            cx, cy = float(np.median(p[:, 0])), float(np.median(p[:, 1]))
            d = {"x": cx, "y": cy, "near": near, "height": top, "width": width, "n": int(sel.sum()), "edge": edge,
                 "bearing": float(math.atan2(cy, cx)), "range": float(math.hypot(cx, cy))}
            if OBJECT_HEIGHT[0] <= top <= OBJECT_HEIGHT[1] and OBJECT_WIDTH[0] <= width <= OBJECT_WIDTH[1] and not edge:
                things.append(d)
                d["sel"] = sel
            else:
                big.append(d)
                # the kerb: taller than anything else (at 10 cm it was told from a cube by width
                # alone, and a cube cut by the picture's edge taken for kerb pulled the pose off, sense6)
                if top >= KERB_MIN and sel.sum() >= 15:
                    kerb.append(np.flatnonzero(sel))
                    kerb_px |= sel
                elif is_crab(d):
                    crab_px |= sel
        # The other crab's legs come out as small clusters beside its big one (probe_sight,
        # 2026-10-15: a crab at 0.26 m is one cluster of 1371 zones, 0.20 m tall, with
        # leg pieces of 3-5 zones round it), so a thing within `CRAB_SKIRT` of a crab-sized
        # cluster is part of the crab.
        crabs = [b for b in big if is_crab(b)]
        kept = []
        for d in things:
            sel = d.pop("sel")
            if all(math.hypot(d["x"] - c["x"], d["y"] - c["y"]) > CRAB_SKIRT for c in crabs):
                kept.append(d)
                thing_px |= sel
            else:
                crab_px |= sel                                    # a leg
        return kept, big, (np.concatenate(kerb) if kerb else np.zeros(0, dtype=int)), (thing_px, kerb_px, crab_px)
