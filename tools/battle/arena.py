"""Two crabs on a plank: the world the duel is fought in.

The toolkit runs one robot per world -- one `robot` entity, one controller, one
pad. This adds the second: a copy of the task's own robot entity under the name
`robot2`, its shell recoloured blue, standing at the far end of a plank facing
the red one, with every event the task gives the red robot cloned for it so a
reset puts both back on their marks. Nothing here reads the task's rewards or
observations, which stay the red robot's: the duel is played through two
`AppPlayer`s (`duel.py`), and the environment is only the physics.

Three things the second robot needs that are invisible in the config:

* **The crabs could not touch.** The robot's collision masks (`constants.py`)
  give each leg a bit and the base none, so two copies of the robot collide
  leg-against-leg only where the leg indices differ, and base against anything
  not at all: a shove would pass through the other crab's shell. `_decorate`
  ORs one bit per crab into every collision geom's `contype` and the *other*
  crab's bit into its `conaffinity`, so every geom of one meets every geom of
  the other and nothing within a crab changes.
* **The world's fall reset would hide the fall.** The family's `fell_over`
  termination teleports a robot upright the step it tips past the limit, and
  `play --app` moves that limit to 80 degrees. A crab shoved off a 0.25 m plank
  tips past that as it lands, so the camera would see the loser vanish mid-air.
  The terminations are moved out of reach and the referee (`referee.py`) decides
  who fell, lets the fall play out, and resets the round itself.
* **An entity with no reset event stays where it was.** mjlab places entities
  in `reset_root_state_uniform`, so `robot2` needs the task's `reset_base` and
  joint reset cloned under its own name (`_second_robots_events`); the clone
  carries the task's own parameters, so the blue crab settles exactly as the
  red one does.

The plank is a static box on the world body -- not a prop entity, since it
never moves -- on two trestles, over whatever ground the scene puts underneath
(`--scene beach` for the sand). Its numbers are environment variables so a take
can be framed without editing this file, the way `tools/fetch/mug.py` places
the mug.
"""
from __future__ import annotations

import copy
import dataclasses
import inspect
import math
import os
from typing import Any

#: The scene's names for the two crabs. The red one is the task's own robot, so
#: everything the task says about `robot` stays true of it.
RED, BLUE = "robot", "robot2"
#: jumper.xml's shell colour (every `rgba="0.85 0.08 0.05 1"` geom is the red
#: shell), and the blue it becomes on the second crab.
SHELL_RED = (0.85, 0.08, 0.05, 1.0)
SHELL_BLUE = (0.10, 0.32, 0.88, 1.0)

ARENA = os.environ.get("BATTLE_ARENA", "disc")                      # disc (Andrew, 2026-10-14: "I prefer that the
                                                                    # platform is circular so the crabs have more
                                                                    # ability to dodge to the side") or plank
DISC_R = float(os.environ.get("BATTLE_DISC_R", "0.8"))              # m: the disc's radius; a crab's stance is ~0.4 m
PLANK_LEN = float(os.environ.get("BATTLE_PLANK_LEN", "2.0"))        # m, along x
PLANK_WIDTH = float(os.environ.get("BATTLE_PLANK_WIDTH", "0.7"))    # m: a crab is ~0.4 m across its stance; at 0.5 the first shove settled it
PLANK_TOP = float(os.environ.get("BATTLE_PLANK_TOP", "0.25"))       # m above the ground: the drop
PLANK_THICK = 0.04
START_GAP = float(os.environ.get("BATTLE_START_GAP", "0.9"))        # m between the two bases at the bell
WOOD = (0.62, 0.45, 0.26, 1.0)
TRESTLE = (0.35, 0.30, 0.25, 1.0)

#: Each crab's depth sensor, as the scene names it (`_tof_for_blue`).
TOF = {RED: "tof", BLUE: "tof2"}

#: One contact bit per crab, above the ground bit (1), the six leg bits (2..7)
#: and the prop channel (`objects.py::PROP_BIT`, 1 << 11).
CROSS_BITS = {RED: 1 << 13, BLUE: 1 << 14}


def _recoloured(spec_fn: Any, old: tuple, new: tuple) -> Any:
    """`spec_fn` with every geom of colour `old` turned `new`."""
    def build():
        spec = spec_fn()
        for geom in spec.geoms:
            if all(abs(float(a) - b) < 0.02 for a, b in zip(geom.rgba, old)):
                geom.rgba = list(new)
        return spec
    return build


def _pin(reset: Any) -> None:
    """Narrow a `reset_base` to its mark: the entity's own `init_state` is the
    place, and the event's x, y and yaw ranges are what would move it off."""
    pose = dict(reset.params.get("pose_range", {}))
    pose.update(x=(0.0, 0.0), y=(0.0, 0.0), yaw=(0.0, 0.0))
    reset.params["pose_range"] = pose


def _second_robots_events(env_cfg: Any) -> list[str]:
    """Clone every startup and reset event the task gives `robot` for `robot2`.

    An event names its robot in `asset_cfg`, or leaves it to the function's
    default, which is `robot`; one whose function takes no `asset_cfg` cannot be
    pointed at the second robot and is skipped, by name, on the terminal.
    """
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    cloned, skipped = [], []
    for name, term in list(env_cfg.events.items()):
        if term.mode not in ("startup", "reset"):
            continue
        asset = (term.params or {}).get("asset_cfg")
        if asset is None:
            if "asset_cfg" not in inspect.signature(term.func).parameters:
                skipped.append(name)
                continue
            asset2 = SceneEntityCfg(BLUE)
        elif asset.name == RED:
            asset2 = dataclasses.replace(asset, name=BLUE)
        else:
            continue
        clone = copy.deepcopy(term)
        clone.params = {**(clone.params or {}), "asset_cfg": asset2}
        env_cfg.events[f"{name}2"] = clone
        cloned.append(name)
    if skipped:
        print(f"[arena] not cloned for the blue crab (no asset_cfg): {', '.join(skipped)}")
    return cloned


def _tof_for_blue(env_cfg: Any, senses: bool) -> bool:
    """The blue crab's dToF beside the red one's, when the task mounts one and a
    pilot looks through it; otherwise neither crab's, since 2 x 2268 rays a step
    are cast for nobody."""
    sensors = list(env_cfg.scene.sensors or ())
    tof = next((s for s in sensors if type(s).__name__ == "ToFSensorCfg" and s.name == "tof"), None)
    if tof is None:
        return False
    if not senses:
        env_cfg.scene.sensors = tuple(s for s in sensors if s is not tof)
        return False
    frame = dataclasses.replace(tof.frame, entity=BLUE)
    pattern = dataclasses.replace(tof.pattern, camera=dataclasses.replace(tof.pattern.camera, entity=BLUE),
                                  body=frame)
    sensors.append(dataclasses.replace(tof, name="tof2", frame=frame, pattern=pattern))
    env_cfg.scene.sensors = tuple(sensors)
    return True


def _decorate(spec: Any) -> None:
    """The scene's last edit: the crabs made to collide, and the plank."""
    import mujoco

    for geom in spec.geoms:
        name = geom.name or ""
        for crab, bit in CROSS_BITS.items():
            if name.startswith(f"{crab}/") and (geom.contype or geom.conaffinity):
                other = next(b for c, b in CROSS_BITS.items() if c != crab)
                geom.contype |= bit
                geom.conaffinity |= other
    box = mujoco.mjtGeom.mjGEOM_BOX
    leg_h = (PLANK_TOP - PLANK_THICK) / 2
    if ARENA == "disc":
        # a round table: the top a cylinder, on one pedestal
        cyl = mujoco.mjtGeom.mjGEOM_CYLINDER
        spec.worldbody.add_geom(name="plank", type=cyl, size=[DISC_R, PLANK_THICK / 2, 0.0],
                                pos=[0.0, 0.0, PLANK_TOP - PLANK_THICK / 2], rgba=list(WOOD),
                                contype=1, conaffinity=1, group=2)
        spec.worldbody.add_geom(name="trestle0", type=cyl, size=[0.18, leg_h, 0.0],
                                pos=[0.0, 0.0, leg_h], rgba=list(TRESTLE), contype=1, conaffinity=1, group=2)
        return
    spec.worldbody.add_geom(name="plank", type=box, size=[PLANK_LEN / 2, PLANK_WIDTH / 2, PLANK_THICK / 2],
                            pos=[0.0, 0.0, PLANK_TOP - PLANK_THICK / 2], rgba=list(WOOD),
                            contype=1, conaffinity=1, group=2)
    for i, x in enumerate((-(PLANK_LEN / 2 - 0.25), PLANK_LEN / 2 - 0.25)):
        spec.worldbody.add_geom(name=f"trestle{i}", type=box, size=[0.04, PLANK_WIDTH / 2 - 0.06, leg_h],
                                pos=[x, 0.0, leg_h], rgba=list(TRESTLE), contype=1, conaffinity=1, group=2)


def margin(x: float, y: float) -> float:
    """How far the point (x, y) is inside the arena's top, in metres (negative outside):
    the disc's radius less the distance from its centre, or the plank's nearer edge."""
    if ARENA == "disc":
        return DISC_R - math.hypot(x, y)
    return min(PLANK_LEN / 2 - abs(x), PLANK_WIDTH / 2 - abs(y))


def apply_arena(env_cfg: Any, *, senses: bool = False) -> None:
    """The second crab, the plank, and a camera on the two of them. `senses`
    keeps the crabs' depth sensors, for a pilot that looks for the other crab."""
    from mjlab.viewer.viewer_config import ViewerConfig

    from scenes.registry import _install_spec_edits
    from tasks.jumper.common.constants import STAND_Z

    entities = env_cfg.scene.entities
    red = entities[RED]
    blue = copy.deepcopy(red)
    blue.spec_fn = _recoloured(blue.spec_fn, SHELL_RED, SHELL_BLUE)
    z = PLANK_TOP + STAND_Z
    red.init_state = dataclasses.replace(red.init_state, pos=(-START_GAP / 2, 0.0, z), rot=(1.0, 0.0, 0.0, 0.0))
    blue.init_state = dataclasses.replace(blue.init_state, pos=(START_GAP / 2, 0.0, z), rot=(0.0, 0.0, 0.0, 1.0))
    entities[BLUE] = blue

    cloned = _second_robots_events(env_cfg)
    if "reset_base" not in env_cfg.events:
        raise ValueError("the arena pins the spawns, and this config has no 'reset_base'")
    _pin(env_cfg.events["reset_base"])
    _pin(env_cfg.events["reset_base2"])

    # The referee's, not the world's: see the module docstring.
    moved = []
    for name, term in env_cfg.terminations.items():
        if "limit_angle" in (term.params or {}):
            term.params = {**term.params, "limit_angle": math.pi}
            moved.append(name)

    tof = _tof_for_blue(env_cfg, senses)
    _install_spec_edits(env_cfg.scene, None, _decorate)

    # Two hexapods and a plank: more contacts than one robot on a plane.
    env_cfg.sim.nconmax = max(int(env_cfg.sim.nconmax or 0), 512)
    env_cfg.sim.njmax = max(int(env_cfg.sim.njmax or 0), 4096)

    # A camera on the plank rather than on one crab: the recorder and the live
    # viewer both keep a `WORLD` origin as the config gives it.
    viewer = env_cfg.viewer
    viewer.origin_type = ViewerConfig.OriginType.WORLD
    viewer.lookat = (float(os.environ.get("BATTLE_CAM_X", "0.0")), 0.0, PLANK_TOP + 0.05)   # along the plank
    viewer.distance = float(os.environ.get("BATTLE_CAM_DIST", "2.3"))
    viewer.azimuth = float(os.environ.get("BATTLE_CAM_AZ", "90"))
    viewer.elevation = float(os.environ.get("BATTLE_CAM_EL", "-12"))

    print(f"[arena] plank {PLANK_LEN:g} x {PLANK_WIDTH:g} m, top at {PLANK_TOP:g} m; red at x {-START_GAP / 2:+.2f} "
          f"facing +x, blue at {START_GAP / 2:+.2f} facing -x")
    print(f"[arena] cloned for the blue crab: {', '.join(cloned) or 'nothing'}"
          + ("; its dToF is `tof2`" if tof else "; no depth sensors (no pilot looks)")
          + (f"; fall resets left to the referee: {', '.join(moved)}" if moved else ""))
