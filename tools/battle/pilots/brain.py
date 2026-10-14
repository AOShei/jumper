"""The brain: a crab that decides, by a weighted roll, what to do next.

Two chargers (`charge.py`) fight the same fight every time, give or take the
simulator's rounding; a match is only a match if the crabs could have done
otherwise. This pilot keeps a small repertoire of **intents** and rolls for one
at every decision point -- the bell, the end of the last intent, the foe coming
close -- from weights set by the crab's **aggression**, 0 to 1: at 1 it is
`charge.py` with a swagger, at 0 it braces, dodges and backs off and charges
only when the foe is already beside it. The roll is a seeded generator of its
own per crab, so `--seed` replays a match move for move and a different seed is
a different match; every decision is printed with the roll that made it.

The repertoire, all in the posture mode (left stick walks, the right stick's
second half turns, R3 with the stick pulled lowers the stance -- the numbers
measured for the fetch, see `charge.py`):

* **charge** -- face the foe and walk into it.
* **brace** -- drop to the lowest stance, face the foe and wait; when it comes
  within `NEAR_M`, shove forward for `SHOVE_S` from the low stance.
* **dodge** -- face the foe and wait for it to come within `DODGE_M`, then
  strafe to one side for `DODGE_S`, so a charger runs past toward the edge, and
  turn on its flank. The side is rolled when both have plank under them and
  is the safer one when not (a dodge off the edge lost the first round).
* **back_off** -- walk backward for `BACK_S`, facing the foe, unless the edge
  is already within `EDGE_M`.
* **circle** -- strafe for `CIRCLE_S` while turning to keep the foe ahead.

Every strafe and the back-off stop at the plank's edge (`EDGE_M` of plank left
in the direction of travel, checked every step): on the 0.7 m plank a circle
at full stick from the mark walked straight off the side in 0.7 s.

`BRAIN_MODE=claw_left` fights the whole round in the claw mode instead: the
crab enters it at the bell, carries its arm straight out, and snaps the claw
at a foe within `SNAP_M`. The probes of 2026-10-14 (`README.md`, "What a crab
can do to another") found the claw decides nothing -- the mouth shuts on air
and the shove is a third of the posture scrum's -- so this is for the look of
two crabs fighting with their claws out; the brace has no crouch there (R3
is the roll shift in a claw mode).

The foe's bearing and range are read from the world, as `charge.py` reads
them; `brawl.py` is the one that looks through the depth sensor, and the two
could be joined once the fight is worth filming. The set pieces -- the taunt
at the bell, the cheer, the victory dance -- are the referee's cues, played
through `moves.Performer`.
"""
from __future__ import annotations

import math
import os

import numpy as np

TURN_DEG = float(os.environ.get("BRAIN_TURN_DEG", "12"))
TURN_STICK = float(os.environ.get("BRAIN_TURN_STICK", "0.62"))
WALK = float(os.environ.get("BRAIN_WALK", "1.0"))
NEAR_M = float(os.environ.get("BRAIN_NEAR_M", "0.42"))         # base to base: the shells about to touch
DODGE_M = float(os.environ.get("BRAIN_DODGE_M", "0.55"))
DODGE_S = float(os.environ.get("BRAIN_DODGE_S", "0.8"))
SHOVE_S = float(os.environ.get("BRAIN_SHOVE_S", "1.0"))
BACK_S = float(os.environ.get("BRAIN_BACK_S", "0.8"))
CIRCLE_S = float(os.environ.get("BRAIN_CIRCLE_S", "1.6"))
CIRCLE_STICK = float(os.environ.get("BRAIN_CIRCLE_STICK", "0.6"))
CROUCH_S = float(os.environ.get("BRAIN_CROUCH_S", "0.7"))      # R3 + stick pulled: 0.7 s reaches the lowest stance
WAIT_S = float(os.environ.get("BRAIN_WAIT_S", "4.0"))          # a brace or a dodge nobody comes to
RETHINK_S = float(os.environ.get("BRAIN_RETHINK_S", "2.5"))    # a charge rolls again this often
EDGE_M = float(os.environ.get("BRAIN_EDGE_M", "0.18"))         # no backing off nearer the edge than this
MIDDLE_M = float(os.environ.get("BRAIN_MIDDLE_M", "0.2"))       # near enough the plank's centre for a set piece
TAUNT_CUT_S = float(os.environ.get("BRAIN_TAUNT_CUT_S", "7.0"))
VICTORY_CUT_S = float(os.environ.get("BRAIN_VICTORY_CUT_S", "12.0"))

MODE = os.environ.get("BRAIN_MODE", "")                        # `claw_left`: fight in the claw mode, arm out, snapping
SNAP_M = float(os.environ.get("BRAIN_SNAP_M", "0.45"))         # the claw snaps at a foe this near
SNAP_S = float(os.environ.get("BRAIN_SNAP_S", "0.5"))          # one snap: shut for half of it, open for half
PRESS_S = 0.3

INTENTS = ("charge", "brace", "dodge", "back_off", "circle", "counter", "wedge", "strike", "dash")

#: The playbooks (Andrew, 2026-10-14: "each crab running a different strategy...
#: crabs should be defensive too... some sport and strategy"). A style is an
#: aggression, the mode it fights in, and the signature intents only it rolls:
#:   bull     -- the charger; it also hops at a foe in range (`dash`), and with
#:               BRAIN_RED_MODE=sprint it fights the round in the sprint mode
#:   matador  -- waits, sidesteps the charge at the last moment, then shoves the
#:               charger on from its flank toward the rim (`counter`)
#:   wrestler -- goes round to the foe's side, crouches, drives its nose under the
#:               trunk and heaves (`wedge`: the lift probe's moves)
#:   boxer    -- the claw mode: circles and swings the arm (`strike`), shoves when
#:               the foe commits
STYLES = {
    "bull": {"aggression": 0.9, "mode": "", "extra": {"dash": 0.8}},     # (the dash only where the bundle has it)
    "matador": {"aggression": 0.2, "mode": "", "extra": {"counter": 0.7}},
    "wrestler": {"aggression": 0.6, "mode": "", "extra": {"wedge": 0.6}},
    "boxer": {"aggression": 0.5, "mode": "claw_left", "extra": {"strike": 0.5}},
}
WEDGE_M = float(os.environ.get("BRAIN_WEDGE_M", "0.17"))        # the wedge drives in to this base-to-base range
FLANK_M = float(os.environ.get("BRAIN_FLANK_M", "0.45"))        # ...from a point this far out on the foe's side
STRIKE_S = float(os.environ.get("BRAIN_STRIKE_S", "0.5"))       # the arm swings out and back this often
ARC_R = float(os.environ.get("BRAIN_ARC_R", "0.45"))            # a dodge arcs round the table's centre at this radius
                                                               # at most (Andrew: "dodges that allow the crab to dodge
                                                               # in an arc around the edge of the platform without
                                                               # falling off"); nearer the rim it is pulled back in
ARC_PULL = float(os.environ.get("BRAIN_ARC_PULL", "5.0"))       # the inward pull per metre outside ARC_R
COMMIT_M = float(os.environ.get("BRAIN_COMMIT_M", "0.32"))      # a charge inside this is committed: it stops re-aiming
                                                               # for COMMIT_S, so a late sidestep makes it run past
                                                               # (re-aimed every step, the bull followed every dodge
                                                               # round, styles4)
COMMIT_S = float(os.environ.get("BRAIN_COMMIT_S", "1.2"))
COUNTER_M = float(os.environ.get("BRAIN_COUNTER_M", "0.36"))    # the matador's sidestep waits until the foe is this near
LOCK_S = float(os.environ.get("BRAIN_LOCK_S", "1.2"))           # head to head and not moving for this long is a LOCK:
                                                               # the crab goes into the claw mode and swings its arm
                                                               # at the foe (Andrew on styles5: "more movement from
                                                               # front claws to try and break the lock when they are
                                                               # both walking straight into each other")
LOCK_SWING_S = float(os.environ.get("BRAIN_LOCK_SWING_S", "2.5"))
LOCK_MOVE_M = float(os.environ.get("BRAIN_LOCK_MOVE_M", "0.04"))  # less than this of ground gained in LOCK_S is a lock
DASH_M = (float(os.environ.get("BRAIN_DASH_MIN_M", "0.55")), float(os.environ.get("BRAIN_DASH_MAX_M", "0.95")))
                                                               # the hop (jumper.dash, 0.47 m of travel) is launched
                                                               # with the foe this far off and dead ahead
DASH_WAIT_S = float(os.environ.get("BRAIN_DASH_WAIT_S", "3.0"))   # the hop plays out and hands back by itself


def weights(aggression: float, *, near: bool, extra: dict | None = None) -> dict[str, float]:
    """The roll's weights at this aggression; `near` is the foe about to touch,
    when waiting games make no sense and the choice is to meet it or to dodge.
    `extra` is the style's signature intents, rolled beside the common five."""
    a = min(1.0, max(0.0, aggression))
    if near:
        w = {"charge": 0.3 + 0.7 * a, "dodge": 0.6 * (1 - a), "brace": 0.3 * (1 - a), "back_off": 0.0, "circle": 0.0}
    else:
        w = {"charge": 0.1 + 0.9 * a, "dodge": 0.15 + 0.35 * (1 - a), "brace": 0.5 * (1 - a),
             "back_off": 0.3 * (1 - a), "circle": 0.05 + 0.2 * (1 - a)}
    for name, weight in (extra or {}).items():
        # a counter wants the foe coming; a wedge wants room and a standing foe; a strike wants it
        # near; a dash wants it at a hop's range, so it is rolled only when the foe is not near
        w[name] = weight * (1.0 if name == "counter" or (name == "strike") == near or name in ("wedge", "dash") and not near else 0.3)
    return w


def _yaw(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def make_pilot(env, player, side):
    import dataclasses

    from arena import margin as arena_margin
    from moves import MOVES, WALKING, Performer

    from controller import GamepadState

    me, foe = env.scene[side.me], env.scene[side.foe]
    referee = side.referee
    rng = np.random.default_rng(side.seed)
    style = STYLES[getattr(side, "style", None) or "bull"]
    aggression = side.aggression if side.aggression is not None else style["aggression"]
    extra = dict(style["extra"])
    if "dash" in extra and "dash" not in side.presses:
        extra.pop("dash")                                        # (no hop in this bundle yet)
    mode = os.environ.get(f"BRAIN_{side.name.upper()}_MODE", MODE or style["mode"])
    only = os.environ.get(f"BRAIN_{side.name.upper()}_ONLY", "")   # a probe: roll this intent and no other
    if mode and mode not in side.presses:
        print(f"\n[{side.name}] the bundle has no {mode} mode: fighting in locomotion")
        mode = ""
    performer = Performer(player, side.presses, label=side.name)
    moves = MOVES[side.name]
    log = lambda t, text: print(f"\n[{side.name}] {t:6.2f}s  {text}")
    log(0.0, f"brain: style {getattr(side, 'style', 'bull')}, aggression {aggression:.2f}, mode {mode or 'locomotion'}, "
             f"seed {side.seed}, moves {moves}")

    st = {"intent": None, "since": 0.0, "go_at": 0.0, "turning": False, "cue_id": -1, "stood": True,
          "side": 1.0, "phase": "", "phase_since": 0.0, "pushed": None, "mode_pressed": None}

    def claws(frame, t: float, near: bool):
        """In a claw mode every frame holds the arm out (d-pad left, the web-up
        preset) and snaps the claw at a foe within SNAP_M; a strike swings the
        arm between the out and the folded presets instead."""
        if not mode.startswith("claw") or frame is None:
            return frame
        shut = near and (t % SNAP_S) < SNAP_S / 2
        hat = -1
        if st.get("swinging") and near:
            hat = -1 if int(t / STRIKE_S) % 2 == 0 else +1
        return dataclasses.replace(frame, hat_x=hat, lt=1.0 if shut else 0.0)

    def enter_mode(t: float):
        """The frame that gets the crab into its mode, or None once it is there."""
        fsm = player.fsm
        if fsm.mode == mode and fsm.running_policy:
            return None
        if st["mode_pressed"] is None:
            if fsm.mode == "locomotion" and fsm.running_policy:
                st["mode_pressed"] = t
                log(t, f"enters {mode}")
                return GamepadState(buttons=side.presses[mode].buttons)
            return GamepadState()
        if t - st["mode_pressed"] < PRESS_S:
            return GamepadState(buttons=side.presses[mode].buttons)
        return GamepadState()

    def foe_point(ahead: float, left: float):
        """A point in the foe's frame (ahead along its nose, left of it), in the world."""
        f = foe.data.root_link_pos_w[0]
        yaw = _yaw(foe.data.root_link_quat_w[0])
        return (float(f[0]) + math.cos(yaw) * ahead - math.sin(yaw) * left,
                float(f[1]) + math.sin(yaw) * ahead + math.cos(yaw) * left)

    def toward(point):
        """Bearing and range from me to a world point."""
        p = me.data.root_link_pos_w[0]
        dx, dy = point[0] - float(p[0]), point[1] - float(p[1])
        bearing = math.atan2(dy, dx) - _yaw(me.data.root_link_quat_w[0])
        return (bearing + math.pi) % (2 * math.pi) - math.pi, math.hypot(dx, dy)

    def margin(x: float, y: float) -> float:
        return arena_margin(x, y)

    def view():
        p = me.data.root_link_pos_w[0]
        f = foe.data.root_link_pos_w[0]
        dx, dy = float(f[0] - p[0]), float(f[1] - p[1])
        bearing = math.atan2(dy, dx) - _yaw(me.data.root_link_quat_w[0])
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        return bearing, math.hypot(dx, dy), margin(float(p[0]), float(p[1]))

    def room(lx: float, ly: float, reach: float = 0.1) -> float:
        """Plank left `reach` metres along the direction the sticks ask for."""
        p = me.data.root_link_pos_w[0]
        yaw = _yaw(me.data.root_link_quat_w[0])
        x, y = float(p[0]), float(p[1])
        fwd, right = (math.cos(yaw), math.sin(yaw)), (math.sin(yaw), -math.cos(yaw))
        n = math.hypot(lx, ly) or 1.0
        dx = (-ly * fwd[0] + lx * right[0]) / n * reach          # ly negative is forward
        dy = (-ly * fwd[1] + lx * right[1]) / n * reach
        return margin(x + dx, y + dy)

    def arc(side: float, stick: float = WALK) -> dict:
        """The sticks that slide the crab sideways ALONG the table's rim rather than
        across it: the tangent of the circle through the crab, in the sense that is
        `side` (+1 right) of its nose, plus an inward pull when it stands outside
        ARC_R. Recomputed every step, so turning to face the foe meanwhile bends the
        path round with the body (a straight sidestep from the mark walks off a
        0.7 m plank in 0.7 s, and stops short of the rim on the disc)."""
        p = me.data.root_link_pos_w[0]
        x, y = float(p[0]), float(p[1])
        r = math.hypot(x, y)
        yaw = _yaw(me.data.root_link_quat_w[0])
        fwd, right = (math.cos(yaw), math.sin(yaw)), (math.sin(yaw), -math.cos(yaw))
        if r < 0.05:
            tangent = right                                     # at the centre any way round is round
        else:
            tangent = (-y / r, x / r)                           # counter-clockwise
            if tangent[0] * right[0] + tangent[1] * right[1] < 0:
                tangent = (-tangent[0], -tangent[1])           # the sense that is to the crab's right
        if side < 0:
            tangent = (-tangent[0], -tangent[1])
        pull = max(0.0, r - ARC_R) * ARC_PULL
        vx, vy = tangent[0] - pull * (x / r if r > 0.05 else 0.0), tangent[1] - pull * (y / r if r > 0.05 else 0.0)
        n = math.hypot(vx, vy) or 1.0
        v_fwd, v_right = (vx * fwd[0] + vy * fwd[1]) / n, (vx * right[0] + vy * right[1]) / n
        return {"ly": -v_fwd * stick, "lx": v_right * stick}

    def radius() -> float:
        p = me.data.root_link_pos_w[0]
        return math.hypot(float(p[0]), float(p[1]))

    def pick_side(reach: float) -> float:
        """The strafe's sign (+1 right): rolled when both sides keep `EDGE_M` of
        plank after `reach` metres, else the side with more plank under it."""
        p = me.data.root_link_pos_w[0]
        yaw = _yaw(me.data.root_link_quat_w[0])
        x, y = float(p[0]), float(p[1])
        right = (math.sin(yaw), -math.cos(yaw))
        margins = {sign: margin(x + sign * right[0] * reach, y + sign * right[1] * reach) for sign in (-1.0, 1.0)}
        if all(m >= EDGE_M for m in margins.values()):
            return float(rng.choice((-1.0, 1.0)))
        return max(margins, key=margins.get)

    def to_the_middle():
        """Walk to the plank's centre, for a set piece with wood on every side:
        the salute of the first brain take was played from the plank's end and
        went off it. None once within `MIDDLE_M`."""
        p = me.data.root_link_pos_w[0]
        x, y = float(p[0]), float(p[1])
        if math.hypot(x, y) < MIDDLE_M:
            return None
        bearing = math.atan2(-y, -x) - _yaw(me.data.root_link_quat_w[0])
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        rx = facing(bearing)
        return GamepadState(rx=rx) if rx else GamepadState(ly=-0.7 * WALK)

    def facing(bearing: float) -> float:
        """The right stick that faces the foe: 0 when it is ahead within the band."""
        limit = math.radians(TURN_DEG / 3 if st["turning"] else TURN_DEG)
        st["turning"] = abs(bearing) > limit
        # Right on the stick is clockwise; a foe to the left (positive bearing) needs counter-clockwise.
        return -math.copysign(TURN_STICK, bearing) if st["turning"] else 0.0

    def decide(t: float, near: bool, why: str) -> None:
        w = weights(aggression, near=near, extra=extra)
        if "wedge" in w and st.get("last_intent") == "wedge":
            w["wedge"] = 0.0                                    # not twice running: the foe has moved
        st["swinging"] = False
        if room(0.0, 1.0, 0.3) < EDGE_M:
            w["back_off"] = 0.0
        if why.startswith("no room"):
            w["dodge"] = w["circle"] = 0.0                       # the strafe that just ended for lack of plank
        if only in w:
            w = {only: 1.0}
        names = [n for n in INTENTS if w.get(n, 0.0) > 0]
        p = np.array([w[n] for n in names])
        roll = float(rng.random())
        pick = names[int(np.searchsorted(np.cumsum(p / p.sum()), roll, side="right").clip(0, len(names) - 1))]
        hesitate = float(rng.uniform(0.0, 1.2 * (1 - aggression))) if why == "the bell" else 0.0
        st.update(intent=pick, since=None, go_at=t + hesitate, turning=False, pushed=None, side=1.0,
                  last_intent=pick, stage="")
        log(t, f"{pick} ({why}; roll {roll:.2f} over " + ", ".join(f"{n} {w[n]:.2f}" for n in names)
            + (f"; hesitates {hesitate:.1f} s" if hesitate else ""))

    def act(t: float):
        bearing, rng_m, _ = view()
        if t < st["go_at"]:
            return GamepadState(rx=facing(bearing))             # hesitating on the mark
        if st["since"] is None:
            st["since"] = t                                     # the intent's clock starts when it moves
            st["side"] = pick_side(0.5)                         # the circle's way round, where it stands now
        since = t - st["since"]
        intent = st["intent"]
        near = rng_m < NEAR_M
        if intent in ("charge", "brace") and near and not mode and st.get("lock") is None:
            p = me.data.root_link_pos_w[0]
            here = (float(p[0]), float(p[1]))
            if st.get("lock_from") is None or t - st["lock_from"][0] > LOCK_S:
                moved = math.hypot(here[0] - st["lock_from"][1][0], here[1] - st["lock_from"][1][1]) if st.get("lock_from") else 1.0
                if st.get("lock_from") and moved < LOCK_MOVE_M:
                    st["lock"] = ("enter", t)
                    log(t, f"locked head to head ({moved * 100:.0f} cm in {LOCK_S:.1f} s): the claw comes out")
                st["lock_from"] = (t, here)
        elif st.get("lock") is None:
            st["lock_from"] = None
        if st.get("lock") is not None:
            # the lock breaker: into the claw mode (LB), the arm swinging at the foe while
            # still pushing, then LB again and back to the fight
            stage, t0 = st["lock"]
            fsm = player.fsm
            if stage == "enter":
                if fsm.mode == "claw_left" and fsm.running_policy:
                    st["lock"] = ("swing", t)
                elif t - t0 < PRESS_S:
                    return GamepadState(buttons=side.presses["claw_left"].buttons)
                elif t - t0 > 3.0:
                    st["lock"] = ("done", t)
                return GamepadState(ly=-WALK)
            if stage == "swing":
                if t - t0 < LOCK_SWING_S:
                    hat = -1 if int(t / STRIKE_S) % 2 == 0 else +1
                    shut = (t % SNAP_S) < SNAP_S / 2
                    return GamepadState(ly=-WALK, hat_x=hat, lt=1.0 if shut else 0.0)
                st["lock"] = ("leave", t)
            if stage == "leave":
                if fsm.mode == "locomotion" and fsm.running_policy:
                    st["lock"] = ("done", t)
                elif t - t0 < PRESS_S:
                    return GamepadState(buttons=side.presses["claw_left"].buttons)
                elif t - t0 > 3.0:
                    st["lock"] = ("done", t)
                return GamepadState(ly=-WALK)
            if stage == "done":
                st["lock"], st["lock_from"] = None, None
                decide(t, near, "after the lock")
                return None
        if intent == "charge":
            if st["pushed"] is None and rng_m < COMMIT_M and abs(bearing) < math.radians(20):
                st["pushed"] = t                                 # committed: head down, no more aiming
            if st["pushed"] is not None and t - st["pushed"] < COMMIT_S:
                return GamepadState(ly=-WALK)
            if since > RETHINK_S:
                decide(t, near, "rethinks")
                return None
            rx = facing(bearing)
            return GamepadState(rx=rx) if rx else GamepadState(ly=-WALK)
        if intent == "brace":
            if since < CROUCH_S and not MODE:
                st["stood"] = False
                return GamepadState(ry=1.0, buttons=frozenset({"R3"}))
            if st["pushed"] is None and near:
                st["pushed"] = t
                log(t, f"shoves from the crouch (foe at {rng_m:.2f} m)")
            if st["pushed"] is not None:
                if t - st["pushed"] < SHOVE_S:
                    return GamepadState(ly=-WALK)
                decide(t, near, "after the shove")
                return None
            if since > WAIT_S:
                decide(t, near, "nobody came")
                return None
            return GamepadState(rx=facing(bearing))
        if intent == "dodge":
            if st["pushed"] is None and rng_m < DODGE_M:
                st["pushed"] = t
                st["side"] = pick_side(0.3)                     # chosen where it stands when it dodges
                log(t, f"dodges {'left' if st['side'] < 0 else 'right'} (foe at {rng_m:.2f} m, me {radius():.2f} m out)")
            if st["pushed"] is not None:
                if t - st["pushed"] < DODGE_S:
                    return GamepadState(rx=facing(bearing), **arc(st["side"]))
                decide(t, near, "after the dodge")
                return None
            if since > WAIT_S:
                decide(t, near, "nobody came")
                return None
            return GamepadState(rx=facing(bearing))
        if intent == "back_off":
            if since < BACK_S and room(0.0, 1.0) > EDGE_M:
                return GamepadState(ly=+0.7 * WALK, rx=facing(bearing))
            decide(t, near, "after backing off")
            return None
        if intent == "circle":
            if since < CIRCLE_S:
                return GamepadState(rx=facing(bearing), **arc(st["side"], CIRCLE_STICK))
            decide(t, near, "after circling")
            return None
        if intent == "counter":
            # The matador's move: wait facing the foe; when it is DODGE_M off, sidestep
            # for DODGE_S; then turn on it and shove it on for SHOVE_S, toward the rim.
            if st["pushed"] is None and rng_m < COUNTER_M:
                st["pushed"] = t
                st["side"] = pick_side(0.3)
                log(t, f"sidesteps {'left' if st['side'] < 0 else 'right'} (foe at {rng_m:.2f} m, me {radius():.2f} m out), then counters")
            if st["pushed"] is not None:
                u = t - st["pushed"]
                if u < DODGE_S:
                    return GamepadState(rx=facing(bearing), **arc(st["side"]))
                if u < DODGE_S + 2.5:
                    rx = facing(bearing)
                    if rx:
                        return GamepadState(rx=rx)
                    if st["stage"] != "shove":
                        st["stage"] = "shove"
                        log(t, f"counters (foe at {rng_m:.2f} m, {math.degrees(bearing):+.0f} deg)")
                    return GamepadState(ly=-WALK)
                decide(t, near, "after the counter")
                return None
            if since > WAIT_S:
                decide(t, near, "nobody came")
                return None
            return GamepadState(rx=facing(bearing))
        if intent == "wedge":
            # The wrestler's move (the lift probe's, `lift.py`): round to the foe's side,
            # crouch, drive in nose down to WEDGE_M base to base, raise the stance and
            # pitch the nose up with the stick forward -- the heave.
            if st["stage"] == "":
                st["stage"], st["side"] = "flank", pick_side(0.5)
                log(t, f"goes round to the foe's {'left' if st['side'] > 0 else 'right'}")
            if st["stage"] == "flank":
                b, r = toward(foe_point(0.06, st["side"] * FLANK_M))
                if r < 0.12 or since > 6.0:
                    st["stage"], st["pushed"] = "crouch", t
                    return GamepadState(ry=1.0, buttons=frozenset({"R3"}))
                if room(0.0, -1.0, 0.3) < EDGE_M and abs(b) < 0.3:      # (ly negative is forward)
                    decide(t, near, "no room to flank")
                    return None
                rx = facing(b)
                return GamepadState(rx=rx) if rx else GamepadState(ly=-WALK)
            if st["stage"] == "crouch":
                if t - st["pushed"] < CROUCH_S:
                    st["stood"] = False
                    return GamepadState(ry=1.0, buttons=frozenset({"R3"}))
                st["stage"], st["pushed"], st["closest"] = "drive", t, None
                log(t, "drives in under the trunk, nose down")
            if st["stage"] == "drive":
                b, r = toward(foe_point(0.06, 0.0))
                if st["closest"] is None or r < st["closest"][0] - 0.01:
                    st["closest"] = (r, t)
                if r < WEDGE_M or t - st["closest"][1] > 1.0 or t - st["pushed"] > 4.0:
                    st["stage"], st["pushed"] = "heave", t
                    log(t, f"heaves (at {r:.2f} m base to base)")
                    return GamepadState(ry=-1.0, buttons=frozenset({"R3"}))
                rx = facing(b)
                return GamepadState(rx=rx, ry=-1.0) if rx else GamepadState(ly=-WALK, ry=-1.0)
            if st["stage"] == "heave":
                u = t - st["pushed"]
                if u < 1.0:
                    return GamepadState(ry=-1.0, buttons=frozenset({"R3"}))     # the stance up
                if u < 2.5:
                    return GamepadState(ly=-WALK, ry=1.0)                       # nose up, pushing
                decide(t, near, "after the heave")
                return None
        if intent == "dash":
            # The bull's hop: face the foe, and with it at a hop's range press the dash
            # (a reference-guided mode that plays out and hands back to the walk); then
            # roll again. Out of range it closes in on foot and tries once more.
            fsm = player.fsm
            if st["stage"] == "":
                rx = facing(bearing)
                if rx:
                    return GamepadState(rx=rx)
                if DASH_M[0] <= rng_m <= DASH_M[1] and fsm.mode == "locomotion" and fsm.running_policy:
                    st["stage"], st["pushed"], st["rng0"] = "hop", t, rng_m
                    log(t, f"dashes at the foe ({rng_m:.2f} m off)")
                    return GamepadState(buttons=side.presses["dash"].buttons)
                if rng_m > DASH_M[1]:
                    return GamepadState(ly=-WALK)
                if since > 2.0:
                    decide(t, near, "too close to dash")
                    return None
                return GamepadState(ly=+0.5 * WALK, rx=facing(bearing))      # a step back for room
            if st["stage"] == "hop":
                u = t - st["pushed"]
                if u < PRESS_S:
                    return GamepadState(buttons=side.presses["dash"].buttons)
                if fsm.mode == "dash" or u < 1.0:
                    return GamepadState()                                   # in the air: the clip drives
                if u < DASH_WAIT_S and not (fsm.mode == "locomotion" and fsm.running_policy):
                    return GamepadState()
                decide(t, near, f"after the dash: {st['rng0']:.2f} -> {rng_m:.2f} m to the foe")
                return None
        if intent == "strike":
            # The boxer's move: circle the foe with the arm swinging whenever it is in reach.
            st["swinging"] = True
            if since < CIRCLE_S + 1.0:
                return GamepadState(rx=facing(bearing), **arc(st["side"], CIRCLE_STICK))
            decide(t, near, "after the strikes")
            return None
        decide(t, near, "the bell")
        return None

    def pilot(t):
        # The referee's cues come first: a taunt, a cheer or the dance, once each.
        if referee.cue_id != st["cue_id"]:
            st["cue_id"] = referee.cue_id
            piece = referee.cue.get(side.name)
            if piece:
                cut = TAUNT_CUT_S if piece == "taunt" else (VICTORY_CUT_S if piece == "victory" else None)
                performer.start(moves[piece], t, cut_s=cut)
        if performer.busy(t):
            return performer.state(t)
        if not referee.live:
            st["intent"] = None
            if referee.state != st["phase"]:
                st["phase"], st["phase_since"], st["turning"] = referee.state, t, False
            since = t - st["phase_since"]
            if referee.state in ("linger", "ready"):
                if not st["stood"]:
                    if since < 0.3:
                        return GamepadState(buttons=frozenset({"R3"}))   # tapped: the stance back to standing
                    st["stood"] = True
                fsm = player.fsm
                if referee.state == "ready" and since > 0.5 and fsm.mode not in WALKING + ("safe",):
                    performer.leave(t)                      # a latched set piece, see Performer.leave
                    return performer.state(t)
                if referee.state == "linger" and fsm.mode in WALKING and fsm.running_policy:
                    return claws(to_the_middle(), t, False)
            return None
        if mode:
            frame = enter_mode(t)
            if frame is not None:
                return frame
            st["mode_pressed"] = None
        if st["intent"] is None:
            decide(t, False, "the bell")
        frame = act(t)
        frame = frame if frame is not None else act(t)
        return claws(frame, t, view()[1] < SNAP_M)

    pilot.busy = performer.busy
    pilot.performer = performer
    pilot.state = st
    return pilot
