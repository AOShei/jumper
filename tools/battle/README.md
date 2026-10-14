# The crab battle: two Jumpers on a table

A red crab and a blue crab on a plank over the beach, each trying to push the
other off, best of three. The toolkit plays one robot per world; this directory
is the second robot and everything that follows from it. Nothing under
`scripts/` changes, and no task is added: the duel is `play --app` with two
controllers in one physics world.

```bash
# a match: both crabs rolling their moves, taunts at the bell, a dance for the winner
MUJOCO_GL=glfw .venv/bin/python tools/battle/duel.py --scene beach --headless --video out/battle/take1.mp4 \
    --red-aggression 0.9 --blue-aggression 0.2 --seed 7

# the plumbing: two chargers, no set pieces
.venv/bin/python tools/battle/duel.py --scene beach --headless --no-moves --red tools/battle/pilots/charge.py --blue tools/battle/pilots/charge.py

# the red crab looking for the blue one in its own depth picture
.venv/bin/python tools/battle/duel.py --scene beach --red tools/battle/pilots/brawl.py

# you on the keyboard against the charger, in the live viewer
.venv/bin/python tools/battle/duel.py --scene beach --red keys
```

| file | what it is |
|---|---|
| `arena.py` | the world: the second robot entity (`robot2`, the task's own robot copied and recoloured), the plank on two trestles, the two start marks, the cloned reset events, the cross-crab collision bits, a camera on the plank |
| `referee.py` | who lost (flipped onto its head or back, off the plank and on the sand, or down and out on its side), the score, the set pieces it cues around each round, the linger before the reset |
| `moves.py` | each colour's taunt, cheer and victory dance by the bundle's mode names, the pad press that enters each read from the app's own FSM table, and the `Performer` that plays one |
| `duel.py` | the driver: builds the world, runs two `AppPlayer`s from one app, hands each a virtual pad with its pilot, loops with the referee, records |
| `eyes.py` | one crab's senses -- the dToF image self-masked and clustered, the IMU, the encoders -- and the other crab found in it |
| `pilots/brain.py` | the default: rolls an intent (charge, brace, dodge, back off, circle) from weights set by its aggression, seeded per crab; plays the referee's cues |
| `register_gestures.py` | adds the three gestures to `deploy/manifests.json` (Menu + X / Y / A) and `deploy/jumper.controller.toml`, once trained and exported; then `scripts/deploy.py --bundle out/bundle_example --allow-incomplete` rebuilds the app |
| `clips.py`, `make_threat.py`, `make_pump.py`, `make_wave.py` | the battle's own gestures, keyframed the way `tools/fetch/make_zoidberg.py` keyframes the Zoidberg: the threat display (blue's taunt), the claw pump (red's victory) and the wave (blue's victory); each writes `tasks/jumper/gesture_<name>/media/<name>.npz` and a preview under `out/battle/` |
| `pilots/charge.py` | face the foe and walk into it, reading both poses from the world (a cheat, for the plumbing); `CHARGE_MODE=claw_left` charges in the claw mode, arm out, snapping |
| `pilots/brawl.py` | the same from the depth picture alone (`SENSES = True` builds the sensors) |
| `pilots/park.py`, `pilots/lift.py`, `pilots/grab.py` | the probes: a crab that stands still (in a claw mode with its arm out under `PARK_MODE`), the body lever scripted against it, and the claw aimed at one of its legs or at its arm; see "What a crab can do to another" |

## What the second robot needed

Three things that are invisible in the config and would have produced a duel
that looked right and was not; `arena.py`'s docstring has the detail.

* **Two copies of the robot cannot touch.** The family's collision masks give
  each leg a bit and the base none, so two crabs' bases and same-numbered legs
  pass through each other. The arena ORs one bit per crab into every collision
  geom, each crab's `conaffinity` carrying the *other's* bit.
  `tests/test_battle.py` runs MuJoCo's pair test on the compiled two-robot
  model, with the undecorated model as the control.
* **The world's fall reset hides the fall.** `fell_over` would teleport the
  loser upright mid-air. The arena moves it out of reach and the referee takes
  the fallen back after `--linger` seconds.
* **An entity with no reset event stays where it was.** Every startup and reset
  event the task gives `robot` is cloned for `robot2`; `reset_base` and its
  clone are pinned to the marks.

## The match

The referee's phases: `ready` (the controllers settle on the marks), `taunt`
(both crabs play their threat display, the red one the Zoidberg, the blue one
the paw swipe; cut short with Menu after `BRAIN_TAUNT_CUT_S`), `fight`,
`linger` (the fall plays out), `cheer` (the round's winner, hello or salute,
while the loser lies where it fell) and, after the match, `victory` (the
Brazilian for red, the maze for blue). Each piece is a mode of the shipped
bundle; `BATTLE_RED_TAUNT=gesture_hello` and the like swap one by name, which is
how the battle's own gestures go in once trained and bundled
(`gesture_threat`, `gesture_pump`, `gesture_wave`). `--no-moves` skips them all.

Two things the first takes taught about set pieces: a chord pressed while the
controller is in `safe` or still ramping is dropped, so `Performer` waits for
the crab to be walking with its policy running before it presses; and the
shove carries the shover -- in two takes of three the winner was on the sand
beside the loser when its cheer was cued -- so a winner not standing on the
plank is stood back up in its middle by the referee first, the loser left where
it fell. A piece that ends in the tilt fallback keeps its toggle latched and
would replay itself when the crab is next upright; the brain presses Menu to
unlatch it before the next round.

A round is lost three ways: **flipped** (tilt past 90 degrees for half a
second), **off the table and on the sand** (base under 0.16 m, which a crab
reaches a tenth of a second before it lands), or **down and out** (on its
side past the controller's own 50-degree limit for 3 s: the controller has
dropped to `safe` and the crab is not getting up).

The brain's roll: at the bell, when an intent ends, and when the foe comes
within `BRAIN_NEAR_M`, it draws one of its intents from weights set by
`--red-aggression` / `--blue-aggression` (0 to 1). At 1 the crab charges
nine times in ten; at 0 it braces, dodges and backs off and charges only when
the foe is already on it. Each crab has its own generator seeded from
`--seed` (red `seed`, blue `seed + 1`), the seed is printed, and every decision
is logged with the roll and the weights, so a take replays move for move and
a dull one is re-rolled rather than re-cut. Every strafe and the back-off stop
at the plank's edge: on the 0.7 m plank a circle at full stick from the mark
walked straight off the side in 0.7 s.

## What a crab can do to another

Measured against a parked crab (`pilots/park.py`) on 2026-10-14, because the
fight was meant to be claws and flips and the first takes were shoves:

* **Nothing on the crab flips a standing crab.** Head on, the two crabs' front
  feet meet at 0.33-0.38 m base to base (the front feet stand 0.18 m ahead of
  the base, `NOMINAL_FOOT_XY`), so a crouched nose-down charge (`lift.py`) never
  reaches the underside, which is 0.096 m over the plank; the tall nose-up push
  only shoves. The shove moved the parked crab 0.35 m per push and tipped it
  3 degrees until it went off the plank's end backwards, where it reads as a
  flip (88 degrees nose up as its rear legs went over). Tipping a 2.54 kg crab
  about its rear feet needs about 20 N above its centre of mass (0.11 m); the
  carried arm holds its preset with 1.75 N*m, 12 N at the claw and 0.09 m up.
* **The claw does not take a leg in a live approach.** The web-up arm carries
  the mouth 0.18 m ahead of the base, 0.09 m over the plank, closing side to
  side (palm at (0.155, 0.07), finger tip at (0.20, -0.05) in the base frame).
  The foe's front strut (its claw, standing on the plank) tops out at 0.075 m,
  under the mouth, and its forearm sits at 0.085 m, 0.27 m from the grabber's
  base with a 12 cm sidestep, which the claw mode's dead-banded walk (`grab.py`,
  strafe pulses and all) brought to within 4-10 cm before stalling, five times
  in five. The finger shut fully every time (+0.10 rad; open is -1.47).
* **Two claws out clash.** With both crabs in a claw mode the arms are bars at
  one height with no front feet under them, and the mouths meet at 0.28 m with
  a 7 cm offset (`grab.py` with `PARK_MODE=claw_left`, aim (0.10, 0.066)): the
  claws touched tip to tip twice, the grip closed on nothing. A claw-mode charge
  (`CHARGE_MODE=claw_left`) shoves the parked crab along at about 4 cm/s, a
  third of the posture scrum's pace, and the charger toppled over the foe at the
  plank's end before the foe fell.

So the plank decides the rounds by shoving, the "flips" are falls off its
end, and the claws can be in the picture (out, snapping, clashing) without
deciding anything. Takes: `out/battle/lift1-3.mp4`, `grab1-5.mp4`,
`clawcharge1.mp4`.

That picture is `BRAIN_MODE=claw_left`: both brains enter the claw mode at
the bell (after the taunts, which play from the walk), carry the arm out and
snap at a foe within `BRAIN_SNAP_M`; the set pieces press from and hand back
to a claw mode as they do from the walk (`moves.WALKING`). `out/battle/claws1.mp4`
(seed 2, 0.9 against 0.2) is the same match as `brain4` fought with claws out:
red 2-0, rounds of 6.9 and 6.8 s, the claws meeting in the middle before the
shove, the hello cheered from the claw mode, 54.6 s of film.

## Styles

A crab fights one of four playbooks, `duel.py --red-style bull --blue-style
matador` (the defaults; `--red-aggression` and the like still override the
style's own weight):

| style | aggression | the extra intent | what it looks like |
|---|---|---|---|
| `bull` | 0.9 | `dash` (only where the bundle carries the hop) | charges, and inside `BRAIN_COMMIT_M` (0.32 m) stops re-aiming for `BRAIN_COMMIT_S` so a sidestep is not followed |
| `matador` | 0.2 | `counter` | braces until the foe is `BRAIN_COUNTER_M` (0.36 m) off, steps aside along the arc, turns and shoves the passing flank |
| `wrestler` | 0.6 | `wedge` | walks to a point `BRAIN_FLANK_M` out on the foe's side, crouches, drives in to `BRAIN_WEDGE_M` base to base and heaves |
| `boxer` | 0.5, claw out | `strike` | circles with the arm swinging every `BRAIN_STRIKE_S` |

Every sidestep, counter, circle and strike moves along the table's tangent,
`arc(side, stick)`: the stick is the tangent at the crab's bearing from the
centre plus an inward pull of `BRAIN_ARC_PULL` per metre outside `BRAIN_ARC_R`
(0.45 m on the 0.8 m disc), so a dodge is a step round the rim rather than
across it. On the plank the first dodges walked straight off the side.

Two bulls walking into each other stall nose to nose. A charge or a brace with
the foe within `BRAIN_NEAR_M` that gains under `BRAIN_LOCK_MOVE_M` (4 cm) in
`BRAIN_LOCK_S` (1.2 s) is a **lock**: the crab presses LB into the claw mode,
swings and snaps the arm for `BRAIN_LOCK_SWING_S` while still pushing, presses
LB back and rolls again. In `out/battle/lock2.mp4` (a bull against a bracing
bull) five locks broke that way.

The **dash** is `jumper.dash`, the high jump's own recording carried 0.47 m
forward (`make_dash.py`, `DASH_M`): the base's x ramps through the push, holds
0.72 m/s over the 0.42 s flight and stops through the absorb, the airborne
feet move with the body and the planted feet stay, the joint angles are the
jump's. It trains by resuming the jump's policy, and the jump's reward weights
were not enough: with them the hop gained 0.10 m a time in the duel, because
the joint and foot terms score the jump's shape standing still as well as
moving, and only the velocity term asks for the travel. `jumper.dash` owns
three numbers of its own (the velocity term at weight 6 and sigma_sq 0.08,
joint tracking at 1.5; `tasks/jumper/dash/env_cfg.py`), and retrained with them
for 2000 iterations (2048 environments, 1.1 s an iteration on the RTX 3060) its
training `travel` metric went from 0.20 to 0.39 m and the hop in the duel
(`BRAIN_RED_ONLY=dash` against `pilots/park.py`, `out/battle/dash4.mp4`) gains
0.21-0.26 m a press with the base 12-13 cm up at its apex, the jump's own 14:
from 0.55 m it lands at 0.38, which is where the front feet meet. The bull's
`dash` intent faces the foe and presses L3 when the foe is `BRAIN_DASH_MIN_M`
to `BRAIN_DASH_MAX_M` off, then waits `BRAIN_DASH_WAIT_S` for the clip to hand
back, and logs the range before and after and the apex. `jumper.sprint`
is the posture walk retrained with a 1.2 m/s ceiling and a longer stride, the
pad's B from `locomotion`; `BRAIN_RED_MODE=sprint` fights the whole round in
it.

The styles so far, on the disc (seeds 1, 2, 3, 5): the bull takes 6 rounds in
11 against the matador, and the matador's counter won `styles7.mp4` 2-1 on a
seed the bull had taken before. Wrestler against boxer went 0-2 and 2-0 with
one 45 s draw: the wedge gets under the foe's trunk (8 mm of lift, the posture
policy yielding under the load) but nothing trained flips a crab, so the
heave is a shove from the side.

## Measured

Framerates on the RTX 3060, warp backend, one environment, 200 Hz world,
headless, 2026-10-14:

| | steps / s | of real time |
|---|---|---|
| two crabs, no depth sensors | 48 | 0.24x |
| two crabs, both depth sensors | 37 | 0.19x |

So the sensors are built only when a pilot says `SENSES = True`. A profile of
the rest put a third of each step in the environment's own command and
observation managers, which `play --app` never reads; `duel.py` skips both
(`skip_the_managers`), as `mjrl.replay.skip_rewards` already skips the rewards.

The first charger-versus-charger match: the crabs meet in the middle within
2 s of the bell and one is on its side within 3-4 s; red lost both rounds, with
symmetric pilots, so the start marks are not yet known to be fair.

The brain takes (`out/battle/brain1..4.mp4`, red at aggression 0.9 against
blue at 0.2, the 0.7 m plank): rounds last 0.8 to 12.8 s and end nine times
in ten with a crab shoved off the plank's end; seed 1 went blue 2-1, red 2-1
and blue 2-1 as the brain was fixed between takes, seed 2 red 2-0. With both
crabs rolling, the same seed and the same brain replay the same match. A full
match with its taunts, cheers and dance is 55 to 98 s of film at 48 steps/s
headless, about four minutes to record.

## Knobs

Environment variables, read by `arena.py` and the pilots: `BATTLE_ARENA`
(`disc`, a round table of radius `BATTLE_DISC_R` 0.8 m on a pedestal, so a crab
can dodge to the side and the edge is the same distance in every direction;
`plank` for the first takes' board), `BATTLE_PLANK_LEN`
(2.0 m), `BATTLE_PLANK_WIDTH` (0.7; at 0.5 the first shove settled every
round), `BATTLE_PLANK_TOP` (0.25), `BATTLE_START_GAP` (0.9), `BATTLE_CAM_DIST` /
`BATTLE_CAM_AZ` / `BATTLE_CAM_EL` / `BATTLE_CAM_X` (2.3 / 90 / -12 / 0; 90 is the side view, the
look-at slides along the plank), `BATTLE_<COLOUR>_<PIECE>` for
the set pieces, `BRAIN_*` for the brain's distances and timings (`BRAIN_MODE`, or `BRAIN_RED_MODE` /
`BRAIN_BLUE_MODE` for one crab, `BRAIN_SNAP_M`, `BRAIN_SNAP_S` for the claw fight; the
styles' knobs are listed above), `CHARGE_*` and
`BRAWL_*` for the other pilots' sticks (`TURN_DEG`, `TURN_STICK`, `WALK`,
`CROUCH_S`).
