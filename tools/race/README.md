# The cleaning race

Andrew (2026-10-15): "a race to clean a room between two crabs where they share a
room and there is a line down the middle, there are objects evenly distributed
throughout the room and the goal of the red crab is to put everything on the right
side and the blue crab tries to put everything on the left side, in the end the
winner is the crab with the most amount of objects on their target side." With
"the depth vision and cost map with the path planner visible for each robot", and
objects with "some variation. We can have boxes, but then some balls that roll
would be nice and maybe some other shapes as well."

    python tools/race/race.py --scene studio --headless --seed 3 --race-s 120 --video out/race/take.mp4
    python tools/race/race.py --red-truth --blue-truth          # the ground-truth control
    python tools/race/race.py --red keys                        # drive the red crab yourself

| file | what |
|---|---|
| `room.py` | the rink: two crabs on a painted floor with a kerb, the line at x = 0, the mirrored layout of cubes, balls, cans and pucks |
| `referee.py` | the count on each half (ground truth, the referee's alone), the clock, the winner's dance |
| `sight.py` | one crab's dToF reading: the clusters that are object-sized, the kerb's points, the other crab |
| `pilots/sweeper.py` | the pilot: remember things, pick the nearest on the wrong half, get behind it, shove it over; `--red-truth` makes it read the world instead |
| `hud.py` | the readout in each top corner: depth image (grey by range; blue the crab's own legs and claws, red the things, yellow the kerb, magenta the other crab), map with the plan and the remembered things, the pad; the score and clock across the top |
| `race.py` | the runner: two `AppPlayer`s on one app, as `tools/battle/duel.py` |

Everything reused is the fetch film's and the battle's: the second crab and the
crabs' collision (`tools/battle/arena.py`), the set pieces (`moves.py`), the dToF
reading (`tools/fetch/sense.py`'s way, in `sight.py`), the occupancy map and A*
(`tools/fetch/mapper.py`). The pilots drive the shipped bundle through virtual
pads, so the whole thing is a `play --app` the robot could run.

## The room

4 x 2.6 m, a 35 cm kerb round it with every corner cut off by a 45-degree wall
0.45 m along each side (a thing in a square corner has two free faces and both
shove it into a wall -- no crab wider than the gap can get behind it, so it
stayed there for half a match; on the chamfer it is shoved along the diagonal
and out), (a ball cannot roll over it, a crab cannot climb
it, and it is what the crab localises against: taller than a crab and every
object, so the depth image cannot mistake it for either; the camera-side wall is
drawn as glass so it hides nothing), each half in its team's colour,
the line and a touchline in white. Round it the set, drawn only: a dark floor,
three walls (none on the camera's side) with a banner in each team's colour on
its half of the back wall, a post at each corner; the daylight scene's sun is
turned overhead so the kerb throws no shadow across the rink. Six objects a side by default
(`RACE_OBJECTS`), the same six mirrored through the line so the start is even,
in colours that are neither team's: cubes 15 cm (0.25 kg), balls 17 cm (0.1 kg,
they roll), cans 11 x 17 cm standing (0.2 kg, they tip and roll on their side),
pucks 19 x 7 cm (0.2 kg). Enlarged twice from the first set (9, 10, 6.6 x 11,
12 x 5 cm), which read small on camera and vanished against the kerb; at 3 cm
tall the first puck was walked over (probe1). The crabs
start at x = +-1.55 on the line's axis, facing each other; red's half is x > 0,
the picture's right with the camera on -y.

## The shove

There is no carrying: the claw lifts a mug by its handle and nothing else
(`tools/fetch/README.md`), and the battle's bull already shoves. The sweeper walks
at the object (the planner's way round whatever is in between) until it is a metre
off, then **orbits** it the way the matador circles the bull: nose on the object,
strafing round it on a circle of 0.45 m, the forward stick holding the radius,
toward the point behind it as seen from its own half (slanting or beside it when
straight behind would be in the kerb, and the other way round when the arc would
run into the kerb). Within 45 degrees of behind it -- or after 6 s of circling,
from wherever it is -- it shoves: walks into the object with the lateral stick
keeping it under the nose until it is 0.3 m past the line, then backs off 0.7 s so
the legs do not drag it back. The first version walked to the point behind the
object, stopped, turned on the spot at 33 deg/s (up to five seconds), and
reversed after every target it gave up; Andrew: the crabs "orient towards an
object and then back up first instead of just going for it... the crabs need to
be more aggressive". Now a reverse follows only a shove, or a shove that stuck.
A cube slides, a can tips and rolls, a puck slides, a ball rolls ahead and is
left as soon as it is over.

Ground truth (`--red-truth --blue-truth`, probe1, 60 s, seed 3): red 7-5, 38
crossings; the balls near the line crossed back and forth under both crabs'
feet in the first two seconds.

## What the crab knows

Each crab reads its dToF (54 x 42 zones, 55 x 42 degrees), its IMU and its
encoders, and nothing else; it is told the room it is in (the kerb's rectangle,
the line, and that it woke up on its own mark), which a robot put down in a room
may know. From the depth image: the floor plane under the lowest feet along the
IMU's up (1.3 cm over the floor at rest, probe_sight), the object clusters
(depth-connected, 3.5-21 cm tall, 3-26 cm across, not touching the picture's
edge), the other crab (12-27 cm tall, over 26 cm wide and over 12 cm deep: its
legs spread front to back where a thing shows only its face), the kerb (anything
27 cm or taller: at 10 cm it was a two-zone sliver at range told from a cube by
width alone, and the crabs lost it), and the small clusters within 0.45 m of a
crab, which are its legs. A thing against the kerb joins the kerb's cluster
through its edges, where its depth runs into the wall's (Andrew: "when objects
are up against a wall the crabs can't see them"): inside a cluster that reaches
kerb height, the low zones nearer than the wall in their column by 4 cm are cut
out and clustered on their own, by depth -- by adjacency, two cubes side by side
came out as one wide low cluster and read as a crab (day12).

Things seen are remembered in the map's frame, matched sighting to sighting
within 0.3 m, forgotten when in plain view and unseen for 1 s (not when the other
crab is within 0.7 m of where they were; at 2 s the crabs walked at ghosts of
things long shoved) or unseen for 15 s anywhere; a fresh sighting is picked
before a stale memory, and a thing the other crab stands at is left to it. A crab
whose memory holds nothing on the wrong half walks to the middle of that half
and turns a full circle before it believes it.

### Localisation

The map's pose is dead-reckoned from the sticks (measured in this room,
probe_speed and the loop's readout: forward 0.56 m/s at a 0.75 stick and 0.7 at
0.9, 0.13 at 0.5, a dead band of 0.4 and 1.4 m/s per unit past it; strafe 0.36,
0.40, 0.59 and 0.70 m/s at 0.5, 0.6, 0.75 and 0.9, a straight line of 0.85 m/s
per unit from 0.08 with no dead band to speak of -- modelling it with the walk's
band put 1.1 m/s on the orbit's 0.9 stick and ran a pose three metres off; turn
33 deg/s at 0.62, ~50 from 0.8 up) and the heading is the gyro's (within 1
degree over two minutes). Pushing into the other crab the walk stick moves
nothing and is not reckoned; the strafe still is (zeroing it too left an orbit
round the other crab unreckoned, 0.7 m in 5 s). The pose is clamped inside the
kerb, where the body has to be. The
fetch's scan match against the map it builds was tried first and failed here:
against twelve things that move and a crab that walks it pulled the pose 0.4-1.0
m off within 30 s (sense4), and widening its search made it worse. What works is
the kerb: with the heading known, every kerb point the crab sees lands some way
off the wall it belongs to, and the median of that per wall is the pose error
along that wall's normal (`Senses.wall_fix`). Only wall-shaped clusters count
(wide, spread over 0.3 m along the wall, within 6 cm of a line across it): a cube
cut by the picture's edge taken for kerb ran the pose 2 m off in 5 s (sense6).
The fix is 0.3 of the discrepancy a frame, 3 cm at most, and the radius a point
may be matched to a wall within grows from 0.6 m by 0.15 m a second while the
pose is outside it, so a pose that ran off while the kerb was hidden is pulled
back (resetting that radius on every landed fix left a 0.9 m error one step
every two seconds: blue's pose sat a metre off for 50 s in day4, with 134 fixes
landing in one 5 s window and the error growing; the synthetic probe showed the
first fix landing and no second). The
reckoning stops while the other crab is right ahead: pushing against it, the
sticks move nothing, and red's pose ran 1.2 m off in 5 s doing exactly that
(sense7).

Result (day6, 120 s, seed 5, the tall kerb): red's map pose within 0.18 m of
the truth at every 5 s check and blue's within 0.27 m, means 3.5 and 5.6 cm, the
truth read only by the log. (With the 10 cm kerb, sense8/sense9: within 0.12 m
but one check.)

### The shove, sensed

Under the nose a thing leaves the picture: the sensor's lower edge meets the
floor 0.31 m out, so a 5 cm puck against the shell is below it. The sweeper then
takes the thing to be where the nose is, moving with it, for up to 3 s, and calls
it over by its own pose; longer than that unseen, it gives up. A thing the
sweeper is shoving that turns out to be the other crab is given up at once.

### Things against the wall, and the claw

Nobody can stand straight behind a thing pressed to the kerb. The standing
point is tried straight behind, then slanting 30, 55 and 75 degrees, toward the
room's middle first, and the first the body fits in (0.24 m clear of every wall,
chamfers included) is taken: every shove keeps a component toward the line, and
a thing against the far wall is peeled off it a slant at a time. (A first rule
shoved it along the end wall to the corner instead; Andrew: red "seemed to just
push all his objects up".) The things nearest the line are picked first.

While a crab shoves, the thing fills the sensor's whole picture and no wall is
in view, so the pose runs on the dead reckoning alone for the length of the
shove and drifts up to 0.3 m before the next kerb sighting pulls it back
(day14: means 0.27 and 0.32 m over the race, under 0.05 m between shoves).
A limit of a forward-looking sensor on a shove, written down rather than hidden.
From close up the kerb shows only the band inside the picture (8-19 cm up at 15
cm), the height of a thing, and red shoved the top wall for a minute (day14);
the tell is a cluster running out of the picture's top row, which nothing
object-sized does between 0.2 and 0.5 m (the view's upper edge is 0.137 +
0.384 r up). Nearer than 0.2 m a thing is cut too -- a 17 cm ball at 10 cm read
as wall and the wall fix pulled the pose 2 m onto it (day17) -- so there the
map's own knowledge of the walls decides, and the wall fix takes no point under
0.25 m. The orbit keeps the way round it chose: flipping between two blocked
ways each frame was a crab jiggling on the spot beside a thing.

Andrew asked for "a walking gait with claws extended" for the shove. Measured
(probe_armpush: red in `claw_left`, the arm out web-up, walking into a staged
cube, ball and can at a 0.8 stick): the cube slid 41 cm in 1.5 s without
tipping, the ball and the can rolled ahead, and the thing stayed a thing in the
picture with kerb points beside it. In a race (`SWEEP_CLAW=1`, day15/16) it lost
badly: the switch into the mode is a ramp the controller ends only at the mode's
home pose, that pose lifts the arm's leg, and against a thing it is never
reached -- both crabs spent the race "ramping", 4 and 10 crossings against 62.
The robot has the same ramp. Off by default.
At contact the thing is 10 cm from the sensor and spans the picture's edges; a
cluster nearer than 0.45 m is a thing regardless (without that the crab went
blind on contact, probe_contact: 1895 zones, edge-cut, not a thing).

Andrew asked for the claws to do this instead ("use their arms to push and move
objects away from the walls"). Measured (probe_rake, 2026-10-15, a 15 cm cube
staged against the -x kerb, red in `claw_left` with the arm out at the web-up
and the thumb-up presets, the nose pitched each way, closing on it and reversing
for 4 s): the mouth stands 29 cm ahead of the base and 8.5 cm over the floor,
at the cube's face, not over it; the 6 cm aperture closed on the face without
taking hold; walking into it moved the cube 5-10 cm along the wall and the
reverse moved it nothing. The claw lifts a mug by its handle and nothing else
(`tools/fetch/README.md`), and none of these objects has one. A claw move that
would work needs something it can take: a 5 cm prize object, or a handle.

## Results

| take | eyes | clock | score | crossings | shoves |
|---|---|---|---|---|---|
| probe1 | truth | 60 s | red 7-5 | 38 | 5 / 4 |
| sense7 | sensed | 60 s | red 7-5 | 10 | 2 / 2 |
| sense8 | sensed | 120 s | blue 7-5 | 22 | 3 / 3 |
| sense9 | sensed, with the search | 120 s | red 9-3 | 46 | 7 / 7 |
| day1 (seed 5, `--scene daylight`) | sensed | 120 s | tie 6-6 | 28 | 4 / 3 |
| day2 (seed 5, the bold set, larger objects) | sensed | 120 s | tie 6-6 | 36 | 6 / 5 |
| day4 (seed 5, the 35 cm kerb) | sensed | 120 s | red 7-5 | 22 | 12 shoves; blue's pose a metre off from 35 s |
| day6 (seed 5, the radius reset fixed) | sensed | 120 s | red 9-3 | 40 | 13 shoves; poses within 0.27 m throughout |
| day7-day10 (the orbit) | sensed | 120 s | -- | -- | the strafe model, an orbit round the other crab, an orbit into the kerb and a too-strict "behind" each found and fixed in turn |
| day11 (seed 5, the orbit) | sensed | 120 s | red 7-4 | 25 | 11 shoves; poses within 0.08 (blue) and 0.45 m (red), means 3 and 6 cm |
| day13 (seed 5, larger objects, wall-side things seen, wall-following shoves) | sensed | 120 s | blue 9-3 | 26 | 29 shoves (corners included); poses means 8 and 16 cm |
| day14 (seed 5, chamfered corners, the peel, contact kept) | sensed | 120 s | red 9-3 | 62 | 11 shoves, no "not seen" give-ups, nothing left in a corner; poses means 27 and 32 cm (the shoves are blind) |
| day15/16 (the claw shove, `SWEEP_CLAW=1`) | sensed | 120 s | tie, blue 7-5 | 4, 10 | the crabs spent the race ramping into the claw mode (see below) |
| day18 (seed 5, the close-wall rule and the orbit's way kept) | sensed | 120 s | red 7-5 | 30 | 11 shoves, no "not seen"; poses means 23 and 24 cm |

Objects shoved deep into a half end up against the far kerb, where the other
crab can only get beside them, not behind: the slanting approaches handle it
at a slower pace. Each shove is 10-15 s from the pick to the back-off.

## Knobs

`RACE_ROOM_X/Y`, `RACE_OBJECTS`, `RACE_MIX`, `RACE_START_X`, `RACE_CAM_DIST/AZ/EL/Y`
on the room; `SWEEP_WALK`, `SWEEP_PUSH`, `SWEEP_BEHIND_M`, `SWEEP_OVER_M`,
`SWEEP_FWD_PER`, `SWEEP_SIDE_PER` on the pilot; `--race-s`, `--objects`,
`--seed`, `--no-hud`, `--no-moves` on the runner.
