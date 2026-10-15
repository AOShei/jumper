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
in colours that are neither team's, and all of them low: round pucks 19 x 7 cm
(0.2 kg), square pucks 16 x 16 x 7 cm (0.25 kg) and balls 12 cm (0.08 kg, they
roll). The sensor's picture reaches 0.137 + 0.384 r m up the room at range r,
so nothing here ever runs out of its top, and a cluster that does is the kerb
or the other crab, from any distance (Andrew: "an object can never fully take
up the crab's view and in this case we know if something does obscure the full
view it must be a wall"). Before, cubes of 15, cans of 17 and balls of 17 cm, enlarged twice
from a first set that read small on camera and vanished against the kerb; a
17 cm ball at 10 cm read as wall and pulled the pose 2 m onto it (day17). At 3
cm tall the first puck was walked over (probe1); at 7 the shell's edge (8 cm
up) meets it. The crabs
start at x = +-1.55 on the line's axis, facing each other; red's half is x > 0,
the picture's right with the camera on -y.

## The shove

There is no carrying: the claw lifts a mug by its handle and nothing else
(`tools/fetch/README.md`), and the battle's bull already shoves. The sweeper walks
at the object (the planner's way round whatever is in between) until it is 0.7 m
off, then **orbits** it the way the matador circles the bull: nose on the object,
strafing round it on a circle of 0.40 m (a 12 cm gap to a puck's edge: the shell's
front is 0.18 m ahead of the base), the forward stick holding the radius,
toward the standing point -- straight behind it as seen from its own half when
the body fits there, else the cheapest of the slants (30, 55, 75 degrees either
way) by the way round from where it is, kept for 1.5 s once chosen so it does
not dither beside a wall, and the other way round when the arc would run into
the kerb (a metre out and a 0.45 m circle had the crabs "strafe around objects
too far away", Andrew). Within 30 degrees of the shove's heading -- or after
10 s of circling, from wherever it is -- it shoves: walks into the object with the lateral stick
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
IMU's up (1.3 cm over the floor at rest, probe_sight; a zone is off the floor
from 3 cm plus 2.5 cm per metre of range, since a 1.4 degree tilt of the plane
put the floor a metre out 2.5 cm up and it read as low things at the kerb's
foot, picked twice in a probe), the object clusters
(depth-connected, 3-14 cm tall, 3-26 cm across, not touching the picture's
edge), the other crab (14-27 cm tall, over 26 cm wide and over 12 cm deep: its
legs spread front to back where a thing shows only its face), the kerb (anything
27 cm or taller: at 10 cm it was a two-zone sliver at range told from a cube by
width alone, and the crabs lost it), and the small clusters within 0.45 m of a
crab, which are its legs. A cluster taller than 14 cm, or cut by the picture's
top row, is never a thing, whatever is remembered there: the kerb from 17 cm
out (the other crab's shell is cut only nearer), and nearer the map says
whether a wall is there. A thing against the kerb joins the kerb's cluster
through its edges, where its depth runs into the wall's (Andrew: "when objects
are up against a wall the crabs can't see them"): inside a cluster that reaches
kerb height, the low zones nearer than the wall in their column by 4 cm are cut
out and clustered on their own, by depth -- by adjacency, two cubes side by side
came out as one wide low cluster and read as a crab (day12).

Things seen are remembered in the map's frame, matched sighting to sighting
within 0.3 m -- and a cluster within 0.3 m of a remembered thing stays a thing
whatever this frame's gates say of it, short of kerb height, so the tracking
does not flicker between thing and wall from frame to frame (the overlay's
colours are a majority of the last three frames for the same reason) -- forgotten when in plain view and unseen for 1 s (not when the other
crab is within 0.7 m of where they were; at 2 s the crabs walked at ghosts of
things long shoved) or unseen for 15 s anywhere; a fresh sighting is picked
before a stale memory, and a thing the other crab stands at is left to it. A crab
whose memory holds nothing on the wrong half walks to the middle of that half
and turns a full circle before it believes it.

### Localisation

The pose is reckoned from the **legs** (`sight.leg_velocity`): forward
kinematics from the encoders puts every foot in the body frame each step, and a
foot standing on the floor does not move in the world, so its motion through
the body frame is the body's motion reversed, less the turn the gyro reports.
Which feet stand is the question: the gait lifts a foot 2.3 cm at most
(probe_feet), so height cannot tell stance from swing -- a 1.5 cm band kept
every middle foot and read a walk 32% short -- but against the direction the
sticks ask for, the standing feet are the ones moving fastest, and the median
of those within 75% of the fastest reads a walk and a reverse within 5% of the
truth (probe_legodom: 1.64 for 1.71 m, 0.86 for 0.86). A strafe reads 27% long
twice over, and is scaled by the measured 0.79 rather than explained. Before
the legs, the pose was reckoned from the sticks, and the map's stall test --
the sticks say walk and the picture does not change, so the crab is wedged --
fired through every shove, since the thing being shoved fills the picture and
moves with the crab: the pose froze while the body walked, which is where the
metre-sized errors came from (day19 and before). The stall test is off with the
legs on (`SWEEP_LEG_ODOM=0` restores the sticks).

The legs slip too, jammed on the kerb: nose-on they say 1.9 m of walk for
0.27 m moved in 5 s, 2.5 m for 0.6 at a 45 degree slant and 1.8 m for 0.6
strafing into it, and the base stops 0.18 m from the kerb's face
(probe_walljam, 2026-10-15, the raw reckoning). Andrew: "whenever the crab walks
into the wall the odom loses track, because even though it is moving its legs
they slide". Three answers, all from what the crab carries. Kerb zones at the
nose (twenty of them nearer than 0.24 m within 12 cm of the nose's axis; a
cluster was tried, and a side wall running alongside at 0.22 m is one cluster
whose nearest point is beside the shell, day26) while the legs say forward are
a slip: it reckons nothing and the shove is given up at once. A leg velocity into a wall the pose
already stands at (0.20 m from it) loses that component. And a shove jammed
with the thing under the nose -- the kerb a puck's width off -- is the wall
fix's: the things are low, so the kerb above the thing stays in view, and the
fix at 3 cm a frame, 50 frames a second, out-runs the 0.56 m/s the slipping
feet report. The pose is kept 0.17 m inside the kerb either way. Two rules
for that last case were tried and dropped: a kerb 0.47 m off behind a thing
taken for a jam gave up every shove along a side wall (49 in a race, day23),
and 10 cm of reckoned walk in half a second with the kerb ahead coming no
nearer fired on nearly every shove (day24): a side wall seen at a slant keeps
its median range as the crab walks beside it, and the nearest kerb cluster
ahead is a different one frame to frame.

Still open: a crab working along a side wall gets no fix on x until an end
wall is in view again, and drifts up to 0.4-0.8 m meanwhile (the legs read a
shove under load a little short). Taking the chamfers as walls too was tried
for that and reverted: a short stretch of side wall near a corner passed the
chamfer's line test with a wrong residual and the pose jumped 1.2 m (day21).

The map's cells decay toward unknown with a 6 s half-life, the known kerb
re-asserted, so a thing shoved away leaves no obstacle for the planner to route
round, and the inset draws the room the crab was given over what it has mapped
lately. The walk and the shove run at a 0.75 stick, 0.56 m/s, where the
reckoning was measured (Andrew: "slow down the speed a little so that the odom
is more accurate").

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
(wide, spread over 0.3 m along the wall or over 80% of the picture's width at
that range for a wall at the nose, within 6 cm of a line across it): a cube
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

Nobody can stand straight behind a thing pressed to the kerb, and nobody fits
between a thing and the kerb either. A thing within 0.3 m of a wall is shoved
ALONG it, toward the line: along the end wall to the nearer corner, along the
chamfer onto the side wall, along the side wall to the line, the crab standing
beside it on the wall's line and the shove's own lateral stick keeping it
under the nose. (A first rule shoved it along the end wall to the corner and
was replaced by a peel -- a slant off the wall, 30, 55 or 75 degrees -- that
had no standing room the body fits in near a wall, so its slants drove the
thing into the wall and the crab jammed on it: 30 jam give-ups in a race,
day25. A jammed thing is picked again after 4 s rather than 20, since the
route from where it now lies is a different one.) In the open the standing
point is tried straight behind, then slanting 30, 55 and 75 degrees, toward
the room's middle first (a slant that drives a thing within 0.6 m of a side
wall into it costs a metre more; and the side of "the middle" was read off the
slant's sign alone before, which is the middle for red and the wall for blue:
blue drove a puck into the camera-side wall, day25), and the cheapest the body
fits in (0.20 m clear of every wall, chamfers included; the jammed base
measures 0.18) is taken and kept for the job unless the thing moves 0.3 m or
it stops fitting (re-chosen every 1.5 s before, it flipped the slant's side
and sent the orbit back the way it came -- Andrew: a crab "circles around an
object and then reverses direction multiple times, even though it should just
go for it"). The things deepest in the other half are picked first.

While a crab shoves, the thing fills the sensor's whole picture and no wall is
in view, so the pose runs on the dead reckoning alone for the length of the
shove and drifts up to 0.3 m before the next kerb sighting pulls it back
(day14: means 0.27 and 0.32 m over the race, under 0.05 m between shoves).
A limit of a forward-looking sensor on a shove, written down rather than hidden.
From close up the kerb shows only the band inside the picture (8-19 cm up at 15
cm), the height of a thing, and red shoved the top wall for a minute (day14);
the tell is a cluster running out of the picture's top row, which no thing
does now that none is taller than 12 cm (the view's upper edge is 0.137 +
0.384 r up; with 17 cm balls a thing was cut nearer than 0.2 m, and one at 10
cm read as wall and pulled the pose 2 m onto it, day17). Nearer than 17 cm the
other crab's shell is cut too, and there the map's own knowledge of the walls
decides. The wall fix takes the kerb under the nose as well: it is the fix a
crab jammed on it needs. The orbit chooses its way round once, the short way
unless it runs into the kerb, and keeps it (a way re-chosen each frame reversed
whenever the goal angle moved under it; flipping between two blocked ways each
frame was a crab jiggling on the spot beside a thing, day14). Blocked short of
the goal by less than a right angle it stops and shoves from there, facing the
thing: for a thing near a corner the standing angle lies in the kerb's
clearance, and going round the long way met the other wall, so the orbit swung
between the two walls' limits for its whole 10 s (probe_orbitwall, 2026-10-15;
with the goal made the clamped standing point's angle and the side check at
0.2 m instead of 0.35, it arrived in a second and the puck went along the side
wall and over). And the shove starts from anywhere within 45 degrees of the
standing angle with the nose on the thing, turning on the spot first when more
than 50 degrees off its heading (walking meanwhile walked into the corner,
day26): the push's own lateral stick takes up the rest.

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
| day19 (seed 5, closer orbit, kept standing point, deepest first, 0.75 sticks, decaying map, sticky tracking) | sensed | 120 s | tie 6-6 | 40 | 16 shoves, 3 ghosts (17-20 before); poses means 21 and 27 cm |
| day20 (seed 5, leg odometry) | sensed | 120 s | blue 8-4 | 62 | 23 shoves, 19 give-ups (40 before); poses means 34 (two side-wall excursions) and 12 cm |
| day21 (the chamfers as walls) | sensed | 120 s | blue 7-5 | 30 | 11 shoves; the pose jumped 1.2 m four times -- reverted |
| day22 (seed 5, leg odometry, confirmation) | sensed | swept at 90 s | red 12-0 | 44 | 14 shoves, 14 give-ups; poses means 10 and 22 cm, worst 0.36 and 0.82 |
| day23 (seed 5, low objects, the jam rules, the orbit's way chosen once) | sensed | 120 s | red 7-5 | 44 | 10 shoves, 49 "jammed" give-ups (the kerb behind a thing, see above); poses means 6 and 5 cm, worst 0.20 and 0.23 |
| day24 (the slip test, the orbit at 0.40 m from 0.7 m out) | sensed | 120 s | blue 9-3 | 8 | no shoves: 49 "jammed" give-ups, nearly every shove (see above); poses means 7 and 6 cm |
| day25 (the nose jam only, the orbit's hysteresis, no things on a kerb line) | sensed | 120 s | red 8-4 | 44 | 16 shoves, 30 jam give-ups (the peel's slants drove things into the walls); poses means 6 and 5 cm, worst 0.21 and 0.20 |
| day26 (the wall route) | sensed | 120 s | blue 9-3 | 32 | 8 shoves, 33 jam give-ups: the jam cluster read a side wall alongside, and the orbit never reached a goal in a corner's clearance (see above); poses means 7 and 4 cm |
| day27 (seed 5: the jam band, the orbit's goal reachable, the floor slope) | sensed | 120 s | blue 11-1 | 90 | 22 shoves, 21 jam give-ups (each a corner turn on the wall route), 6 ghosts; poses means 7 and 7 cm, worst 0.23 and 0.19 |

Objects shoved deep into a half end up against the far kerb, where the other
crab can only get beside them, not behind: the slanting approaches handle it
at a slower pace. Each shove is 10-15 s from the pick to the back-off.

## Knobs

`RACE_ROOM_X/Y`, `RACE_OBJECTS`, `RACE_MIX`, `RACE_START_X`, `RACE_CAM_DIST/AZ/EL/Y`
on the room; `SWEEP_WALK`, `SWEEP_PUSH`, `SWEEP_BEHIND_M`, `SWEEP_OVER_M`,
`SWEEP_FWD_PER`, `SWEEP_SIDE_PER` on the pilot; `--race-s`, `--objects`,
`--seed`, `--no-hud`, `--no-moves` on the runner.
