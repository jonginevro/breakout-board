# Fantasy Football Breakout Board

A live tool that ranks NFL skill players by breakout potential: the chance their fantasy value is about to spike, not by how good they already are.

It's built to avoid the two traps:

- "Every rookie looks elite" - Opportunity, catalyst and efficiency are combined with a geometric mean, so a player must score on all three.
- "The breakout is unfeasible"- The catalyst factor is anchored to concrete signals (vacated touches, depth-chart rank, draft capital, age window), and the value term rewards the cheap-but-on-the-radar sweet spot rather than totally off-radar dart throws.

## The model

For each player, four factors are computed and normalized within his position:

| Factor | Question | Main inputs |
|---|---|---|
| Opportunity | How much room is there for volume to grow? | team touches vacated by departed players, personal usage headroom, depth-chart gap |
| Catalyst | Will that room open for him? | draft capital, age window, 2nd-year leap, depth proximity, your manual notes |
| Efficiency | Can he convert volume to points? | prior per-touch production, usage quality (WOPR/target share/EPA), talent prior |
| Suppression | Is he cheap but still relevant? | Sleeper market value, scored on a "sweet spot" curve |

```
core = geomean(opportunity, catalyst, efficiency) ** shape
value = value_floor + (1 - value_floor) * suppression
score = 100 * core * value * momentum
```

## Data

- nflverse: release data: weekly stats, rosters, depth charts, draft picks
- Sleeper API: market value + live "trending adds" for the
  in-season momentum signal.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./run.sh
```

## Tuning

- `config/weights.yaml`: every weight, the value sweet-spot, age windows, and the momentum boost. Change a number, rerun.
- `config/manual_catalysts.yaml`: the human-judgment signals stats can't see: new coaching staffs, scheme changes, an aging starter ahead, camp buzz.