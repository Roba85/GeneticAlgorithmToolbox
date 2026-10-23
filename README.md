# GeneticAlgorithmToolbox

Python program for soft computing assignment.

## CVRP (Part 2)

### Quick start

```bash
python -m cvrp.main
python -m cvrp.main --quiet
python -m cvrp.main --help
```

The default instance is the [example from the assignment](https://developers.google.com/optimization/routing/cvrp#cvrp_example).

### Options

| Flag | Default | Description |
|---|---|---|
| `--instance PATH` | assignment example | JSON instance file |
| `--pop-size N` | 200 | population size (must be even) |
| `--generations N` | 1000 | maximum number of generations |
| `--seed N` | 42 | RNG seed; the same seed reproduces the run |
| `--selection {tournament,roulette}` | tournament | parent selection method |
| `--mutation {inversion,swap}` | inversion | mutation operator |
| `--replacement {elitist,generational}` | elitist | replacement strategy |
| `--cx-rate P` | 0.9 | order-crossover probability |
| `--mut-rate P` | 0.3 | mutation probability |
| `--tournament-size N` | 3 | candidates per tournament draw |
| `--patience N` | 200 | stop after N generations without improvement |
| `--log-every N` | 10 | print statistics every N generations |
| `--quiet` | off | print only the solution |

### Examples

```bash
# assignment example with the default configuration
python -m cvrp.main --quiet

# a different combination of operators
python -m cvrp.main --selection roulette --mutation swap --replacement generational

# short run, useful to see the search in progress
python -m cvrp.main --pop-size 50 --generations 20

# reproducible run
python -m cvrp.main --quiet --seed 7

# load another instance
python -m cvrp.main --instance path/to/instance.json
```

### Expected output

```
Objective: 6208
Route for vehicle 0:
 0 Load(0) ->  10 Load(2) ->  16 Load(10) ->  14 Load(14) ->  9 Load(15) ->  0 Load(15)
Distance of the route: 1552m
Load of the route: 15

Route for vehicle 1:
 ...

Total Distance of all routes: 6208m
Total Load of all routes: 60
```

With the default settings the program reaches the known optimum (6208) in about half a second.
