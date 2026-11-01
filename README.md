# Value Iteration and Policy Iteration

Finds the best move in every square of a 3x4 grid world. The grid has a wall, a +1 exit and a -1 exit.

It solves the grid in two ways and prints the value of each square and the best move:

- **Value iteration:** keeps updating square values until they stop changing
- **Policy iteration:** picks moves, scores them, and improves them until they stop changing

## Files

| File | What it does |
|---|---|
| `src/gridworld.py` | Builds the grid world |
| `src/planning.py` | Value iteration and policy iteration |
| `src/main.py` | Runs both and prints the results |
| `tests/test_planning.py` | Tests |

## Run it

No packages needed. Run from this folder:

```bash
python src/main.py
```

Run the tests:

```bash
python -m unittest discover tests
```
