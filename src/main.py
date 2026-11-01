from gridworld import (COLS, MINUS_TERMINAL, PLUS_TERMINAL, ROWS, WALL,
                       make_gridworld)
from planning import bellman_residual, policy_iteration, value_iteration


def cell_label(cell):
    """Return the marker for the wall or a terminal cell, or None."""
    if cell == WALL:
        return "WALL"
    if cell == PLUS_TERMINAL:
        return "+1 TERM"
    if cell == MINUS_TERMINAL:
        return "-1 TERM"
    return None


def print_grid(title, contents):
    """Print a 3-by-4 grid with row 3 at the top and column 1 on the left."""
    print(title)
    for r in range(ROWS, 0, -1):
        row = []
        for c in range(1, COLS + 1):
            label = cell_label((r, c))
            row.append(f"{label if label else contents[(r, c)]:>9}")
        print(" ".join(row))
    print()


def report(name, mdp, V, policy):
    """Print the value grid, action grid and Bellman residual."""
    print(f"=== {name} ===")
    print_grid("Final state values:",
               {s: f"{v:.4f}" for s, v in V.items()})
    print_grid("Greedy actions:", policy)
    print(f"Maximum Bellman optimality residual: {bellman_residual(mdp, V):.3e}")


def main():
    mdp = make_gridworld()

    V, policy, sweeps = value_iteration(mdp)
    report("Value Iteration", mdp, V, policy)
    print(f"Value-iteration sweeps: {sweeps}")
    print()

    V, policy, rounds, eval_sweeps = policy_iteration(mdp)
    report("Policy Iteration", mdp, V, policy)
    print(f"Policy-improvement rounds: {rounds}")
    print(f"Total policy-evaluation sweeps: {eval_sweeps}")


if __name__ == "__main__":
    main()
