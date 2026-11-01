ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]

MOVES = {"UP": (1, 0), "RIGHT": (0, 1), "DOWN": (-1, 0), "LEFT": (0, -1)}

PERPENDICULAR = {
    "UP": ("LEFT", "RIGHT"),
    "DOWN": ("LEFT", "RIGHT"),
    "RIGHT": ("UP", "DOWN"),
    "LEFT": ("UP", "DOWN"),
}

ROWS = 3
COLS = 4
WALL = (2, 2)
START = (1, 1)
PLUS_TERMINAL = (3, 4)
MINUS_TERMINAL = (2, 4)


class MDP:
    """A finite MDP M = (S, A, P, R, gamma).

    transition_table maps state -> {action: {successor: probability}}.
    Terminal states have no entry, so they have no actions.
    reward is a function reward(s, a, s2).
    """

    def __init__(self, states, terminals, transition_table, reward, gamma):
        self.states = states
        self.terminals = terminals
        self.transition_table = transition_table
        self.reward = reward
        self.gamma = gamma

    def actions(self, s):
        """Return the actions available in s, in tie-breaking order."""
        return list(self.transition_table.get(s, {}))

    def transitions(self, s, a):
        """Return {successor: probability} for taking a in s."""
        return self.transition_table[s][a]


def move(s, direction, states):
    """Return the cell reached by moving from s in direction.

    A move into the wall or outside the grid leaves the agent in s.
    """
    dr, dc = MOVES[direction]
    s2 = (s[0] + dr, s[1] + dc)
    return s2 if s2 in states else s


def gridworld_reward(s, a, s2):
    """+1 for entering (3, 4), -1 for entering (2, 4), -0.04 otherwise."""
    if s2 == PLUS_TERMINAL:
        return 1.0
    if s2 == MINUS_TERMINAL:
        return -1.0
    return -0.04


def make_gridworld():
    """Build the 3x4 gridworld from Exercise 2."""
    states = [(r, c) for r in range(1, ROWS + 1) for c in range(1, COLS + 1)
              if (r, c) != WALL]
    terminals = {PLUS_TERMINAL, MINUS_TERMINAL}

    transition_table = {}
    for s in states:
        if s in terminals:
            continue
        transition_table[s] = {}
        for a in ACTIONS:
            outcomes = [(a, 0.8)] + [(p, 0.1) for p in PERPENDICULAR[a]]
            probs = {}
            for direction, p in outcomes:
                s2 = move(s, direction, states)
                # Combine outcomes that lead to the same successor cell
                probs[s2] = probs.get(s2, 0.0) + p
            transition_table[s][a] = probs

    return MDP(states, terminals, transition_table, gridworld_reward, 0.9)
