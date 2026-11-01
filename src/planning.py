TIE_TOLERANCE = 1e-12
VI_THRESHOLD = 1e-8
EVAL_THRESHOLD = 1e-10


def q_value(mdp, V, s, a):
    """Q(s, a) = sum over s2 of P(s2 | s, a) * [R(s, a, s2) + gamma * V(s2)]."""
    return sum(p * (mdp.reward(s, a, s2) + mdp.gamma * V[s2])
               for s2, p in mdp.transitions(s, a).items())


def greedy_action(mdp, V, s):
    """Return the action with the highest Q(s, a).

    Actions within TIE_TOLERANCE of each other are tied, and the tie goes
    to the action listed first (UP, RIGHT, DOWN, LEFT).
    """
    best_action, best_q = None, None
    for a in mdp.actions(s):
        q = q_value(mdp, V, s, a)
        if best_q is None or q > best_q + TIE_TOLERANCE:
            best_action, best_q = a, q
    return best_action


def bellman_sweep(mdp, V):
    """Return V_(k+1) from V_k with one synchronous Bellman optimality backup.

    The result is a new dictionary; terminal states keep value zero.
    """
    V_next = {}
    for s in mdp.states:
        if s in mdp.terminals:
            V_next[s] = 0.0
        else:
            V_next[s] = max(q_value(mdp, V, s, a) for a in mdp.actions(s))
    return V_next


def extract_policy(mdp, V):
    """Return the greedy policy for V at every nonterminal state."""
    return {s: greedy_action(mdp, V, s) for s in mdp.states
            if s not in mdp.terminals}


def value_iteration(mdp):
    """Synchronous value iteration starting from all values zero.

    Returns (V, policy, sweeps).
    """
    V = {s: 0.0 for s in mdp.states}
    sweeps = 0
    while True:
        V_next = bellman_sweep(mdp, V)
        sweeps += 1
        delta = max(abs(V_next[s] - V[s]) for s in mdp.states)
        V = V_next
        if delta < VI_THRESHOLD:
            break
    return V, extract_policy(mdp, V), sweeps


def policy_evaluation(mdp, policy, V):
    """Iteratively evaluate a fixed policy, starting from V.

    Returns (V, sweeps) once the maximum value change is below EVAL_THRESHOLD.
    """
    sweeps = 0
    while True:
        V_next = {}
        for s in mdp.states:
            if s in mdp.terminals:
                V_next[s] = 0.0
            else:
                V_next[s] = q_value(mdp, V, s, policy[s])
        sweeps += 1
        delta = max(abs(V_next[s] - V[s]) for s in mdp.states)
        V = V_next
        if delta < EVAL_THRESHOLD:
            return V, sweeps


def policy_iteration(mdp):
    """Policy iteration starting from UP in every nonterminal state.

    Returns (V, policy, improvement_rounds, total_eval_sweeps).
    """
    policy = {s: "UP" for s in mdp.states if s not in mdp.terminals}
    V = {s: 0.0 for s in mdp.states}
    rounds = 0
    total_sweeps = 0
    while True:
        V, sweeps = policy_evaluation(mdp, policy, V)
        total_sweeps += sweeps
        new_policy = extract_policy(mdp, V)
        rounds += 1
        if new_policy == policy:
            break
        policy = new_policy
    return V, policy, rounds, total_sweeps


def bellman_residual(mdp, V):
    """Return max over nonterminal s of |max_a Q(s, a) - V(s)|."""
    return max(abs(max(q_value(mdp, V, s, a) for a in mdp.actions(s)) - V[s])
               for s in mdp.states if s not in mdp.terminals)
