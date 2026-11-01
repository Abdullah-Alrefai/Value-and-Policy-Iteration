import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gridworld import MDP, make_gridworld
from planning import bellman_sweep, policy_iteration, value_iteration


class TestPlanning(unittest.TestCase):

    def test_transition_probabilities(self):
        """Probabilities are nonnegative and sum to 1 for every pair."""
        mdp = make_gridworld()
        for s in mdp.states:
            if s in mdp.terminals:
                continue
            for a in mdp.actions(s):
                probs = mdp.transitions(s, a)
                for p in probs.values():
                    self.assertGreaterEqual(p, 0.0)
                self.assertAlmostEqual(sum(probs.values()), 1.0)

    def test_up_from_start(self):
        """UP at (1, 1) gives (2, 1), (1, 1), (1, 2) with 0.8, 0.1, 0.1."""
        probs = make_gridworld().transitions((1, 1), "UP")
        self.assertEqual(set(probs), {(2, 1), (1, 1), (1, 2)})
        self.assertAlmostEqual(probs[(2, 1)], 0.8)
        self.assertAlmostEqual(probs[(1, 1)], 0.1)
        self.assertAlmostEqual(probs[(1, 2)], 0.1)

    def test_toy_value_iteration(self):
        """Toy MDP A -> B -> terminal, rewards 0 then +1, gamma = 0.9."""
        rewards = {("A", "B"): 0.0, ("B", "T"): 1.0}
        toy = MDP(states=["A", "B", "T"], terminals={"T"},
                  transition_table={"A": {"GO": {"B": 1.0}},
                                    "B": {"GO": {"T": 1.0}}},
                  reward=lambda s, a, s2: rewards[(s, s2)], gamma=0.9)

        V0 = {s: 0.0 for s in toy.states}
        V1 = bellman_sweep(toy, V0)
        self.assertAlmostEqual(V1["A"], 0.0)
        self.assertAlmostEqual(V1["B"], 1.0)

        V2 = bellman_sweep(toy, V1)
        self.assertAlmostEqual(V2["A"], 0.9)
        self.assertAlmostEqual(V2["B"], 1.0)

    def test_algorithms_agree(self):
        """Final values agree to within 1e-6 at every nonterminal state."""
        mdp = make_gridworld()
        V_vi = value_iteration(mdp)[0]
        V_pi = policy_iteration(mdp)[0]
        for s in mdp.states:
            if s not in mdp.terminals:
                self.assertLess(abs(V_vi[s] - V_pi[s]), 1e-6)


if __name__ == "__main__":
    unittest.main()
