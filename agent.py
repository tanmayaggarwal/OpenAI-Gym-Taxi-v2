import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 1.0

    def epsilon_greedy_probs(self, Q_s):
        # obtains the action probabilities corresponding to epsilon-greedy policy
        decay_rate = 0.99999
        epsilon = max(self.epsilon * decay_rate, 0.01)
        self.epsilon = epsilon

        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)

        return policy_s

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # get epsilon-greedy action probabilities
        policy_s = self.epsilon_greedy_probs(self.Q[state])
        # pick action A
        return np.random.choice(self.nA, p = policy_s)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        gamma = 1
        alpha = 0.03

        if not done:
            # pick next action A'
            next_action = self.select_action(next_state)
            # update TD estimate of Q
            self.Q[state][action] += alpha * (reward + (self.Q[next_state][next_action] * gamma) - self.Q[state][action])

        if done:
            # update TD estimate of Q
            self.Q[state][action] += alpha * (reward + (0 * gamma) - self.Q[state][action])

        return self.Q[state][action]
