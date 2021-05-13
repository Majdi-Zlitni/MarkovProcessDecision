# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*
        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.
          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        iteration = 0

        while (iteration < self.iterations):  # loop for k iterations

            states = self.mdp.getStates()  # get states of mdp for current iteration
            current_iteration_values = util.Counter()  # final values for this iteration

            for i in range(len(states)):  # loop on states

                current_state = states[i]

                actions = self.mdp.getPossibleActions(current_state)  # get possible actions from current state
                q_value = util.Counter()

                if (not self.mdp.isTerminal(current_state)):  # if it's not a terminal state

                    for action in actions:  # loop on the possible actions

                        # get Q_star for each action and store in a dict
                        q_value[action] = self.computeQValueFromValues(current_state, action)

                    current_iteration_values[current_state] = max(q_value.values())

            self.values = current_iteration_values  # apply values to each state in the current iteration
            iteration += 1  # get to the next iteration

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        possible_transitions = self.mdp.getTransitionStatesAndProbs(state, action)

        Q_star = 0

        for i in range(len(possible_transitions)):
            Q_star += possible_transitions[i][1] * (
                        self.mdp.getReward(state, action, possible_transitions[i][0]) + self.discount * self.values[
                    possible_transitions[i][0]])

        return Q_star


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        actions = self.mdp.getPossibleActions(state)

        if (self.mdp.isTerminal(state) or len(actions) == 0):
            return None

        q_values = util.Counter()

        for action in actions:
            q_values[action] = self.computeQValueFromValues(state, action)

        return q_values.argMax()


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)