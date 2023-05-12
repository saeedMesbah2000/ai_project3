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
from math import inf
import math

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
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
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        allStates = self.mdp.getStates()
        numberOfIterations = self.iterations
        print("---- this is all states : {}" .format(allStates))
        print("---- this number of iterations : {}" .format(numberOfIterations))
        print("\\\\\\\\\\\ this is counter : {}" .format(util.Counter()))

        for state in allStates:
            self.values[state] = 0

        # print("this is all states values ------before : {}" .format(self.values))

        for i in range(numberOfIterations): 
            tempValues = self.values.copy() 
            for state in allStates:
                #A counter keeps track of counts for each state value
                Qvalues = util.Counter() 
                actions = self.mdp.getPossibleActions(state)
                # print("this is state : {}".format(state))
                for action in actions:
                    Qvalues[action] = self.getQValue(state, action)

                maxValueAction = 0
                bestAction = ""
                for action in Qvalues:
                    if(maxValueAction<=Qvalues[action]):
                        maxValueAction = Qvalues[action]
                        bestAction = action
                print("this is best action : {}" .format(bestAction))
                print("this is max value : {}" .format(maxValueAction))
                print("this is with argMax value : {}" .format(Qvalues[Qvalues.argMax()]))
                print("this is with argMax action : {}" .format(Qvalues.argMax()))
                # tempValues[state] = Qvalues[bestAction]           # this is not working
                # tempValues[state] = maxValueAction                # this is not working
                tempValues[state] = Qvalues[Qvalues.argMax()]
            self.values = tempValues.copy()

        # print("this is all states values -----after : {}" .format(self.values))       


        


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
        nextSteps = self.mdp.getTransitionStatesAndProbs(state, action)
        # print("this is next step : {}" .format(nextSteps))
        Qvalue = 0
        for nextStep in nextSteps:
            nextState = nextStep[0]
            # print("next state : {}" .format(nextState))
            probability = nextStep[1]
            # print("probability : {}" .format(probability))
            Qvalue += (probability * (self.mdp.getReward(state, action, nextState) + (self.discount * self.values[nextState])))

        return Qvalue

        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return ""
        else :
            allActions = self.mdp.getPossibleActions(state)
            maxValue = -inf
            finalAction = ""
            for action in allActions:
                # print("action type : {}" .format(action))
                value = self.computeQValueFromValues(state, action)
                if maxValue<=value or (maxValue == 0 and action == ""):
                    finalAction = action
                    maxValue = value
            return finalAction

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        allStates = self.mdp.getStates()
        numberOfIterations = self.iterations
        self.values = collections.defaultdict(float)
        print("---- this is all states : {}" .format(allStates))
        print("---- this number of iterations : {}" .format(numberOfIterations))

        statesSize = len(allStates)

        for i in range(numberOfIterations):
            state = allStates[i % statesSize]
            if not self.mdp.isTerminal(state):
                action = self.computeActionFromValues(state)
                Qvalue = self.computeQValueFromValues(state, action)
                self.values[state] = Qvalue



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        print("------ final ------")

