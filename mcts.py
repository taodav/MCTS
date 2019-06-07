"""
Loosely based on https://github.com/kudkudak/python-ai-samples/blob/master/AI-UCTAlgorithm/UCT.py
"""

import datetime
import copy
import math
from collections import defaultdict
from random import choice, shuffle


class UCBPolicy(object):
    C = math.sqrt(2)
    """ Class for best child policy based on UCB bound v"""
    def __init__(self, C = math.sqrt(2)):
        self.C = 1 / (2 * math.sqrt(2))  # mozna lepiej rozwiazac
        UCBPolicy.C = self.C

    def setParams(self, C):
        self.C = C

    @staticmethod
    def getScore(n):
        """ Returns UCB1 score for node (not root) """
        return n.getExpectedValue() + UCBPolicy.C*math.sqrt(2*math.log(n.parent.N)/n.N)

    def bestChild(self, node):
        """
            UCT Method: Assumes that all childs are expanded
            Implements given policy
        """
        L = [n.getExpectedValue() + self.C*math.sqrt(2*math.log(node.N)/n.N) for n in node.children]
        return node.children[L.index(max(L))]


class Node:
    def __init__(self, env, action=None, parent_agent=None):
        self.env = env
        self.parent_agent = parent_agent
        self.action = action
        self.actions_remaining = list(range(env.nActions()))
        shuffle(self.actions_remaining)
        self.N = 0
        self.Q = 0.0
        self.parent = None

    def getExpectedValue(self):
        """ returns expected value, if transposition option is on uses dict """
        return self.Q / float(self.N)

    def isFullyExpanded(self):
        return len(self.actions_remaining) == 0

    def isTerminal(self):
        return self.env.inTerminalState()


class MonteCarloAgent(object):
    def __init__(self, environment, best_child_policy=UCBPolicy, **kwargs):
        # Takes an instance of a Board and optionally some keyword
        # arguments.  Initializes the list of game states and the
        # statistics tables.
        self._environment = environment
        self._runtime = datetime.timedelta(seconds=kwargs.get('runtime', 10))
        self._max_depth = kwargs.get('max_depth', 100)
        self.bestChild = best_child_policy().bestChild
        self.Qdict = defaultdict(float)
        self.Ndict = defaultdict(float)

        self._states = []

    def update(self):
        # Takes a game state, and appends it to the history.
        state = self._environment.observe()[0]
        self._states.append(state)

    def get_action(self):
        # Calculate the best move from the
        # current game state and return it.
        root = Node(copy.deepcopy(self._environment), parent_agent=self)
        begin = datetime.datetime.utcnow()
        counter = 0
        while datetime.datetime.utcnow() - begin < self._runtime:
            counter += 1
            v = self.tree_policy(root)
            evaluation = self.defaultPolicy(v)
            self.backup(v, evaluation)

        L = [n.getExpectedValue() for n in root.children]
        return root.children[L.index(max(L))].action

    def tree_policy(self, node):
        while not node.isTerminal():
            if not node.isFullyExpanded():
                return self.expand(node)
            else:
                node = self.bestChild(node)

        return node

    def expand(self, node):
        actions_remaining = node.actions_remaining
        action = node.actions_remaining.pop()

