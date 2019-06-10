"""
Loosely based on https://github.com/kudkudak/python-ai-samples/blob/master/AI-UCTAlgorithm/UCT.py
Read http://mcts.ai/about/index.html for a good description of MCTS
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
        self.children = []

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
        self._gamma = kwargs.get('gamma', 0.99)
        self.bestChild = best_child_policy().bestChild
        self.Qdict = defaultdict(float)
        self.Ndict = defaultdict(float)

        self._states = []

    def get_action(self):
        # Calculate the best move from the
        # current game state and return it.
        root = Node(copy.deepcopy(self._environment), parent_agent=self)
        begin = datetime.datetime.utcnow()
        counter = 0
        while datetime.datetime.utcnow() - begin < self._runtime:
            counter += 1
            leaf = self.select_leaf(root)
            q = self.simulate(leaf)
            self.backup(leaf, q)

        L = [n.getExpectedValue() for n in root.children]
        return root.children[L.index(max(L))].action

    def pick_move_policy(self, env):
        # Policy for simulation, currently set at random
        return choice(list(range(env.nActions)))

    def simulate(self, node):
        """Simulate from the current node"""
        copy_env = copy.deepcopy(node._environment)
        simulation_depth = 0
        rewards = []
        while not copy_env.isTerminalState() and simulation_depth <= self._max_depth:
            action = self.pick_move_policy(copy_env)
            reward = copy_env.act(action)
            rewards.append(reward)
            simulation_depth += 1

        discounted_rewards = [(self._gamma ** i) * r for i, r in zip(range(len(rewards)), rewards)]
        return sum(discounted_rewards)

    def select_leaf(self, node):
        """
        Find the leaf to expand
        :param node: node to start from
        :return: child leaf or terminal node
        """
        while not node.isTerminal():
            if not node.isFullyExpanded():
                return self.expand(node)
            else:
                node = self.bestChild(node)

        return node

    def expand(self, node):
        """
        Expands node by one random step.
        :param node: Node to expand
        :return: child node
        """
        action = node.actions_remaining.pop()
        new_env = copy.deepcopy(node._environment)
        new_env.act(action)
        child_node = Node(new_env, action=action, parent_agent=self)
        child_node.parent = node
        node.children.append(child_node)
        return child_node

    def backup(self, node, q):
        """
        Backup and add all the new q values to the nodes, from the leaf upwards.
        :param node:
        :param q:
        :return:
        """
        while node != None:
            node.Q += q
            node.N += 1
            node = node.parent


