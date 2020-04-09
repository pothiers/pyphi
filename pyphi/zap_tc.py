"""Perturb one node to catalyze transitions. Propagate downstream.

Count the transitions downstream from the perturbation over specified
time interval. 
zaptc("zapticy") = zap_TransitionCounts

For a degree D graph, we expect only D character changes in the
statestr from one state to the other. @@@
"""
# Python release packages
from random import choice
from collections import Counter
# External packages
import networkx as nx
# Local packages
import pyphi.data_models as dm


class Zaptc():
    """Change the state of a node or nodes and count state changes."""

    def __init__(self, net=None):
        self.net = net
        self.states = dm.States(net=net)
        # Could do large, distributed, zaptc jobs; merge counters afterwards
        self.transition_counter = Counter()
        self.current_state = [0] * len(net)
        
    
    def etc(self, node_label, state, time):
        """Effect Transition Count: random state for node my be same as prev."""
        #print(f'etc({node_label}, {state}, {time})')
        if time < 0:
            return

        successors = self.net.successors(node_label)
        state1 = self.states.flip_chars(state, successors)
        self.transition_counter.update([(state,state1)])

        #!diffs = len([x for x,y in zip(state,state1) if x != y]) # NU @@@
        #!print(f'etc; num node-state changes = {diffs}')# NU @@@

        # propagate effect
        for n in successors:
            self.etc(n, state1, time-1)

    def zap_tc(self, node_label, state, time):
        """Count transition from zapped node to successors state"""
        assert type(state) == str
        assert len(state) == len(self.net), (
            f'state({len(state)}) and net({len(self.net)}) must be same size.')

        state0 = self.states.flip_char(state, node_label,  must_change=True)
        successors = self.net.successors(node_label)
        state1 = self.states.flip_chars(state0, successors)
        self.transition_counter.update([(state0,state1)])

        # propagate effect
        for s_node in successors:
            self.etc(s_node, state1, time-1)
    
    def zapall(self, time):
        for node in self.net.nodes:
            # @@@ Should use more random states then num_nodes
            statestr = self.states.gen_random_state()
            print(f'Zap node={node.label} state={statestr}')
            self.zap_tc(node.label, statestr, time)
