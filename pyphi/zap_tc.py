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
import itertools
# External packages
import networkx as nx
import pandas as pd
# Local packages
from . import data_models as dm


class Zaptc():
    """Change the state of a node or nodes and count state changes."""

    def __init__(self, net=None):
        self.net = net
        self.states = dm.States(net=net)
        # Could do large, distributed, zaptc jobs; merge counters afterwards
        self.transition_counter = Counter()
        self.current_state = [0] * len(net)
        self.tpg = None

        # Forced to provide all states in TPM to validate.py
        states = itertools.product(*[range(2) for i in range(len(self.net))])
        hstates = [''.join(f'{s:x}' for s in sv)  for sv in states]
        self.hstates = hstates

    
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
            ss = statestr.replace('0','.')
            print(f'Zap node: {node.label} state: {ss}')
            self.zap_tc(node.label, statestr, time)

    def tpg(self):
        """Transition Probability Graph"""
        total = sum(self.transition_counter.values())
        G = nx.DiGraph()
        G.add_weighted_edges_from(
            [(i,j,c/total) for ((i,j), c) in self.transition_counter.items()],
            weight='p')
        return G

    @property
    def tpm_sbn(self):
        """Transition Probability Matrix (rectangular: States x Nodes).
        Binary nodes only.
        """
        N = len(self.net)        
        instates = set(s1 for s1,s2 in self.transition_counter.keys())

        # Forced to create bigger array than needed because
        # validate.py errors if there isn't one
        # row for all POSSIBLE input states.
        # Processing COULD assume a missing row means zero probability
        # of transition
        #
        #!df = pd.DataFrame(index=instates, columns=self.net.nodes).fillna(0)
        df = pd.DataFrame(index=self.hstates, columns=self.net.nodes).fillna(0)
        #! for s0 in self.hstates:
        #!     df.loc[s0] = [0] * N
            
        for ((s0,s1), count) in self.transition_counter.items():
            cols = [n for c,n in zip(s1,self.net.nodes) if c=='1']
            df.loc[s0,cols] = df.loc[s0,cols] + count

        # return normalized form
        return df.div(df.sum(axis=1), axis=0)
            
