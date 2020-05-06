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


# InstanceVars: net, states, transition_counter, hstates
class Zaptc():
    """Change the state of a node or nodes and count state changes.

ZAPTC EXPERIMENT:

Zaptc ("zapticy") attempts to get useful counts of transitions to be
used to create a TPM.  It does this by walking the network
(initialized to a specific state) for a specified number of time
steps. It does this without knowing anything about how a node's state
is affected by its input (no knowledge of the node's mechanism). In
terms of cause-effect, it trades definitive state changes for what
MIGHT happen for a random and unknown mechanism. During the network
walk, the state of a downstream node may change iff one of its
upstream nodes changes state. Downstream state changes are stochastic.

The result of a full zaptc experiment is a table indexed by (in-state,
out-state) with a value of the number of times that state transition
was encountered in the walk. One walk is done for each of the nodes in
the system. At the start of the walk, a random system state is chosen
and the state of the selected node is changed.

The transition table is normalized into a TPM suitable for the rest of
calculation.
"""

    def __init__(self, net=None):
        self.net = net
        self.states = dm.States(net=net)
        # Could do large, distributed, zaptc jobs; merge counters afterwards
        self.transition_counter = Counter()

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
    
    def zapall(self, time, verbose=False):
        print(f'Gathering system state transition counts (time={time}) ...')
        for node in self.net.nodes:
            # @@@ Should use more random states then num_nodes
            statestr = self.states.gen_random_state()
            ss = statestr.replace('0','.')
            if verbose:
                print(f'Zap node: {node.label} state: {ss}')
            self.zap_tc(node.label, statestr, time)
        print('DONE')
        
    def tpg(self):
        """Transition Probability Graph"""
        total = sum(self.transition_counter.values())
        G = nx.DiGraph()
        G.add_weighted_edges_from(
            [(i,j,c/total) for ((i,j), c) in self.transition_counter.items()],
            weight='p')
        return G

    def tpm_sbn(self, pad=False):
        """Transition Probability Matrix (rectangular: States x Nodes).
        Binary nodes only. DataFrame type.
        """
        N = len(self.net)        
        instates = set(s1 for s1,s2 in self.transition_counter.keys())
    
        df = pd.DataFrame(index=instates, columns=self.net.nodes).fillna(0)

        # Forced to create bigger array than needed because
        # validate.py errors if there isn't one
        # row for all POSSIBLE input states.
        # Processing COULD assume a missing row means zero probability
        # of transition.
        # pyphi.validate.py:state_reachable(subsystem) complains
        # if a row contains all zeros. It says state cannot be reached even
        # though given state is index of DF. Forced me to fill with 0.5
        # to satisfy state_reachable()
        if pad:
            df = pd.DataFrame(index=self.hstates,
                              columns=self.net.nodes).fillna(0.5) #@@@

        for ((s0,s1), count) in self.transition_counter.items():
            cols = [n for c,n in zip(s1,self.net.nodes) if c=='1']
            df.loc[s0,cols] = df.loc[s0,cols] + count

        # return normalized form

        dfnorm = df.div(df.sum(axis=1), axis=0)
        return dfnorm.fillna(0)
    
