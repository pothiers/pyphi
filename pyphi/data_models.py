# see:
#  https://github.com/wmayner/pyphi/blob/feature/implicit-tpm/pyphi/_tpm.py
#    e.g. ~/Downloads/_tpm.py
#  tpm.py

# TODO:
#  vitual matrix via func(state) -> state
#  prev, next to iterate through square TPM
#  Example code that runs with network of size: 10^1,10^2,10^3,10^4 (...)
#    High state count.  Long state-vector as DF index.
#    Use generators instead of lists.
#  Non-square CM matrix
#  Doc functional requirements of system (as built vs future)
#  States as HEX; for up to 16 states in compact idx easily converted to decimal
#  DF index by state-vector (state for each node) could use MultiIndex EXCEPT
#    this would not be scalable to high number of nodes.

# Python internals
import itertools
from random import random
from collections.abc import Sequence
from functools import reduce
import operator
import subprocess
# External packages
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import pandas as pd
import numpy as np
from numpy.random import choice as pchoice
# Local packages
import pyphi
from pyphi.examples import basic_noisy_selfloop_network
from pyphi.network import Network as LegacyNetwork
from pyphi.convert import sbs2sbn, sbn2sbs

# DEV notes:
#
# The number of states per node may be greater than 2.
# This is not handled by all the current software
#
# The number of nodes might be much higher than examples.
# Make sure displays (e.g. dataframes) are ok with large number of nodes.
# Perhaps "state" index should be actual base-N number.
# For possibilities, see:
# https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
#
# Often example TPMs are defined with binary Prob yielding sparse matrix.
# Real systems may have continuous Prob yielding nearly fully connected matrix.
#
# Supporting heterogenous states per node may break things.
#
# For network of 4 nodes, the SBN of TPM (multi-dim form) is as:
#   first 4 dims are state of Node(1,2,3,4) at t
#   last dim is prob of Node(1,2,3,4) being ON at t+1
#   ESSENTIALY: 2d[state, node] => prob of node being ON
#   This can only support binary nodes because prob is for single state 
#   (other binary state = 1-prob)



def prod(iterable):
    return reduce(operator.mul, iterable, 1)


# The binding between states and the nodes they represent must be maintained.
# How???
def example_causation_tpm():
    # Non-square TPM
    # Input (binary) nodes: A,B,C
    # Output (binary) node: D
    CBA_states = [(0,0,0),
                  (0,0,1),
                  (0,1,0),
                  (0,1,1),
                  (1,0,0),
                  (1,0,1),
                  (1,1,0),
                  (1,1,1),  ]
    D_states = [0, 1]
    # Following are just random probabilities
    probs = np.array([[0.68893508, 0.40176953],
                      [0.44095309, 0.84035902],
                      [0.09975545, 0.58225631],
                      [0.86475645, 0.18650795],
                      [0.50721989, 0.86299773],
                      [0.62045787, 0.90525779],
                      [0.88270204, 0.46225991],
                      [0.51548114, 0.89159624]])
    df = pd.DataFrame(data=probs, index=CBA_states, columns=D_states)
    return df


class Node():  # NU:: Not Used!!!
    """Node in network. Supports more than just two states but downstream 
    software may be built for only binary nodes. Auto increment node id that 
    will be used as label if one isn't provided on creation.
    """
    _id = 0

    def __init__(self, label=None, numStates=2):
        self.state = 0
        self.states = list(range(numStates))
        if label is None:
            self._label = Node._id
            Node._id += 1
        else:
            self._label = label 

    @property
    def numStates(self):
        return len(self.states)

    @property
    def state(self):
        return self.state

    @property
    def states(self):
        return self.states

    @property
    def label(self):
        return self._label




def seq2(val):
    """True iff val is a non-string sequence of length 2"""
    return ((not isinstance(val,str))
            and isinstance(val,Sequence)
            and (len(val)==2))

def gen_node_lut(nodes):
    lab_state_list  = map(lambda x: x if seq2(x) else (x,2), nodes)
    # lut[index] => (label,state_cnt) e.g. {0: ('A', 2),  1: ('C', 3)}
    lut = dict(enumerate(lab_state_list))
    labels = [str(lab) for (lab,cnt) in lut.values()]
    num_combo_states = prod([cnt for (lab,cnt) in lut.values()])
    return lut, labels, num_combo_states

def all_state_combos(node_lut):
    """Return generator for states as HEX."""
    # node_lut[index] => (label,state_cnt) e.g. {0: ('A', 2),  1: ('C', 3)}
    states = itertools.product(*[range(cnt) for (lab,cnt) in node_lut.values()])
    hstates = (''.join(hex(s)[2:] for s in sv) for sv in states)
    return hstates

# lut, labs, combos = gen_node_lut(['A', 2, ('C', 3), [4, 6]])
# all_state_combos(lut) => [...,(1, 1, 0, 5), (1, 1, 1, 0), (1, 1, 1, 1),...]


class CM():  # @@@ Allow this to be non-square (e.g. feed-forward)
    """Connectivity Matrix"""
    # for node names
    nn = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz') 

    def __init__(self, am=None, node_labels=None):  # construct CM
        """Construct CM

        Keyword Args:
            am (np.ndarray): The numpy array containing adjacency matrix
            node_labels (list): List of nodes where a node can be an 
               int, string, tuple, or list.  If tuple or list it must
               have 2 elements  (<node_label>, <number_of_states_of_node>)
        """        

        if am is None:
            am = np.zeros((2,2))
        node_cnt = len(am)
        labels = list(range(node_cnt))
        if node_labels is not None:
            assert node_cnt == len(node_labels), (
                f'Number of labels {len(node_labels)} must be same as '
                f'len(cm) {node_cnt}')
            labels = node_labels
        self._node_labels = labels
        self._df = pd.DataFrame(data=am, index=labels, columns=labels)

    def __len__(self):
        return len(self._node_labels)
    
    def __repr__(self):
        # HEX outut of hash (unique to instance)
        return (f'CM({self.__hash__():x}): {self._df.shape} '
                f'labels: {self._node_labels}')

    @property
    def connectivity_matrix(self):
        return self._df
    df=connectivity_matrix

    @property
    def node_labels(self):
        return self._node_labels
    labels=node_labels
    
    @property
    def to_legacy(self):
        """Pure numpy representation which ignores node labels."""
        return self._df.to_numpy()

    def from_legacy(self, cm, labels=None):
        if labels is None:
            labels = list(range(len(cm)))
        self._node_labels = labels
        self._df = pd.DataFrame(data=cm, index=labels, columns=labels)
        return self
        
    def graph(self, pngfile=None):
        """Return networkx DiGraph.    """
        G = nx.DiGraph(self._df)
        if pngfile is not None:
            dotfile = pngfile + ".dot"
            write_dot(G, dotfile)
            cmd = (f'dot -Tpng -o{pngfile} {dotfile} ')
            with open(pngfile,'w') as f:
                subprocess.check_output(cmd, shell=True)
        return G


    
class TransProb(): # @@@ This could be "virtual". Func instead of matrix.
    """TRANSition PROBability matrix (and other formats)"""
    
    def __init__(self, in_nodes=None, out_nodes=None, probabilities=None,
                 defaultspn=2):
        """Create and store a Transition Probability Matrix

        This is a directed adjacency matrix from the states of in_nodes 
        to the states of out_nodes.  The number of States Per Node 
        (for all nodes) is given by spn. Thus the number of rows is:
        spn^|in_nodes|.  The number of columns is: spn^|out_nodes|. 
        The probabilities parameter is an 2D array that must this shape.
        If not given, probabilities defaults to zeros of correct shape.

        in_nodes, out_nodes: list of nodes where a node can be an int, string, 
        tuple, or list.  If tuple or list it must have 2 elements 
        (<node_label>, <number_of_states_of_node>)
        """
        if in_nodes is None:
            in_nodes = range(2)
        if out_nodes is None:
            out_nodes = range(2)
        (self.in_node_lut,
         self.in_node_labels,
         self.num_in_states) = gen_node_lut(in_nodes)

        (self.out_node_lut,
         self.out_node_labels,
         self.num_out_states) = gen_node_lut(out_nodes)

        if probabilities is None:
            probabilities = np.zeros((self.num_in_states, self.num_out_states))
        assert self.num_in_states == probabilities.shape[0], (
            f'dim0[{probabilities.shape[0]}] of probabilities '
            f'must equal {self.num_in_states} '
            '(prod of state count for all in_nodes)')
        assert self.num_out_states == probabilities.shape[1], (
            f'dim1[{probabilities.shape[1]}] of probabilities '
            f'must equal {self.num_out_states} '
            '(prod of state count for all out_nodes)')

        self.istates = list(all_state_combos(self.in_node_lut))
        self.ostates = list(all_state_combos(self.out_node_lut))
        self._current = self.istates[0] # default start state
        
        self._df = pd.DataFrame(data=probabilities,
                                index=self.istates,
                                columns=self.ostates)
        
    def __len__(self):
        return len(self.in_node_labels) * len(self.out_node_labels)

    def __repr__(self):
        # HEX outut of hash (unique to instance)
        return (f'TransProb({self.__hash__():x}): {self._df.shape} '
                f'in_labels: {self.in_node_labels} '
                f'out_labels: {self.out_node_labels}')

    @property
    def state(self):
        return self._current

    def set_state(self, state):
        self._current = state

    def _state_probs(self, state):
        # Allow probabilities to not sum to one. Better to fix on input! @@@
        def norm(raw):
            return [float(i)/sum(raw) for i in raw]
        return norm(self._df[state])
        
    def next_state(self, current=None):
        """Return a valid state at t+1 given current state at t.
        The next state is chosen from list of possibles to fit the distribution
        of out states.
        """
        state = self._current if current is None else current
        probs = self._state_probs(state)
        nextstate = pchoice(list(self._df.columns), 1, p=probs)[0]
        self._current = nextstate
        return nextstate

    @property
    def in_nodes(self):
        return self.in_node_labels

    @property
    def in_states(self):
        return list(self._df.index)

    @property
    def out_nodes(self):
        return self.out_node_labels
    
    @property
    def out_states(self):
        return list(self._df.columns)

    @property
    def df(self):
        return self._df

    @property
    def sbs(self):
        """State-By-State form. Rows are for t-1. Columns for t.
        """
        return self._df.to_numpy()

    @property
    def legacy_sbn(self):
        """State-By-Node form. 
        """
        return sbs2sbn(self._df.to_numpy())

    # For network of 4 nodes, the SBN of TPM (multi-dim form) is as:
    #   first 4 dims are state of Node(1,2,3,4) at t
    #   last dim is prob of Node(1,2,3,4) being ON at t+1
    #   ESSENTIALY: 2d[state, node] => prob of node being ON
    #   This can only support binary nodes because prob is for single state 
    #   (other binary state = 1-prob)
    @property
    def sbn(self):
        """State-By-Node form. 
        """
        nodes = self.in_node_labels + self.in_node_labels
        (lut, labels, num_states) = gen_node_lut(nodes)
        # Make sure all nodes are binary
        assert all([(cnt == 2) for (lab,cnt) in lut.values()])

        states = all_state_combos(lut)
        # see convert.py "little-endian"
        revstates = (reversed(s) for s in states) # generator
  
    def graph(self, min_prob=0.02, pngfile=None):
        """Return networkx DiGraph with edge.weight=probability.
        """
        df = self._df
        dod = dict()
        for i in self._df.index:
            d = dict()
            for j in df.columns:
                prob = df[j][i]
                if prob > min_prob:
                    d[j] = {'label': f'{int(100*round(prob,2))}'}
            dod[i] = d
        G = nx.DiGraph(dod)
        if pngfile is not None:
            dotfile = pngfile + ".dot"
            write_dot(G, dotfile)
            cmd = (f'dot -Tpng -o{pngfile} {dotfile} ')
            with open(pngfile,'w') as f:
                subprocess.check_output(cmd, shell=True)
        return G

        G = nx.DiGraph(self._df)
        if pngfile is not None:
            dotfile = pngfile + ".dot"
            write_dot(G, dotfile)
            cmd = (f'dot -Tpng -o{pngfile} {dotfile} ')
            with open(pngfile,'w') as f:
                subprocess.check_output(cmd, shell=True)
        return G

    @property
    def to_legacy(self):
        """Pure numpy representation which ignores node labels."""
        return self._df.to_numpy()

    def from_legacy(self, legacy_tpm, labels=None):
        """Overwrite content of self to make this TPM contain legacy tpm.
        """
        self. __init__(in_nodes=labels, out_nodes=labels,
                       probabilities=legacy_tpm,
                       defaultspn=2)
        return self
        

    


class Network():  
    """A network of nodes.

    Pandas baised matrices hold indices appropriate for each matrix.
    """
    
    ### TPM: def __init__(self, in_nodes, out_nodes, probabilities, spn=2)
    def __init__(self, tp=None, cm=None):
        """Create Network
        
        Keyword Args:
           tp (|TransProb|): Transition Probability container
           cm (|CM|): Connectivity Matrix container
        """
        self._tp = tp  # Transition Probabilities DataFrame [tpm]
        self._cm = cm  # Node Connectivity DataFrame [cm]
        self._node_labels = None  # Node Labels ordered same as in States

        if cm is None:
            cm = CM()
            self._cm = cm
        self._node_labels = cm.node_labels

    def __repr__(self):
        # HEX outut of hash (unique to instance)
        return (f'Network({self.__hash__():x}): '
                f'node_connectivity: {self._cm.df.shape}, '
                f'transitions: {self._tp.df.shape}')

    @property
    def node_labels(self):
        return self._node_labels

    @property
    def tpm(self):
        return self._tp.df

    @property
    def cm(self):
        return self._cm.df
        
    def from_legacy(self, legacyNetwork):
        labels = list(legacyNetwork.node_labels)
        self._cm = CM(legacyNetwork.cm, labels)
        ltpm = sbn2sbs(legacyNetwork.tpm)
        self._tp = TransProb(labels, labels, ltpm, defaultspn=2) # binary nodes
        return self

    def to_legacy(self):
        """Return old-style version of network for use in legacy functions.
        """
        tpm = sbs2sbn(self._tp.df.to_numpy())
        cm = self._cm.df.to_numpy()
        labels = self._node_labels
        return LegacyNetwork(tpm, cm=cm, node_labels=labels, purview_cache=None)

    def cm_graph(self, pngfile=None):
        """Return networkx DiGraph with edge.weight=probability.
        """
        return self._cm.graph(pngfile)

    def tpm_graph(self, pngfile=None):
        """Return networkx DiGraph with edge.weight=probability.
        """
        return self._tp.graph(pngfile)
    
            
def tryit():
    net = basic_noisy_selfloop_network()
    tpm = net.tpm
    
