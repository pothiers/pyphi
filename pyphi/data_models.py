# see
#  https://github.com/wmayner/pyphi/blob/feature/implicit-tpm/pyphi/_tpm.py
#    ~/Downloads/_tpm.py
#  tpm.py
# Python internals
import itertools
from random import random
from collections.abc import Sequence
from functools import reduce
import operator
# External packages
import networkx as nx
import pandas as pd
import numpy as np
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

class NU_Node():  # NU:: Not Used!!!
    """Node in network. Supports more than just two states but downstream 
    software may be built for only binary nodes. Auto increment node id that 
    will be used as label if one isn't provided on creation.
    """
    _id = 0

    def __init__(self, label=None, numStates=2):
        self._numStates = numStates
        if label is None:
            self._label = Node._id
            Node._id += 1
        else:
            self._label = label 

    @property
    def numStates(self):
        return self._numStates

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
    return itertools.product(*[range(cnt) for (lab,cnt) in node_lut.values()])

# lut, labs, combos = gen_node_lut(['A', 2, ('C', 3), [4, 6]])
# all_state_combos(lut) => [...,(1, 1, 0, 5), (1, 1, 1, 0), (1, 1, 1, 1),...]


class CM():
    """Connectivity Matrix

    cm is Adjacency matrix as python list of lists """
    def __init__(self, cm=None, node_labels=None):  # construct CM
        """Construct CM

        Keyword Args:
            cm (np.ndarray): The numpy array containing adjacency matrix
            node_labels (list): List of nodes where a node can be an 
               int, string, tuple, or list.  If tuple or list it must
               have 2 elements  (<node_label>, <number_of_states_of_node>)
        """        

        if cm is not None:
            node_cnt = len(cm)
            nodes = list(range(node_cnt))
        elif node_labels is not None:
            node_cnt = len(node_labels)
            nodes = node_labels
        else: # default to two nodes
            node_cnt = 2
            nodes = list(range(node_cnt))
        lut, labels,combos = gen_node_lut(nodes)
        self._node_labels = labels
        self._df = pd.DataFrame(data=cm, index=labels, columns=labels)

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
        

            
class TransProb():
    """TRANSition PROBability matrix (and other formats)"""

    
    def __init__(self, in_nodes, out_nodes, probabilities, defaultspn=2):
        """Create and store a Transition Probability Matrix

        This is a directed adjacency matrix from the states of in_nodes 
        to the states of out_nodes.  The number of States Per Node 
        (for all nodes) is given by spn. Thus the number of rows is:
        spn^|in_nodes|.  The number of columns is: spn^|out_nodes|. 
        The probabilities parameter is an 2D array that must this shape.

        in_nodes, out_nodes: list of nodes where a node can be an int, string, 
        tuple, or list.  If tuple or list it must have 2 elements 
        (<node_label>, <number_of_states_of_node>)
        """

        lut, labels,combos = gen_node_lut(in_nodes)
        self.in_node_lut = lut
        self.in_node_labels = labels
        self.num_in_states = combos
        assert combos == probabilities.shape[0], (
            f'dim0[{probabilities.shape[0]}] of probabilities '
            f'must equal {combos} '
            '(prod of state count for all in_nodes)')


        lut, labels, combos = gen_node_lut(out_nodes)
        self.out_node_lut = lut
        self.out_node_labels = labels
        self.num_out_states = combos
        assert combos == probabilities.shape[1], (
            f'dim1[{probabilities.shape[1]}] of probabilities '
            f'must equal {combos} '
            '(prod of state count for all out_nodes)')

        in_states = all_state_combos(self.in_node_lut)
        out_states = all_state_combos(self.out_node_lut)

        self._df = pd.DataFrame(data=probabilities,
                                index=in_states,
                                columns=out_states)
        
    def __len__(self):
        return len(self.in_node_labels) * len(self.out_node_labels)

    def __repr__(self):
        # HEX outut of hash (unique to instance)
        return (f'TransProb({self.__hash__():x}): {self._df.shape} '
                f'in_labels: {self.in_node_labels} '
                f'out_labels: {self.out_node_labels}')

    @property
    def in_nodes(self):
        return self.in_node_labels

    @property
    def out_nodes(self):
        return self.out_node_labels

    @property
    def df(self):
        return self._df

    @property
    def sbs(self):
        """State-By-State form. Rows are for t-1. Columns for t.
        """
        return self._df.to_numpy()

    @property
    def sbn(self):
        """State-By-Node form. 
        """
        return sbs2sbn(self._df.to_numpy())
  
    def graph(self):
        """Return networkx DiGraph with edge.weight=probability.
        """
        return nx.DiGraph(_tpm)

    @property
    def to_legacy(self):
        """Pure numpy representation which ignores node labels."""
        return self._df.to_numpy()

    def from_legacy(self, tpm, labels=None):
        if labels is None:
            labels = list(range(len(tpm)))
        lut, labels,combos = gen_node_lut(labels)
        states = all_state_combos(lut)

        self.in_node_lut = lut.copy()
        self.out_node_lut = lut.copy()        

        self.in_node_labels = labels
        self.out_node_labels = labels

        self._df = pd.DataFrame(data=tpm, index=states, columns=states)
        return self
        

    

class Network():  # Replacement for network.py:Network()
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

    def cm_graph(self):
        """Return networkx DiGraph with edge.weight=probability.
        """
        return nx.DiGraph(_cm.df)

    def tpm_graph(self):
        """Return networkx DiGraph with edge.weight=probability.
        """
        return nx.DiGraph(_tp.df)
    
            
def tryit():
    net = basic_noisy_selfloop_network()
    tpm = net.tpm
    
