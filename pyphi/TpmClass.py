# see
#  https://github.com/wmayner/pyphi/blob/feature/implicit-tpm/pyphi/_tpm.py
#    ~/Downloads/_tpm.py
#  tpm.py
# Python internals
import itertools
from random import random
# External packages
import networkx as nx
import pandas as pd
import numpy as np
# Local packages
import pyphi
from pyphi.examples import basic_noisy_selfloop_network
from pyphi.network import Network as LegacyNetwork

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



# The binding betwee states and the nodes they represent must be maintained.
# How???
def causation_tpm():
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


  
class TransProb():
    """TRANSition PROBability matrix (and other formats)"""
    
    def __init__(self, in_nodes, out_nodes, probabilities, spn=2):
        """Create and store a Transition Probability Matrix

        This is an directed adjacency matrix from the states of in_nodes 
        to the states of out_nodes.  The number of States Per Node 
        (for all nodes) is given by spn. Thus the number of rows is:
        spn^|in_nodes|.  The number of columns is: spn^|out_nodes|. 
        The probabilities parameter is an 2D array that must this shape.
        """
        
        assert spn**len(in_nodes) == probabilities.shape[0], (
            f'spn**len(in_nodes)[{spn**len(in_nodes)}] must match '
            f'dim0[{probabilities.shape[0]}] of probabilities')
        assert spn**len(out_nodes) == probabilities.shape[1], (
            f'spn**len(out_nodes)[{spn**len(out_nodes)}] must match '
            f'dim1[{probabilities.shape[1]}] of probabilities')
        
        # This is strictly Binary nodes.  Support non-homogenous N-state nodes?
        self._in_nodes = in_nodes
        self._out_nodes = out_nodes
        nstates = list(range(spn))
        in_states = list(itertools.product(nstates, repeat=len(in_nodes)))
        out_states = list(itertools.product(nstates, repeat=len(out_nodes)))
        self._df = pd.DataFrame(data=probabilities,
                                index=in_states, columns=out_states)
        
        
    @property
    def in_nodes(self):
        return self._in_nodes

    @property
    def out_nodes(self):
        return self._out_nodes

    @property
    def df(self):
        return self._df

    def graph(self):
        """Return networkx DiGraph with edge.weight=probability.
        """
        return nx.DiGraph(_tpm)

    
class Node():
    """Node in network. Supports more than just two states but downstream 
    software may be built for only binary nodes. Auto increment node id that 
    will be used as label if one isn't provided on creation.
    """
    _id = 0

    def __init__(self, label=None, numStates=2):
        self._numStates = numStates
        if label == None:
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


class Network():  # Replacement for network.py:Network()
    """A network of nodes.

    Pandas baised matrices hold indices appropriate for each matrix.
    """
    
    #def __init__(self, tpm, cm, node_labels=None):
    def __init__(self, network):
        """Create from old network class"""
        tpm = pyphi.convert.sbn2sbs(network.tpm)
        cm = network.cm
        node_labels=network.node_labels
        
        self._ncdf = None # Node Connectivity DataFrame [cm]
        self._tpdf = None # Transition Probabilities DataFrame [tpm]
        self._node_labels = list(node_labels) if node_labels else (
            list(range(cm.shape[0])))
        self._ncdf = pd.DataFrame(data=cm,
                                  index=self._node_labels,
                                  columns=self._node_labels)

        N = len(self._node_labels)
        states = list(itertools.product((0,1), repeat=N))
        self._tpdf = pd.DataFrame(tpm, index=states, columns=states)

    def legacy(self):
        """Return old-style version of network for use in legacy functions.
        """
        tpm = pyphi.convert.sbs2sbn(self._tpdf.to_numpy())
        cm = self._ncdf.to_numpy()
        labels = self._node_labels
        return LegacyNetwork(tpm, cm=cm, node_labels=labels, purview_cache=None)

            
def tryit():
    net = basic_noisy_selfloop_network()
    tpm = net.tpm
    
