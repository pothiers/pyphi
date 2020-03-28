# see
#  https://github.com/wmayner/pyphi/blob/feature/implicit-tpm/pyphi/_tpm.py
#  tpm.py
import itertools
#
import networkx as nx
import pandas as pd
import pyphi
from pyphi.examples import basic_noisy_selfloop_network

class TPM():
    """Transition Probability Matrix"""

    def __init__(self, tpm, cm=None, node_labels=None, purview_cache=None):
        self._tpm = None # full (N^2 x N^2) form as 2D numpy array

    ###########################
    ### Formats
    def sbn(self):
        """State-By-Node form

        Shape (2x2...x2xN)
        Indexed by 
        """
        pass

    def nbn(self):
        """Node-By-Node form

        Shape (N^2xN^2) where N is number of nodes.  
        Indexed by i and j = tuple containing states of those nodes.
        nbn(i,j) => probability; that the state at time t+1 will be j given
        state at time i is t.
        """
        pass

    def graph(self):
        """Return networkx DiGraph with edge.weight=probability.
        """
        return nx.DiGraph(_tpm)
    

    
    ###########################
    ### Utils
    def _validate(self):
        """Validate input TPM"""
        pass

    
class Node():
    _id = 0

    def __init__(self, label=None, numStates=2):
        self._numStates = numStates
        self._label = label if label else Node._id
        Node._id += 1

    @property
    def numStates(self):
        return self._numStates

    @property
    def label(self):
        return self._label


class Net():  # Replacement for network.py:Network()
    
    def __init__(self, tpm, cm, node_labels=None):
        self._node_labels = list(node_labels) if node_labels else (
            list(range(cm.shape[0])))
        self._tpdf = None # Transition Probabilities DataFrame [tpm]
        self._ncdf = None # Node Connectivity DataFrame [cm]

        self._ncdf = pd.DataFrame(data=cm,
                                  index=self._node_labels,
                                  columns=self._node_labels)

        N = len(self._node_labels)
        states = list(itertools.product("01", repeat=N))
        self._tpdf = pd.DataFrame(tpm, index=states, columns=states)
            
def tryit():
    net = basic_noisy_selfloop_network()
    tpm = net.tpm
    
