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
#
# see: pyphi-design-poth.org

# Python internals
import itertools
from random import random
from collections.abc import Sequence
from functools import reduce
import operator
import subprocess
from random import choice
import re
# External packages
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
from networkx.drawing.nx_pydot import pydot_layout
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


#!
#!    @property
#!    def label(self):
#!        return self._label

def seq2(val):
    """True iff val is a non-string sequence of length 2"""
    return ((not isinstance(val,str))
            and isinstance(val,Sequence)
            and (len(val)==2))

# lut, labs, combos = gen_node_lut(['A', 2, ('C', 3), [4, 6]])
# all_state_combos(lut) => [...,(1, 1, 0, 5), (1, 1, 1, 0), (1, 1, 1, 1),...]
def gen_node_lut(nodes, dspn=2):
    """Return LookUp table to get node Label, StateCnt given Int.

    `nodes` can be string, int, list, tuple.
    If tuple or list it must have 2 elements (<label>, <number_states>).
    """
    lab_state_list  = map(lambda x: x if seq2(x) else (x,dspn), nodes)
    # lut[index] => (label,state_cnt) e.g. {0: ('A', 2),  1: ('C', 3)}
    lut = dict(enumerate(lab_state_list))
    labels = [f'{lab}' for (lab,cnt) in lut.values()]
    num_combo_states = prod([cnt for (lab,cnt) in lut.values()])
    return lut, labels, num_combo_states

def all_state_combos(node_lut):
    """Return generator for states as HEX."""
    # node_lut[index] => (label,state_cnt) e.g. {0: ('A', 2),  1: ('C', 3)}
    states = itertools.product(*[range(cnt) for (lab,cnt) in node_lut.values()])
    hstates = (''.join(hex(s)[2:] for s in sv) for sv in states)
    return hstates


    


class CM():  # @@@ Allow this to be non-square (e.g. feed-forward)
    """Connectivity Matrix"""

    def __init__(self, am=None, in_nodes=None, out_nodes=None, nodes=None):
        """Construct CM

        Keyword Args:
            am (np.ndarray, lol): The numpy array containing adjacency matrix
            in/out_nodes (list): List of nodes where a node can be an 
            int, string, tuple, or list.  If tuple or list it must
            have 2 elements  (<node_label>, <number_of_states_of_node>)
            nodes (list): Convenience for setting both in/out_nodes to same val.
        """
        
        nn = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
        # node_lut[index] => (label,state_cnt) e.g. {0: ('A', 2),  1: ('C', 3)}

        if am is None:
            am = np.zeros((2,2))
        in_cnt,out_cnt = np.array(am).shape
        if nodes:
            in_nodes = nodes
            out_nodes = nodes

        if in_nodes is None:
            in_nodes = [nn[i] for i in range(in_cnt)]
        else:
            assert in_cnt == len(in_nodes), (
                f'Number of in_nodes {len(in_nodes)} must be same as '
                f'rows of am {in_cnt}')
        (self.in_node_lut , self.in_node_labels, S) = gen_node_lut(in_nodes)


        if out_nodes is None:
            out_nodes = [nn[i] for i in range(out_cnt)]
        else:
            assert out_cnt == len(out_nodes), (
                f'Number of out_nodes {len(out_nodes)} must be same as '
                f'rows of am {out_cnt}')
        (self.out_node_lut, self.out_node_labels, S) = gen_node_lut(out_nodes)

        self.df = pd.DataFrame(data=am,
                               index=self.in_node_labels,
                               columns=self.out_node_labels)

    def __len__(self):
        return len(self.in_node_labels + self.out_node_labels)
    
    def __repr__(self):
        # HEX outut of hash (unique to instance)
        return (f'CM({self.__hash__():x}): {self.df.shape} '
                f'labels: {self.in_node_labels}, {self.out_node_labels}')

    #!@property
    #!def connectivity_matrix(self):
    #!    return self._df
    #!df=connectivity_matrix

    @property
    def in_nodes(self):
        return self.in_node_labels
    
    @property
    def out_nodes(self):
        return self.out_node_labels
    
    @property
    def to_legacy(self):
        """Pure numpy representation which ignores node labels."""
        return self.df.to_numpy()

    def from_legacy(self, cm, labels=None):
        if labels is None:
            labels = list(range(len(cm)))
        self.in_node_labels = labels
        self.out_node_labels = labels
        self.df = pd.DataFrame(data=cm, index=labels, columns=labels)
        return self
        
    def graph(self, pngfile=None):
        """Return networkx DiGraph.    """
        G = nx.DiGraph(self.df)
        if pngfile is not None:
            dotfile = pngfile + ".dot"
            write_dot(G, dotfile)
            cmd = (f'dot -Tpng -o{pngfile} {dotfile} ')
            with open(pngfile,'w') as f:
                subprocess.check_output(cmd, shell=True)
        return G

def dfvalcolor(x):
    if x > 0:
        return 'background-color: green'
    else:
        return 'backround-color: white'
    
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

        df = pd.DataFrame(data=probabilities,
                          index=self.istates,
                          columns=self.ostates)
        # df.style.applymap(dfvalcolor)
        self._df = df
        
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
                       probabilities=legacy_tpm)
        return self
        

def or_func(*args):
    invals = [v != 0 for v in args]
    return reduce(operator.or_, invals)    

def xor_func(*args):
    invals = [v != 0 for v in args]
    return reduce(operator.xor, invals)

def and_func(*args):
    invals = [v != 0 for v in args]
    return reduce(operator.and_, invals)    

# This may be too heavy-weight for 10^5++ nodes.
# NB: This does NOT hold the state of a node.  That would increase load
# on processing multiple states -- each with its own set of nodes!
# Instead, a statestr contains states for all nodes a specific time.
#
# InstanceVars: id, label, state_lut
class Node():
    """Node in network. Supports more than just two states but downstream 
    software may be built for only binary nodes. Auto increment node id that 
    will be used as label if one isn't provided on creation.
    """
    _id = 0

    def __init__(self,label=None, num_states=2, id=None, func=or_func):
        if id is None:
            id = Node._id
            Node._id += 1
        self.id = id

        if label is None:
            label = id
        self.label = label
        self.num_states = num_states
        self.func = func 
    
        
    @property
    def random_state(self):
        return choice(range(self.num_states))

    @property
    def states(self):
        """States supported by this node."""
        return range(self.num_states)

    def __repr__(self):
        return str(self.id)


# Implementation:
# A "state" is a hex string with each character representing the state of
# a specific node. Therefore: there can be at most 16 node states.
# InstanceVars: net
class States():
    """Holds info on all states for net.  Provides book-keeping operations.
    """

    def __init__(self, net=None):
        self.net = net

    # Slides show states varying by LEFT-MOST node fastest.
    # I call this BACKWARDS since each string would have to be reversed
    # to make sorted order match indices in Slides.
    # 
    # condition='...1' to fix D=1 and allow others to vary
    def tpm(self, backwards=False, condition=None):
        cand_states = [f'{i:04b}' for i in range(2**len(self.net))]
        candids = [n.id for n in self.net.nodes] # CANdidate node ids
        if condition is not None:
            pat = re.compile(condition)
            cand_states = [s for s in cand_states if re.match(pat,s)]
            print(f'cand_states={cand_states}')
            candids = [n.id for n,s in zip(self.net.nodes,condition) if s=='.']
        
        G = nx.DiGraph()
        G.add_nodes_from([self.substate(c,candids) for c in cand_states])
        for state0 in cand_states:
            state1 = self.next_state(state0)
            G.add_edge(self.substate(state0, candids),
                       self.substate(state1, candids))
        print(f'edges={list(G.edges)}')
        if backwards:
            substates = [self.substate(s,candids)[::-1] for s in cand_states]
        else:
            substates = [self.substate(s,candids) for s in cand_states]
        df = nx.to_pandas_adjacency(G, nodelist=substates, dtype=int)
        dft = df.style.applymap(dfvalcolor)
        nll = [n.label for n in self.net.nodes]
        p = nx.to_numpy_array(G)
        #tp = TransProb(in_nodes=nll, out_nodes=nll, probabilities=p)
        return dft
            
    @classmethod
    def substate(self, statestr, nodeid_list):
        """Parts of STATE that correspond to nodes"""
        return ''.join([statestr[i] for i in nodeid_list])

    # assumes func are applied using orig_statestr inputs
    def next_state(self, orig_statestr):
        next_chars = list(orig_statestr) # write-only
        G = self.net.graph
        for node in self.net.nodes:
            f = node.func
            aa = [self.node_state(lab,orig_statestr)
                  for lab in G.predecessors(node.label)]
            #!res = f(*aa)
            #!print(f'{node.label} {f.__name__}(*{aa}) \t=> {res}')
            next_chars[node.id] = f'{f(*aa):x}'
        return ''.join(next_chars)  

    
    def gen_random_state(self):
        chars = ['0'] * len(self.net)
        for node in self.net.nodes:
            chars[node.id] = f'{choice(node.states):x}'
        return ''.join(chars)        
            
        
    def flip_char(self, state, node_label, must_change=False):
        """Change the char in STATE corresponding to NODE to a different
        node state. Return new STATE."""
        chars = list(state)
        node = self.net.get_node(node_label)
        avail = node.states
        if must_change:
            avail = [s for s in avail if s != state[node.id]]
        chars[node.id] = f'{choice(avail):x}'
        return ''.join(chars)        

    def flip_chars(self, state, node_label_list, must_change=False):
        #!print(f'DBG state={state}')
        #!print(f'DBG node_label_list={node_label_list}')
        chars = list(state)
        for node_label in node_label_list:
            node = self.net.get_node(node_label)
            avail = node.states
            if must_change:
                avail = [s for s in avail if s != state[node.id]]
            chars[node.id] = f'{choice(avail):x}'
        return ''.join(chars)        
            
    def node_state(self, node_label, state):
        node = self.net.get_node(node_label)
        return int(state[node.id],16)

    @classmethod
    def state_str2tuple(cls,statestr):
        """RETURN tuple form of statestr. Only works for nodes with maximum
        of 10 node states.
        """
        return tuple([int(c,16) for c in statestr])

    @classmethod
    def state_tuple2str(cls,statetuple):
        """RETURN statestr form of statetuple.  """
        return ''.join(hex(s)[2:] for s in statetuple)
        

    #! @classmethod
    #! def state_from_node_states(cls, prev_state, ns_list):
    #!     chars = list(prev_state)
    #!     for (n,s) in ns_list:
    #!         chars[net.nodeidx_lut[n]] = s
    #!     return ''.join(chars)

# InstanceVars: graph, states, node_lut
class Net():
    nn = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
    
    def __init__(self,
                 edges = None, # connectivity edges; e.g. [(0,1), (1,2), (2,0)]
                 N = 5, # Number of nodes
                 graph = None, # networkx graph
                 #nodes = None, # e.g. list('ABCD')
                 #cm = None, # connectivity matrix
                 SpN = 2,  # States per Node
                 title = None, # Label for graph
                 ):

        if edges is None:
            n_list = range(N)
        else:
            i,j = zip(*edges)
            n_list = sorted(set(i+j))

        nodes = [Node(id=i, label=Net.nn[i], num_states=SpN) for i in n_list]

        # lut[label] -> Node
        self.node_lut = dict((n.label,n) for n in nodes)
            
        if edges is None:
            if nodes is None:
                graph = nx.DiGraph(np.ones((N,N))) # fully connected
            else:
                graph = nx.empty_graph(N, create_using=nx.DiGraph())
        else:
            graph = nx.DiGraph(edges)
        nx.relabel_nodes(graph,
                         dict(zip(graph,[n.label for n in nodes])),
                         copy=False) 
        self.graph = graph
        self.states = States(self)

    
    @property
    def cm(self):
        return nx.to_numpy_array(self.graph)

    @property
    def df(self):
        return nx.to_pandas_adjacency(self.graph, nodelist=self.node_labels)

    @property
    def nodes(self):
        return self.node_lut.values()

    @property
    def node_labels(self):
        return [n.label for n in self.node_lut.values()]

    def successors(self, node_label):
        return list(self.graph.neighbors(node_label))

    def get_node(self, node_label):
        return self.node_lut[node_label]
    
    def __len__(self):
        return len(self.graph)

    def graph(self, pngfile=None):
        """Return networkx DiGraph. Maybe write to PNG file."""
        G = nx.DiGraph(self.graph)
        if pngfile is not None:
            dotfile = pngfile + ".dot"
            write_dot(G, dotfile)
            cmd = (f'dot -Tpng -o{pngfile} {dotfile} ')
            with open(pngfile,'w') as f:
                subprocess.check_output(cmd, shell=True)
        return G

    def draw(self):
        nx.draw(self.graph,
                pos=pydot_layout(self.graph),
                # label='gnp_random_graph({N},{p})',
                with_labels=True )

    
            
def tryit():
    net = basic_noisy_selfloop_network()
    tpm = net.tpm
    
