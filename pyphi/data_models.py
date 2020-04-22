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
from pyphi.zap_tc import Zaptc

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


# modified slightly from
# https://docs.python.org/3/library/itertools.html?highlight=itertools
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r)
                                         for r in range(len(s)+1))

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

def hilite_pos(x):
    if x > 0:
        return 'background-color: green'
    else:
        return 'backround-color: white'

def fill_using_mechanism(tp,states):
    for i in tp._df.index:
        state0 = i+'0'
        state1 = states.next_state(state0)
        d1 = int(state1[-1])
        d0 = 0 if d1 > 0 else 1
        tp._df.loc[i,:] = [d0,d1]


# InstanceVars:
#   net, _current, _df
#   in_node_lut, in_node_labels,num_in_states,istates
#   out_node_lut, out_node_labels,num_out_states,ostates
#
# @@@ num_*_states and *states can be derived from DF
class TransProb(): # @@@ This could be "virtual". Func instead of matrix.
    """TRANSition PROBability matrix (and other formats)"""
    
    def __init__(self, in_nodes=None, out_nodes=None, probabilities=None,
                net=None):
        """Create and store a Transition Probability Matrix

        This is a directed adjacency matrix from the states of in_nodes 
        to the states of out_nodes.  The number of states per node 
        is given by Node().num_states. Thus the number of rows is
        the product of num_states from in_nodes.  Number of columns
        similarly from out_nodes.
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
        # df.style.applymap(hilite_pos)
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
        

def noop_func(*args):
    return None

def ma_func(*args):
    """Mean Activation"""
    if len(args) == 0:
        return None # No result when no inputs
    return sum(args)/len(args)

def maz_func(*args):
    """Mean Activation gt zero"""
    if len(args) == 0:
        return None # No result when no inputs
    return (sum(args)/len(args)) > 0

def or_func(*args):
    if len(args) == 0:
        return None # No result when no inputs
    invals = [v != 0 for v in args]
    return reduce(operator.or_, invals)    

def xor_func(*args):
    if len(args) == 0:
        return None # No result when no inputs
    invals = [v != 0 for v in args]
    return reduce(operator.xor, invals)

def and_func(*args):
    if len(args) == 0:
        return None # No result when no inputs
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

    def __init__(self,label=None, num_states=2, id=None, func=ma_func):
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


class Stp(): # @@@ Sparse Transition Probabilities (not matrix)
    def __init__(self, net=None):
        self.net = net
        self.states = States(net)


    def tps(self, node_label):
        """Probabilities of Transition(S(t-1) => S(t)) given all possible
        states of predecessors of NODE_LABELS."""
        sscounter = Counter()
        
        #! preds = set(itertools.chain.from_iterable(
        #!     self.net.graph.predecessors(n) for n in node_labels))
        preds = set(self.net.graph.predecessors(node_label))
        print(f'DBG: preds={preds}')
        for s0 in self.states.gen_all_states(preds):
            for s1 in sell.states.gen_all_states(node_labels):
                print(f'DBG s0={s0} s1={s1}')
                for nlab in node_labels:
                    nstate = self.state.eval_node(nlab,s0)
                sscounter.update([(s0,s1)])
        return instates
        
        
        
    
    
# Implementation:
# A "state" is a hex string with each character representing the state of
# a specific node. Therefore: there can be at most 16 node states.
#
# Should make use of nx.subgraph_view to filter nodes (with relabeling)
# as we proceed thru slides.
#
# InstanceVars: net
class States():
    """Holds info on all states for net.  Provides book-keeping operations.

    The statestr '23._' is interpreted as, Network contains 4 nodes and:
      Node0.state2, Node1.state3, Node2.ANYstate, node3.IGNOREstate
    """

    def __init__(self, net=None):
        self.net = net
        self.condition = {}
        self.graph = None
        self.state = None  # Selected (current) state

    def state_to_dict(self, statestr, exclude=None):
        nodes = self.net.nodes
        return dict((n.label,s) for s,n in zip(statestr, nodes)
                    if n not in exclude)
        
    #! # Slides show states varying by LEFT-MOST node fastest.
    #! # I call this BACKWARDS since each string would have to be reversed
    #! # to make sorted order match indices in Slides.
    #! def tpg(self, condition=None, draw=False):
    #!     """Transition Prob Graph. Condition and Marginalize-out node(s)
    #!     Condition is statestr regexp.
    #!     """
    #!     if condition is not None:
    #!         self.condition = condition
    #!     #! # Full system states
    #!     cand_states = [f'{i:04b}' for i in range(2**len(self.net))]
    #!     #! cand_ids = [n.id for n in self.net.nodes] # CANdidate node ids
    #! 
    #!     # condition=dict(D=1) to fix D=1 and allow others to vary
    #!     # cond_rexp: e.g. '...1'; a regexp matching sys statestr
    #! 
    #!     cond_rexp = self.condition_rexp(self.condition)
    #!     pat = re.compile(cond_rexp)
    #!     cand_states = [s for s in cand_states if re.match(pat,s)]
    #!     cand_ids = [n.id for n,s in zip(self.net.nodes,cond_rexp) if s=='.']
    #!     G = nx.DiGraph()
    #!     G.add_nodes_from([self.substate(c,cand_ids) for c in cand_states])
    #!     for state0 in cand_states:
    #!         state1 = self.next_state(state0)
    #!         s0 = self.substate(state0, cand_ids)
    #!         s1 = self.substate(state1, cand_ids)
    #!         # Weight property is the count of connections between 2 states.
    #!         # Multiple output states get combined on Conditioning
    #!         if G.has_edge(s0,s1):
    #!             G[s0][s1]['weight'] += 1
    #!         else:
    #!             G.add_edge(s0,s1,weight=1)
    #!     self.graph = G
    #!     if draw:
    #!         nx.draw(G, pos=pydot_layout(G), with_labels=True)
    #!     return G
    #! 
    #! def tpm(self, backwards=False, condition=None):
    #!     """Condition is a dict with keys that are node labels, values that 
    #!     are states for that node."""
    #!     G = self.tpg(condition=condition)
    #!     substates = [label for label in G.nodes]
    #!     if backwards:
    #!         substates = [label[::-1] for label in substates]
    #!     df = nx.to_pandas_adjacency(G, nodelist=substates, dtype=int)
    #!     return df


    def tpg(self, statestr,
             mechanism=None, candidate_system=None, purview=None,
             draw=False):
        if mechanism is None:
            mech = self.net.nodes
        else:
            mech = self.net.get_nodes(mechanism)

        if candidate_system is None:
            cand_sys = mech
        else:
            cand_sys = self.net.get_nodes(candidate_system)
        self.cand_sys = cand_sys

        if purview is None:
            pur = mech
        else:
            pur = self.net.get_nodes(purview)
        self.pur = pur
        
        bg_condition = self.state_to_dict(statestr,exclude=cand_sys)
        mech_state = self.substate(statestr, (n.id for n in mech))

        G = nx.DiGraph()

        for instate in self.gen_all_states(n.label for n in cand_sys):
            s0 = instate.replace('_','')
            s1 = self.substate(self.next_state(instate,condition=bg_condition),
                               (n.id for n in pur))
            if G.has_edge(s0,s1):
                G[s0][s1]['weight'] += 1
            else:
                G.add_edge(s0,s1,weight=1)
            
        print(f'inferered={dict(bg_cond=bg_condition, mech_state=mech_state)}')
        self.graph = G
        if draw:
            nx.draw(G, pos=pydot_layout(G), with_labels=True)
        return G

    def tpm(self, *args, backwards=False,**kwargs):
        G = self.tpg(*args, **kwargs)
        ins,outs = zip(*G.edges())
        print(f'cand_sys={self.cand_sys}, pur={self.pur}')
        source = 'i' + ''.join(n.label for n in self.cand_sys)
        target = 'o' + ''.join(n.label for n in self.pur)
        print(f'source={source}, target={target}')
        df = nx.to_pandas_edgelist(G,
                                   source=source,
                                   target=target,
                                   nodelist=set(ins) | set(outs) )

        mat = df.pivot(index=source, columns=target,
                       values='weight').fillna(value=0).astype('int')
        if backwards:
            newindex= sorted(mat.index, key= lambda lab: lab[::-1])
            newcol= sorted(mat.columns, key = lambda lab: lab[::-1])
            return mat.reindex(index=newindex, columns=newcol)
        return mat
        #return df


    #! def effect_repertoire(self, purview=None, state=None):
    #!     if self.graph is None:
    #!         G = self.tpg(condition=self.condition)
    #!         self.graph = G
    #!     if purview is None:
    #!         return self.graph.succ[state].keys()
    #!     else:
    #!         # nodes not in purview (combine these in out state)
    #!         margin = set(self.net.nodes) - self.get_nodes(purview)
    #! 
    #! 
    #!     for state0,state1 in self.graph.edges():
    #!         next_regexp
    #!         s1 = self.substate(state1)
        
    def purview(self, out_node_labels):
        if self.graph is None:
            G = self.tpg(condition=self.condition)
            self.graph = G
        G = self.graph

        #!instates,outstates = zip(*self.graph.edges())
        #!cond_rexp = self.condition_rexp(self.condition)
        #!nodes = [self.node.get_node(label) for label in out_node_labels]
        #!out_ids = [n.id for n,s in zip(nodes,cond_rexp) if s=='.']

        for state0,state1 in G.edges():
            s1 = self.substate(state1, cand_ids)
            # Weight property is the count of connections between 2 states.
            # Multiple output states get combined on Conditioning
            if G.has_edge(s0,s1):
                G[s0][s1]['weight'] += 1
            else:
                G.add_edge(s0,s1,weight=1)
        
    
    def condition_rexp(self, cond_dict):
        lablut = dict((n.id,n.label) for n in self.net.nodes)
        cond = dict((k,f'{v:x}') for k,v in cond_dict.items())
        return ''.join(cond.get(lablut[i],'.') for i in range(len(lablut)))

    @classmethod
    def substate(self, statestr, nodeid_list):
        """Part of system state (statestr) that correspond to nodes"""
        return ''.join([statestr[i] for i in nodeid_list])

    def apply_condition(self, condition, cur_statestr):
        next_chars = list(cur_statestr) # write-only
        for k,v in condition.items():
            next_chars[self.net.get_node(k).id] = v
        return ''.join(next_chars)
    
    # assumes func are applied using orig_statestr inputs
    def next_state(self, cur_statestr, condition={}):
        """condition will overwrite cur_statestr before applying func"""
        statestr = self.apply_condition(condition, cur_statestr)
        next_chars = list(statestr) # write-only
        G = self.net.graph
        for node in self.net.nodes:
            args = [self.node_state(lab,statestr)
                    for lab in G.predecessors(node.label)]
            result = node.func(*args)
            #!print(f'{node.label} {f.__name__}(*{args}) \t=> {res}')
            if result is not None:
                next_chars[node.id] = f'{result:x}'
        return ''.join(next_chars)  

    def eval_node(self, node_label, statestr):
        """Return the state that results from running the func of node
        against the state of its predecessors according to statestr."""
        args = [self.node_state(lab,statestr)
              for lab in G.predecessors(node_label)]
        result = self.net.get_node(node_label).func(*args)
        return f'{result:x}'
    
    def gen_random_state(self):
        chars = ['0'] * len(self.net)
        for node in self.net.nodes:
            chars[node.id] = f'{choice(node.states):x}'
        return ''.join(chars)        
    
    def gen_all_states(self, node_labels=None):
        """States produced always have one character per network node.
        But, all characters that do NOT correspond to node_labels are '_'.
        If node_labels is a subset of all nodes, the resulting list of 
        states would have duplicates. Only unique states are returned.
        """
        states = [list(tup)
                  for tup in itertools.product(*[[f'{i:x}'
                                                  for i in range(n.num_states)]
                                                 for n in self.net.nodes])]
        if node_labels is not None:
            bg = set(self.net.nodes) - set(self.net.get_nodes(node_labels))
            for node in bg:
                for sv in states:
                    sv[node.id] = '_'

        #return (''.join(hex(s)[2:] for s in sv) for sv in states)
        #return set(''.join(f'{s:x}' for s in sv) for sv in states)
        
        return set([''.join(sv) for sv in states])
        #return states
        
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


# InstanceVars: graph, hstates, tpm
class Gnet():
    """Binary nodes only."""
    nn = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')

    def __init__(self, nxgraph=None, tpm=None):
        self.graph = nx.relabel_nodes(nxgraph, dict(enumerate(Gnet.nn)))
        states = itertools.product(*[range(2) for i in self.graph.nodes])
        hstates = [''.join(f'{s:x}' for s in sv)  for sv in states]
        self.hstates = hstates
        # TPM:: states x nodes
        if tpm is None:
            ns = len(hstates)
            nn = len(self.graph)
            #tpm = np.ones((ns, nn))
            tpm = np.random.rand(ns,nn)
        self.tpm = pd.DataFrame(tpm, index=hstates, columns=self.graph.nodes)

    def draw(self):
        nx.draw(self.graph, pos=pydot_layout(self.graph), with_labels=True)
        return self

    @property
    def legacy_network(self):
        return pyphi.Network(self.tpm.to_numpy(),
                             cm=nx.to_numpy_array(self.graph),
                             node_labels=self.graph.nodes)

    def discover_tpm(self, time_steps=10, verbose=False):
        dd = dict((l,i) for (i,l) in enumerate(self.graph.nodes))
        edges = [(dd[u],dd[v]) for (u,v) in self.graph.edges()]
        
        ztc = Zaptc(net=Net(edges=edges))
        ztc.zapall(time_steps, verbose=verbose)
        self.tpm = ztc.tpm_sbn   

    def phi(self, statestr=None):
        # first output state
        if statestr is None:
            statestr = choice(self.tpm.index)
        state = [int(c) for c in list(statestr)] 
        print(f'Calculating \u03A6 at state={state}')
        node_indices = tuple(range(len(self.graph)))
        subsystem = pyphi.Subsystem(self.legacy_network, state, node_indices)
        return pyphi.compute.phi(subsystem)
        

# Mechanism :: Nodes part of Input state
# Purview :: Nodes part of Output state
# Repertoire :: row of “local TPM” that corresponds to selected state
#               of the mechanism
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
                 func = maz_func, # default mechanism for all nodes
                 ):
        G = nx.DiGraph()
        if edges is None:
            n_list = range(N)
        else:
            i,j = zip(*edges)
            minid = min(i)
            maxid = max(j)
            n_list = sorted(range(minid,maxid+1))
        nodes = [Node(id=i, label=Net.nn[i], num_states=SpN, func=func)
                 for i in n_list]

        # lut[label] -> Node
        self.node_lut = dict((n.label,n) for n in nodes)
        #invlut[i] -> label
        invlut = dict(((v.id,v.label) for v in self.node_lut.values())) 
            
        G.add_nodes_from(self.node_lut.keys())
        if edges is not None:
            G.add_edges_from([(invlut[i],invlut[j]) for (i,j) in edges])
        self.graph = G
        self.graph.name = title
        self.states = States(self)

    def discover_tpm(self, time_steps=10, verbose=False):
        #!dd = dict((n.label,n.id) for n in enumerate(self.nodes))
        #!edges = [(dd[u],dd[v]) for (u,v) in self.graph.edges()]
        ztc = Zaptc(net=self)
        ztc.zapall(time_steps, verbose=verbose)
        self.ztc = ztc
        self.tpm = ztc.tpm_sbn   

    @classmethod
    def candidate_mechanisms(cls, candidate_system):  
        ps = powerset(set(candidate_system)) # iterator
        return (ss for ss in ps if ss != ()) # remove empty, return GENERATOR 

    

    @property
    def cm(self):
        return nx.to_numpy_array(self.graph)

    @property
    def df(self):
        return nx.to_pandas_adjacency(self.graph, nodelist=self.node_labels)


    @property
    def nodes(self):
        """Return list of all nodes in ID order."""
        return sorted(self.node_lut.values(), key=lambda n: n.id)

    def node(self, node_id):
        return list(self.node_lut.keys())[node_id]

    @property
    def node_labels(self):
        return [n.label for n in self.node_lut.values()]

    def successors(self, node_label):
        return list(self.graph.neighbors(node_label))

    def get_node(self, node_label):
        return self.node_lut[node_label]

    def get_nodes(self, node_labels):
        return [self.node_lut[label] for label in node_labels]
    
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
        return self
    
            
def tryit():
    net = basic_noisy_selfloop_network()
    tpm = net.tpm
    
