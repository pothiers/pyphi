# Python standard library
from collections import Counter, defaultdict
import itertools
from functools import reduce
import operator
from random import choice
# External packages
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
from networkx.drawing.nx_pydot import pydot_layout
import pandas as pd
import numpy as np
# Local packages
import pyphi
import pyphi.network 



def noop_func(*args):
    return None

def ma_func(*args):
    """Mean Activation"""
    if len(args) == 0:
        return None # No result when no inputs
    return sum(args)/len(args)
copy_func = ma_func #  Indended to have exactly one input node. 

def maz_func(*args):
    """Mean Activation gt zero"""
    if len(args) == 0:
        return None # No result when no inputs
    return (1 if ((sum(args)/len(args)) > 0) else 0)

def tri_func(*args):
    """Count input states. Produces 3 states"""
    if len(args) == 0:
        return 0
    if sum(args) == 0:
        return 0
    if sum(args) == 1:
        return 1
    else:
        return 2


def or_func(*args):
    if len(args) == 0:
        return None # No result when no inputs
    invals = [v != 0 for v in args]
    return int(reduce(operator.or_, invals)    )

# COPY in papers seems to assume exactly one in-edge
def copy_func(*args):
    if len(args) == 0:
        return None # No result when no inputs
    invals = [v != 0 for v in args]
    return int(reduce(operator.or_, invals))


def xor_func(*args):
    if len(args) == 0:
        return None # No result when no inputs
    invals = [v != 0 for v in args]
    return int(reduce(operator.xor, invals))

def and_func(*args):
    if len(args) == 0:
        return None # No result when no inputs
    invals = [v != 0 for v in args]
    return int(reduce(operator.and_, invals))

# This may be too heavy-weight for 10^5++ nodes.
# NB: This does NOT hold the state of a node.  That would increase load
# on processing multiple states -- each with its own set of nodes!
# Instead, a statestr contains states for all nodes a specific time.
#
# InstanceVars: id, label, num_states, func
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
        self.label = label or id
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

    def __str__(self):
        return f'{self.label}({self.id}): {self.num_states},{self.func.__name__}'


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
                 #graph = None, # networkx graph
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
            minid = min(i+j)
            maxid = max(i+j)
            n_list = sorted(range(minid,maxid+1))
        print(f'edges={edges} n_list={n_list}')
        nodes = [Node(id=i, label=Net.nn[i], num_states=SpN, func=func)
                 for i in n_list]

        # lut[label] -> Node
        self.node_lut = dict((n.label,n) for n in nodes)
        #invlut[i] -> label
        invlut = dict(((n.id,n.label) for n in self.node_lut.values())) 
            
        G.add_nodes_from(self.node_lut.keys())
        if edges is not None:
            G.add_edges_from([(invlut[i],invlut[j]) for (i,j) in edges])
        self.graph = G
        self.graph.name = title
        #self.states = States(net=self)
        self.tpm_df = self.tpm



        
    def node_state_counts(self, node):
        """Truth table of node.func run over all possible inputs.
        Inputs are predecessor nodes with all possible states."""
        #node = self.get_node(node_label)
        preds = (self.get_node(l)
                 for l in set(self.graph.predecessors(node.label)))
        counter = Counter()
        counter.update(node.func(*sv)
                       for sv in itertools.product(*[n.states for n in preds]))
        return counter

    def eval_node(self, node, system_state_tup):
        preds_id = set([self.get_node(l).id
                    for l in set(self.graph.predecessors(node.label))])
        args = [system_state_tup[i] for i in preds_id]
        return node.func(*args)
        

    def node_pd(self, node):
        """Probability Distribution of NODE states given all possible inputs
        constrained by graph."""
        #node = self.get_node(node_label)
        counts = self.node_state_counts(node)
        total = sum(counts.values())
        return [counts[i]/total for i in node.states]


    @property
    def tpm(self):
        """Iterate over all possible states(!!!) using node funcs
        to calculate output state. State-to-State form. Allows non-binary"""
        backwards=True  # I hate the order the papers use!!
        allstates = list(itertools.product(*[n.states for n in self.nodes]))
        N = len(self.nodes)
        allstatesstr = [''.join([f'{s:x}' for s in sv]) for sv in allstates]
        df = pd.DataFrame(index=allstatesstr,
                          columns=[n.label for n in self.nodes]).fillna(0)
        
        for sv in allstates:
            s0 = ''.join(f'{s:x}' for s in sv)
            for i in range(N):
                node = self.nodes[i]
                nodestate = self.eval_node(node,sv)
                df.loc[s0,node.label] =  nodestate

        if backwards:
            newindex= sorted(df.index, key= lambda lab: lab[::-1])
            return df.reindex(index=newindex)
        return df

    def nodes_to_tpm(self, verbose=False):
        """Iterate over all possible states(!!!) using node funcs
        to calculate output state. State-to-Node form. binary nodes only"""
        transitions = defaultdict(int) # transitions[(s0,s1)] => count
        allstates = list(itertools.product(*[n.states for n in self.nodes]))
        for sv in allstates:
            s0 = ''.join(f'{s:x}' for s in sv)
            sv1 = list(sv)
            for i in range(len(sv)):
                node = self.nodes[i]
                cnt = self.node_state_counts(node)
                for nodestate in node.states:
                    sv1[i] = nodestate
                    s1 = ''.join(f'{s:x}' for s in sv1)
                    transitions[(s0,s1)] += cnt[nodestate]
            # pad because pyphi requires full matrix
            for sv1 in allstates:
                s1 = ''.join(f'{s:x}' for s in sv1)                
                transitions[(s0,s1)] += 0

        # Convert counts to DF
        allstatesstr = [''.join([f'{s:x}' for s in sv]) for sv in allstates]
        allnodes = [n.label for n in self.nodes()]
        df = pd.DataFrame(index=allstatesstr, columns=allnodes).fillna(0)
        for ((s0,s1),count) in transitions.items():
            df.loc[s0,s1] =  df.loc[s0,s1] + count
            
        # return normalized form
        return df.fillna(0)
    
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

    @property
    def pyphi_network(self):
        return pyphi.network.Network(self.tpm.to_numpy(),
                                     cm=self.cm,
                                     node_labels=self.node_labels)
    
    def phi(self, statestr=None):
        # first output state
        if statestr is None:
            instatestr = choice(self.tpm_df.index)
            statestr = ''.join(f'{s:x}' for s in self.tpm_df.loc[instatestr,:])
        state = [int(c) for c in list(statestr)] 
        print(f'Calculating \u03A6 at state={state}')
        node_indices = tuple(range(len(self.graph)))
        subsystem = pyphi.Subsystem(self.pyphi_network, state, node_indices)
        return pyphi.compute.phi(subsystem)

