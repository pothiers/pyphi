"""Perturb one node to catalyze transitions.

Count the transitions downstream from the perturbation over specified
time interval. 
zaptc("zapticy") = zap_TransitionCounts
"""
# Python release packages
from random import choice
from collections import Counter
# External packages
import networkx as nx


global current_state  # state of whole network
global transition_counter
transition_counter = Counter()
global net

# state is hex str in order of cm.node_labels
def state_apply_nodes(state, ns_list):
    chars = list(state)
    for (n,s) in ns_list:
        chars[net.nodeidx_lut[n]] = s
    return ''.join(chars)

    

def zap(node, nodestate):
    """Get valid new state for node that is different from nodestate."""
    #!return choice(node.states-set(nodestate))
    return '0' if nodestate=='1' else '1'

def rstate(node):
    """Return a random state for node."""
    #! return choice(node.states)
    return choice(list('01'))


    
def etc(node, state, time):
    """Effect Transition Count: random state for node my be same as prev."""
    global net
    if time == 0:
        return

    state1 = state_apply_nodes(state, [(n,rstate(n))
                                       for n in net.neighbors(node)])
    transition_counter.update([(state,state1)])

    # propagate effect
    for n in net.neighbors(node):
        etc(n, state1, time-1)

def zap_tc(cm, node, ns, state, time):
    """Count transition from zapped node to successors state"""
    global net
    if time == 0:
        return
    net = cm.graph()
    # lut[i] => node_label
    net.nodeidx_lut = dict((l,i) for (i,l) in enumerate(cm.node_labels))
    net.state = state

    new_ns = zap(node, ns)
    state0 = state_apply_nodes(state, [(node,new_ns)])
    successors = [(n,rstate(n)) for n in net.neighbors(node)]
    state1 = state_apply_nodes(state0, successors)
    transition_counter.update([(state0,state1)])

    # propagate effect
    for (s_node,s_ns) in successors:
        etc(s_node, state1, time-1)
    

    
