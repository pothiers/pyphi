"""Intended as temporary scaffolding to allow use of various IIT datastructures
in graph form."""
import pyphi
import networkx as nx
import matplotlib.pyplot as plt

from pprint import pprint as pp
from networkx.drawing.nx_pydot import write_dot


def network2graph(network, dotfile='digraph.dot'):
    G = nx.DiGraph(network.cm)
    labels = dict(list(zip(range(len(network)), network.node_labels)))
    G = nx.relabel_nodes(G, labels)
    if dotfile != None:
        write_dot(G,dotfile)
        print(f'Wrote "{dotfile}". Display with: dot -Tpng {dotfile}  '
              f'| display -')
    return G

"""Examples:
ipython -pylab

G = network2graph(pyphi.examples.actual_causation())
"""    
