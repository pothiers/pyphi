#!/usr/bin/env python
# coding: utf-8
'''Calculate phi for a network given a networkx graph.

This is a wrapper around pyphi routines intended to give a feel for
how the calculation performs on different networks for people with no
prior knowledge of IIT.

The minimum input is a networkx DiGraph (directed graph).
Networkx has a collection of graph generators.  See:
https://networkx.github.io/documentation/stable/reference/generators.html

Optional parameters are:
  tpm:  Transition Probability Matrix
        Defaults to one that is experimentally determined using zaptc.
        For experiment details, see below.

  state: The starting state. 
         Defaults to a random choice from the set provided in the TPM
         (row indices).

WARNING: The time (and memory) used by the calculation can be roughly
exponential over the number of nodes in the graph. When just trying things out,
it is best to keep the numbers of nodes <= 5.

Pay attention to output like this:
Evaluating Φ cuts:  26%|████▏           | 66/254 [3:43:19<34:29:54, 660.61s/it]

The above is telling you that the current estimate for time to finish
evaluating cuts is over 34 hours (after running for over 3 hours). 

EXAMPLE:
  g = nx.gnp_random_graph(4, 0.4, directed=True) # num_nodes, prob_edge
  gn = dm.Gnet(g)             # load the network
  gn.discover_tpm()           # gen TPM with experiment
  gn.phi()                    # calculate PHI using rnd state
  # => <phi-value>


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

'''
# Standard packages
import sys, argparse, logging
# External packages
import networkx as nx
from networkx.drawing.nx_pydot import pydot_layout
import pandas as pd
# Internal packages
import pyphi.data_models as dm  # Prototype code
from pyphi.zap_tc import Zaptc
from pyphi.convert import sbs2sbn, sbn2sbs      
import pyphi



def calcPhi(gmlfile=None,
            number_nodes=5,  # if gmlfile=None
            prob_edge=0.4,   # if gmlfile=None
            state=None,
            savefile=None, # GML file to save graph into
            draw=False,    # Draw graph used; broken @@@
            time_steps=10):
    if gmlfile is None:
        g = nx.gnp_random_graph(number_nodes, prob_edge, directed=True)
        g.name = f'gnp_random_graph({number_nodes}, {prob_edge})'
    else:
        g = nx.read_gml(gmlfile)
        g.name = f'Read from: {gmlfile}'
    print(f'\nAbout the graph: \n{nx.info(g)}\n')
    if savefile is not None:
        print(f'Writing graph to: {savefile}')
        nx.write_gml(g, savefile)
    
    gn = dm.Gnet(g)
    if draw:
        print('(Cannot draw graph)\n')
        gn.draw()
    gn.discover_tpm(time_steps=time_steps)
    phi = gn.phi(statestr=state)
    print(f'\u03A6 = {phi}')
    return phi




##############################################################################

url='https://networkx.github.io/documentation/stable/reference/readwrite/gml.html'
def main():
    #! print('EXECUTING: {}\n\n'.format(' '.join(sys.argv)))
    parser = argparse.ArgumentParser(
        #!version='1.0.1',
        description='Calculate phi for a network given graph and state',
        epilog='EXAMPLE: %(prog)s mygraph.gml'
        )
    dft_time = 13
    dft_nc = 5 # node count default
    dft_pe = 0.4 # probability of generate edge (default)
    parser.add_argument('-g', '--gmlfile',
                        #type=argparse.FileType('r') ,
                        help=f'Graph file in GML format. see: {url}')
    parser.add_argument('-N','--number_nodes', type=int, default=dft_nc,
                        help=('Number of nodes in random graph '
                              '(when gmlfile not given). '
                              f'[default={dft_nc}]'))
    parser.add_argument('-p','--prob_edge', type=float, default=dft_pe,
                        help=('Probality of edge between 2 nodes in random '
                              'graph (when gmlfile not given). '
                              f'[default={dft_pe}]'))
    parser.add_argument('--state',
                        help=('State of system for calculation. '
                              '[default: random choice from known states]'))
    parser.add_argument('-s', '--savefile', 
                        help='File to save graph into (dflt=None)')
    parser.add_argument('-t', '--time_steps', type=int, default=dft_time,
                        help=('Number of steps to walk network when '
                              'collecting transition statistics. '
                              f'[default={dft_time}]'))
    parser.add_argument('--loglevel',      help='Kind of diagnostic output',
                        choices = ['CRTICAL','ERROR','WARNING','INFO','DEBUG'],
                        default='WARNING',
                        )
    args = parser.parse_args()

    log_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(log_level, int):
        parser.error('Invalid log level: %s' % args.loglevel) 
    logging.basicConfig(level = log_level,
                        format='%(levelname)s %(message)s',
                        datefmt='%m-%d %H:%M'
                        )
    logging.debug('Debug output is enabled!!!')

    res = None
    try:
        res = calcPhi(args.gmlfile,
                      number_nodes=args.number_nodes,
                      prob_edge=args.prob_edge,
                      state=args.state,
                      time_steps=args.time_steps,
                      savefile=args.savefile)
    except Exception as err:
        print(f'Could not calculate Phi; {err}')
        
    print(f'# Phi={res}')

if __name__ == '__main__':
    main()
