#!/usr/bin/env python
# How do our data-structures scale?

import numpy as np
import pandas as pd
from IPython.display import Image
import matplotlib.pyplot as plt
from collections import Counter
import functools, operator, random, itertools
import sys, argparse, logging
import subprocess

import pyphi
from pyphi.examples import basic_noisy_selfloop_network
import pyphi.data_models as dm  # Prototype code


# Max run N=10 on chimp20 before memory exceeded
def gen_net(N=10, C=5, S=3):
    #! N = 10 # number of Nodes in system
    #! C = 5 # maximum number of Connections from a node to downstream nodes
    #! S = 3 # number of States per node

    # for node names
    nn = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789') 

    labels = nn[:N]
    nodes = [(l,S) for l in labels] # node (Label, NumStates)

    print(f'''
{N} # number of Nodes
{C} # maximum number of Connections from a node
{S} # number of States per node
{labels} # node labels
    ''')

    am = np.zeros((N,N))
    cm = dm.CM(am=am,node_labels=labels)

    for r in labels: cm.df.loc[r,random.sample(labels,C)] = 1
    png=f'C_{C}.png'
    
    cm.graph(pngfile=png)
    subprocess.check_output(f'firefox {png}', shell=True)
    print(f'Created graph in: {png} (display in firefox)')
    print(f'cm=\n{cm.df}')
    #!Image(filename=png, width=500)
    

    p=np.random.random((S**N,S**N))
    tp = dm.TransProb(in_nodes=nodes,out_nodes=nodes,probabilities=p)
    print(f'tp=\n{tp.df}')  # state-to-state Transition Probabilities

    net = dm.Network(tp=tp, cm=cm)
    print(f'net={net}')


##############################################################################

def main():
    #!print('EXECUTING: {}\n\n'.format(' '.join(sys.argv)))
    parser = argparse.ArgumentParser(
        #!version='1.0.1',
        description='My shiny new python program',
        epilog='EXAMPLE: %(prog)s a b"'
        )
    parser.add_argument('-N', default=10, type=int,
                        help='number of Nodes in system'
                        )
    parser.add_argument('-C', default=5,  type=int,
                        help='maximum number of Connections from a node')
    parser.add_argument('-S', default=3,  type=int,
                        help='number of States per node')
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


    gen_net(args.N, args.C, args.S)

if __name__ == '__main__':
    main()
