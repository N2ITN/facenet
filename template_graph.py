#!/usr/bin/env python
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

def main():
    graphviz = GraphvizOutput()
    graphviz.output_file = '/home/zach/repos/basic.png'

    with PyCallGraph(output=graphviz):            
        stack()
main()