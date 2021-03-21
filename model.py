import csv
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import networkx as nx

def init():
  G = nx.DiGraph()
  with open("soc-sign-bitcoinotc.csv") as f:
    reader = csv.reader(f)
    for src, dst, w, time in reader:
      G.add_node(src)
      G.add_node(dst)
      G.add_edge(src, dst, weight=float(w), time=time)
  return G
