import torch
import random
import networkx as nx

# Unbiased nodes are trying to figure out which nodes in the system are trustworthy and rate
# based on that.
UNBIASED="unbiased"
# Malicious nodes are essentially trying to ruin the trust in the system just by adding noise
MALICIOUS="malicious"
# Omniscient nodes know the real distribution of other nodes in the system
OMNISCIENT="omniscient"

# If there are not enough previous ratings, just give a random assessment
UNKNOWN_THRESH=10

# A simulator for the rating system
class Simulator():
  def __init__(
    self,
    size = 1000,
    num_malicious = 50,
    num_unbiased = 500,
  ):
    self.G = nx.DiGraph()
    self.t = 0

    assert(size >= num_malicious + num_unbiased)
    num_omniscient = size - num_malicious + num_unbiased
    self.size = size
    self.num_malicious = num_malicious
    self.num_unbiased = num_unbiased
    self.num_omni = num_omniscient
    # ordered as unbiased, malicious, omniscient
  def step(self):
    self.t += 1
    src = random.randint(0, self.size - 1)
    dst = random.randint(0, self.size - 1)
    while dst == src: dst = random.randint(0, self.size - 1)

    node_kind = self.node_kind(src)
    if node_kind is UNBIASED:
      self.G.add_edge(src, dst, weight=self.information_cascade(dst))
    elif node_kind is MALICIOUS:
      # just add random noise into the graph
      # TODO maybe come up with something more complex here
      self.G.add_edge(src, dst, weight=random.random())
    elif node_kind is OMNISCIENT:
      dst_node_kind = self.node_kind(dst)
      if dst_node_kind is MALICIOUS: self.G.add_edge(src, dst, weight=0)
      elif dst_node_kind is OMNISCIENT: self.G.add_edge(src, dst, weight=1)
      else: self.G.add_edge(src, dst, weight=0.5)
  def node_kind(self, v):
    assert(v < self.size)
    if v < self.num_unbiased: return UNBIASED
    elif v < self.num_unbiased + self.num_malicious: return MALICIOUS
    else: return OMNISCIENT
  def information_cascade(self, dst):
    # looks at the ratings on dst and tries to predict another rating for it
    in_edges = self.G.in_edges([dst])
    if len(in_edges) < 1: return random.random()
    # just return average of all previous weights
    return sum(self.G.edges[u, dst]['weight'] for u,_ in in_edges)/len(in_edges)
  def summary_stats(self):
    out = [0.0] * self.size
    for i in range(self.size):
      in_edges = self.G.in_edges(i)
      if len(in_edges) == 0: continue
      out[i] = sum(self.G.edges[u, i]['weight'] for u,_ in in_edges)/len(in_edges)
    return out

def main():
  sim = Simulator()
  for i in range(10000):
    sim.step()
  print(sim.summary_stats())

if __name__ == "__main__": main()
