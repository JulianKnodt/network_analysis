import numpy as np
import torch
import torch.nn.functional as F
import random
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

# Unbiased nodes are trying to figure out which nodes in the system are trustworthy and rate
# based on that.
UNBIASED="unbiased"
# Malicious nodes are essentially trying to ruin the trust in the system just by adding noise
MALICIOUS="malicious"
# Omniscient nodes know the real distribution of other nodes in the system
OMNISCIENT="omniscient"

# If there are not enough previous ratings, just give a random assessment
UNKNOWN_THRESH=2


# A simulator for the rating system
class Simulator():
  def __init__(
    self,
    size = 500,
    num_malicious = 50,
    num_unbiased = 400,
  ):
    self.G = nx.DiGraph()
    self.t = 0

    assert(size >= num_malicious + num_unbiased)
    num_omniscient = size - num_malicious - num_unbiased
    self.size = size
    self.num_malicious = num_malicious
    self.num_unbiased = num_unbiased
    self.num_omni = num_omniscient
    # ordered as unbiased, malicious, omniscient

    self.txs = {}
  def step(self):
    self.t += 1
    src = random.randint(0, self.size - 1)
    dst = random.randint(0, self.size - 1)
    while dst == src: dst = random.randint(0, self.size - 1)

    # currently unused but could be useful for a predictor
    tx = self.mk_tx(dst)
    node_kind = self.node_kind(src)
    if node_kind is UNBIASED:
      self.G.add_edge(src, dst, weight=self.information_cascade(dst, tx))
    elif node_kind is MALICIOUS:
      # just add random noise into the graph
      # TODO maybe come up with something more complex here
      self.G.add_edge(src, dst, weight=random.random())
    elif node_kind is OMNISCIENT:
      dst_node_kind = self.node_kind(dst)
      if dst_node_kind is MALICIOUS: self.G.add_edge(src, dst, weight=0)
      elif dst_node_kind is OMNISCIENT: self.G.add_edge(src, dst, weight=1)
      else: self.G.add_edge(src, dst, weight=0.5)
  # makes a private transaction of some good from dst to src (inverse of rating)
  def mk_tx(self, dst):
    node_kind = self.node_kind(dst)
    # unbiased nodes are fully random txs
    if node_kind is UNBIASED: val = random.random()
    # malicious nodes will tend to have bad txs
    elif node_kind is MALICIOUS: val = np.random.binomial(100, 0.1)/100
    # expert nodes will tend to have good txs
    elif node_kind is OMNISCIENT: val = np.random.binomial(100, 0.9)/100
    self.txs.setdefault(dst, []).append(val)
    return val
  def node_kind(self, v):
    assert(v < self.size)
    if v < self.num_unbiased: return UNBIASED
    elif v < self.num_unbiased + self.num_malicious: return MALICIOUS
    else: return OMNISCIENT

  # use average of previous information to generate the next rating
  def information_cascade(self, dst, tx):
    # looks at the ratings on dst and tries to predict another rating for it
    in_edges = self.G.in_edges([dst])
    if len(in_edges) < UNKNOWN_THRESH: return tx
    # just return average of all previous weights, maybe it'd be worth adding
    # data from its own private transaction into it, which would be additional noise
    return sum(self.G.edges[u, dst]['weight'] for u,_ in in_edges)/len(in_edges)
  def summary_stats(self):
    out = [0.0] * self.size
    for i in range(self.size):
      in_edges = self.G.in_edges([i])
      if len(in_edges) == 0: continue
      out[i] = sum(self.G.edges[u, i]['weight'] for u,_ in in_edges)/len(in_edges)
    return out
  def expected(self):
    return [0.5] * self.num_unbiased + \
      [0] * self.num_malicious + \
      [1] * self.num_omni

def analysis_of_num_malicious():
  xs = np.arange(0, 100, 2)
  ys = []
  for m in tqdm(xs):
    sim = Simulator(num_malicious=m)
    for i in range(20000):
      sim.step()
    got = torch.tensor(sim.summary_stats())
    exp = torch.tensor(sim.expected(), dtype=torch.float)
    bce_loss = F.binary_cross_entropy(got, exp.float())
    ys.append(bce_loss.item())
  plt.plot(xs, ys)
  plt.show()

def analysis_of_iterations_on_convergence(num_mal, num_unb=600):
  steps_per = 500
  xs = np.arange(0, 25_000, steps_per)
  ys = []
  sim = Simulator(size=1000, num_unbiased=num_unb, num_malicious=num_mal)
  # warm up
  for i in range(200): sim.step()

  for m in tqdm(xs):
    for i in range(steps_per): sim.step()
    got = torch.tensor(sim.summary_stats())
    exp = torch.tensor(sim.expected(), dtype=torch.float)
    bce_loss = F.binary_cross_entropy(got, exp.float())
    ys.append(bce_loss.log().item())
  plt.plot(xs, ys, label=f"{num_mal} Malicious")

def main():
  #analysis_of_num_malicious()
  plt.title("Change in cross-entropy w.r.t. information propagation in the network")
  plt.xlabel("Steps")
  plt.ylabel("Log cross-entropy")
  analysis_of_iterations_on_convergence(400)
  analysis_of_iterations_on_convergence(300)
  analysis_of_iterations_on_convergence(200)
  analysis_of_iterations_on_convergence(100)
  analysis_of_iterations_on_convergence(000)
  #analysis_of_iterations_on_convergence(0,100)
  plt.plot(
    np.arange(0, 25_000, 500),np.log(0.5+np.zeros(25_000//500)), label="assigning all random"
  )
  plt.legend()
  plt.show()

if __name__ == "__main__": main()
