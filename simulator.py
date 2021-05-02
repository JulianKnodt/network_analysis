import numpy as np
import torch
import torch.nn.functional as F
import random
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

# import model to use for predictions
from model import predict_trust, train_model

# Unbiased nodes are trying to figure out which nodes in the system are trustworthy and rate
# based on that.
UNBIASED="unbiased"
# Malicious nodes are essentially trying to ruin the trust in the system just by adding noise
MALICIOUS="malicious"
# Omniscient nodes know the real distribution of other nodes in the system
OMNISCIENT="omniscient"

class Phase:
  def __init__(
    self,
    phase_len:int=1000,
    num:int=1000,
  ):
    self.t = 0
    self.total_ratings = 0
    self.n_total = 0
    self.rtr_sums = [0] * num
    self.rtr_n = [0] * num
    self.rte_sums = [0] * num
    self.rte_n = [0] * num

    self.neighborhood = {}
    self.num_neighborhood = {}
    self.phase_len = phase_len

  def is_full(self): return self.t == self.phase_len
  def add_tx(self, src, dst, rating):
    self.total_ratings += rating
    self.n_total += 1
    self.rtr_sums[src] += rating
    self.rtr_n[src] += 1

    self.rte_sums[dst] += rating
    self.rte_n[dst] += 1

    self.neighborhood.setdefault(src, {}).setdefault(dst, 0)
    self.neighborhood[src][dst] += rating
    self.num_neighborhood.setdefault(src, {}).setdefault(dst, 0)
    self.num_neighborhood[src][dst] += 1

    self.t += 1


# A simulator for the rating system
class Simulator():
  def __init__(
    self,
    size = 500,
    num_malicious = 50,
    num_unbiased = 400,

    unknown_thresh = 1,
    predictor = None,
    phase_len = 50,
  ):
    self.new_phase = lambda: Phase(num=size, phase_len=phase_len)
    self.G = nx.DiGraph()
    self.t = 0
    self.prev_phase = None
    self.curr_phase = self.new_phase()
    self.global_phase = self.new_phase()

    assert(size >= num_malicious + num_unbiased)
    num_omniscient = size - num_malicious - num_unbiased
    self.size = size
    self.num_malicious = num_malicious
    self.num_unbiased = num_unbiased
    self.num_omni = num_omniscient
    # ordered as unbiased, malicious, omniscient

    self.txs = {}
    self.unknown_thresh=unknown_thresh
    self.predictor = predictor
  def step(self):
    self.t += 1

    src = random.randint(0, self.size - 1)
    dst = random.randint(0, self.size - 1)
    while dst == src: dst = random.randint(0, self.size - 1)

    tx = self.mk_tx(dst)
    src_kind = self.node_kind(src)

    if src_kind is UNBIASED:
      if self.predictor is not None and self.prev_phase is not None:
        rating = predict_trust(
          self.predictor,

          src, dst,

          self.global_phase.total_ratings,
          self.global_phase.n_total,

          self.curr_phase.rtr_sums,
          self.curr_phase.rtr_n,

          self.curr_phase.rte_sums,
          self.curr_phase.rte_n,

          self.curr_phase.neighborhood,
          self.curr_phase.num_neighborhood,

          self.global_phase.rtr_sums,
          self.global_phase.rtr_n,

          self.global_phase.rte_sums,
          self.global_phase.rte_n,

          self.global_phase.neighborhood,
          self.global_phase.num_neighborhood,

          self.prev_phase.rtr_sums,
          self.prev_phase.rtr_n,

          self.prev_phase.rte_sums,
          self.prev_phase.rte_n,
        )
      else:
        rating = self.information_cascade(dst, tx)
        self.G.add_edge(src, dst, weight=self.information_cascade(dst, tx))
    elif src_kind is MALICIOUS:
      # just add random noise into the graph
      # TODO maybe come up with something more complex here
      rating = random.random()
      self.G.add_edge(src, dst, weight=rating)
    elif src_kind is OMNISCIENT:
      dst_node_kind = self.node_kind(dst)
      if dst_node_kind is MALICIOUS: self.G.add_edge(src, dst, weight=0)
      elif dst_node_kind is OMNISCIENT: self.G.add_edge(src, dst, weight=1)
      else: self.G.add_edge(src, dst, weight=1)
      rating = 0 if dst_node_kind is MALICIOUS else 1

    self.curr_phase.add_tx(src, dst, rating)
    self.global_phase.add_tx(src, dst, rating)
    if self.curr_phase.is_full():
      self.prev_phase = self.curr_phase
      self.curr_phase = self.new_phase()
  # makes a private transaction of some good from dst to src (inverse of rating)
  def mk_tx(self, txr):
    node_kind = self.node_kind(txr)
    # malicious nodes will tend to have bad txs
    if node_kind is MALICIOUS: val = np.random.binomial(100, 0.1)/100
    # expert nodes and unbiased nodes will tend to have good txs
    elif node_kind is UNBIASED or node_kind is OMNISCIENT:
      val = np.random.binomial(100, 0.9)/100
    self.txs.setdefault(txr, []).append(val)
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
    if len(in_edges) < self.unknown_thresh: return random.random()#(tx + random.random())/2
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
    return [1] * self.num_unbiased + \
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
    bce_loss = F.binary_cross_entropy(got, exp)
    ys.append(bce_loss.item())
  plt.plot(xs, ys)
  plt.show()

def analysis_of_iterations_on_convergence(num_mal, num_unb=600, predictor=None):
  steps_per = 500
  xs = np.arange(0, 30_000, steps_per)
  ys = []
  sim = Simulator(
    size=1000, num_unbiased=num_unb, num_malicious=num_mal,
    predictor=predictor,
  )
  # warm up
  for i in range(500): sim.step()

  for m in tqdm(xs):
    for i in range(steps_per): sim.step()
    got = torch.tensor(sim.summary_stats())
    exp = torch.tensor(sim.expected(), dtype=torch.float)
    bce_loss = F.binary_cross_entropy(got, exp)
    ys.append(bce_loss.log().item())
  plt.plot(xs, ys, label=f"{num_mal} Malicious")

def main(predictor=None):
  #analysis_of_num_malicious()
  plt.title("Change in cross-entropy w.r.t. information propagation in the network")
  plt.xlabel("Steps")
  plt.ylabel("Log cross-entropy")
  analysis_of_iterations_on_convergence(400, predictor=predictor)
  analysis_of_iterations_on_convergence(300, predictor=predictor)
  analysis_of_iterations_on_convergence(200, predictor=predictor)
  analysis_of_iterations_on_convergence(100, predictor=predictor)
  analysis_of_iterations_on_convergence(000, predictor=predictor)
  analysis_of_iterations_on_convergence(1000,0, predictor=predictor)
  v = 10000
  rand = F.binary_cross_entropy(torch.rand(v), torch.rand(v)).log().item()
  plt.plot(
    np.arange(0, 30_000, 500), rand+np.zeros(30_000//500), label="assigning all random"
  )
  plt.legend()
  plt.show()

def run_with_predictor(): main(predictor=train_model())

if __name__ == "__main__":
  main()
  #run_with_predictor()
