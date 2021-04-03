import csv
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda:0")

def normalize(v: [-10, 10]) -> [0,1]: return (v+10)/20

def update(incre: int, dict, key1, key2):
    if key1 in dict:
        if key2 in dict[key1]:
            dict[key1][key2]=dict[key1][key2]+incre
        else:
            dict[key1][key2]=incre
    else: 
        dict[key1][key2]=incre
    return

def check_entry(dict, key1, key2):
    if key1 in dict:
        if key2 in dict[key1]:
            return dict[key1][key2]
    return 0

def feature_vectors(max_node: int, src: [int], dst: [int], ratings: [float], phase_len=3000):
  assert(len(src) == len(ratings))
  assert(len(dst) == len(ratings))
  print(len(src))
  G = nx.DiGraph()
  T = len(ratings)
  rater_sums = {}
  n_raters = {}
  ratee_sums = {}
  n_ratees = {}
  total_ratings = 0
  n_total_ratings = 0
  epsilon = 1.e-17

  phases = T//phase_len
  FEATURES = 5

  G = nx.DiGraph()
    
  # feature vectors for each phase for each node
  feats = torch.zeros(max_node, phases, FEATURES, device=device, dtype=torch.float)
  # How was this person rated on average before this phase?
  labels = torch.zeros(max_node, phases, device=device, dtype=torch.float)
  nbhd_s = {}
  n_nbhd = {}
  for p in range(0, phases):
    start = phase_len * p
    rtr_sums = [0] * max_node
    rte_sums = [0] * max_node
    n_rtr = [0] * max_node
    n_rte = [0] * max_node
    for t in range(start, start+phase_len):
      s,d,r = src[t], dst[t], ratings[t]
      rtr_sums[s] += r
      n_rtr[s] += 1
      rte_sums[d] += r
<<<<<<< HEAD
      n_rte[d] += 1
      #for nbr in G.adj[s]
      #  nbhd_s[nbr][d] += w
      #  n_nbhd[nbr][d] += 1
      #G.add_edge(s, d, weight=r)
    # TODO do the rest
    total_ratings += r
    for n in range(0, max_node):
      feats[n, p, 0] = rtr_sums[n]/n_rtr[n] if n_rtr[n] != 0 else 0
      feats[n, p, 1] = total_ratings
      # current label is the average of its ratings
      labels[n, p] = rte_sums[n]/n_rte[n] if n_rte[n] != 0 else 0
=======
      n_rtr[d] += 1
      for nbr in G.neighbors(n):
        update(w, nbhd_s, nbr, d)
        update(1, n_nbhd, nbr, d)
      G.add_node(s)
      G.add_node(d)
      G.add_edge(s, d)
      #this should not create duplicates (see MultiDiGraph)
      total_ratings += r
      n_total_ratings += 1
      feats[t, 0] = rtr_sums[s]/n_rtr[s]
      feats[t, 1] = n_rtr[s]
      feats[t, 2] = total_ratings/ n_total_ratings
      feats[t, 3] = n_total_ratings
      feats[t, 4] = rte_sums[d]/n_rte[d]
      feats[t, 5] = n_rte[d]
      feats[t, 6] = check_entry(nbhd_s,s,d)/(check_entry(n_nbhd,s,d)+epsilon)
      feats[t, 7] = check_entry(n_nbhd,s,d)
      labels[t] = r
>>>>>>> added more featuer variables
  return feats, labels

def init_graph():
  G = nx.DiGraph()
  max_node = 0
  with open("soc-sign-bitcoinotc.csv") as f:
    reader = csv.reader(f)
    for src, dst, w, time in reader:
      G.add_node(src)
      G.add_node(dst)
      G.add_edge(src, dst, weight=normalize(w), time=time)
      max_node = max(src, max(dst, max_node))
  return G, max_node

def get_vectors():
  with open("soc-sign-bitcoinotc.csv") as f:
    reader = csv.reader(f)
    srcs = []
    dsts = []
    ratings = []
    max_node = 0
    for src, dst, w, time in reader:
      src = int(src)
      dst = int(dst)
      w = float(w)
      srcs.append(src)
      dsts.append(dst)
      ratings.append(normalize(w))
      max_node = max(src, max(dst, max_node))
  return srcs, dsts, ratings, max_node

srcs, dsts, ratings, max_node = get_vectors()

feats, labels = feature_vectors(max_node, srcs, dsts, ratings)

# returns a feature matrix for all nodes in the graph
def rater_summaries(G):
  out = []
  #path_lens = list(nx.all_pairs_dijkstra_path_length(G))
  for n in G.nodes:
    outgoing = G.adj[n]
    avg_rating = 0
    if len(outgoing) != 0:
      avg_rating = sum(outgoing[x]['weight'] for x in outgoing)/len(outgoing)
    out.append([
      avg_rating,
    ])
  return out

# A very basic predictor model with two linear layers.
class Predictor(nn.Module):
  def __init__(
    self,
    hidden_size=64,
    features=5,
    out = 1,
  ):
    super().__init__()
    self.layer1 = nn.Linear(features, hidden_size)
    self.layer2 = nn.Linear(hidden_size, out)
    self.act = nn.LeakyReLU()
  def forward(self, feat_vecs: ["BATCH", "FEATURES"]):
    x = self.layer1(feat_vecs)
    y = self.layer2(self.act(feat_vecs))
    return y

# Given some model, and a node "n", as well as a graph G which has prior ratings and timestamps,
# and outputs a predicted "trust" in [0,1].
def predict_trust(model, n, G):
  ...

# Given some model, and a node "n", as well as a graph G which has prior ratings and timestamps,
# and outputs a predicted "local trust" based on its neighbors ratings in [0,1].
def predict_local_trust(model, n, G):
  ...

class Simulator():
  def __init__(
    self,
    size = 1000,
    num_malicious = 50,
    num_unbiased = 800,
  ):
    self.G = nx.DiGraph()
    self.t = 0

    assert(size >= num_malicious + num_unbiased)
    num_omniscient = size - num_malicious + num_unbiased

    self.update_fn = None
  def step(self):
    # TODO add something here for adding a transaction after some random time
    ...
