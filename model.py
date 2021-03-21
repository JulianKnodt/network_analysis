import csv
import networkx as nx

def init_graph():
  G = nx.DiGraph()
  with open("soc-sign-bitcoinotc.csv") as f:
    reader = csv.reader(f)
    for src, dst, w, time in reader:
      G.add_node(src)
      G.add_node(dst)

      G.add_edge(src, dst, weight=(float(w)+10)/20, time=time)
  return G

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

G = init_graph()
rater_summaries(G)

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

