import csv
import networkx as nx
    
#T: total time, step: duration of each phase 
#this ignores that T may not be divisible by step, but can throw away some last bit if we want  
def increment(T, step, src, dst, w, time):
    #assume that src etc are given as vectors
    #output variables;;;;ratee(r)'s average rating 
    avg_ratee_out[], num_ratee_out[]
    #output variables;;;ratee(r)'s number of ratings 
    avg_rater_out[], num_rater_out[]
    #output variables;;;rateer's neighbors' average/number of ratings of some fixed ratee
    avg_nbhd_out[][], num_nbhd_out[][] 
    # output variable;
    avg_global_out[]
    # output variable;
    sum_global_out[]
    
    #ratee(r)'s average rating with phase=(something)
    sum_ratee[][],sum_rater[][]
    #ratee(r)'s number of ratings with phase=(something)
    num_rater[][],num_rater[][]
    #rater's neighbors' total ratings of some fixed ratee;;;this does not take "phase" as argument
    #rater's neighbors' number of some fixed ratee;;;this does not take "phase" as argument
    sum_nbhd[][],num_nbhd[][] 
    #adjacency list? need an array of <set> for example adj[person] is the adjency list of person
    adj
    #(global) total ratings 
    sum_global 
    # (global) total number of transactions
    num_global
    
    # it's actually better to iterate over current and ise phase=current/step
    
    for phase in range(0,T/step):
        for t in range(0,step):
            current = phase*step+t
            sum_rater[src[current]][phase]+=w[current]
            num_rater[src[current]][phase]++
            sum_ratee[dst[current]][phase]+=w[current]
            num_ratee[dst[current]][phase]++
            num_global++
            sum_global+=w[current]
            for neighbor in adj[src[current]]:
                sum_nbhd[neighbor][dst[current]]+=w[current]
                num_nbhd[neighbor][dst[current]]++
            #not sure if it is the right method
            #also try not to make a suplicate object
            adj[src[current]].add(dst[current])
            adj[dst[current]].add(src[current])
            
            avg_ratee_out[current]=sum_ratee[dst[current]][phase]/num_ratee[dst[current]][phase]
            num_ratee_out[current]=num_ratee[dst[current]][phase]
            avg_rater_out[current]=sum_rater[src[current]][phase]/num_rater[src[current]][phase]
            num_rater_out[current]=num_rater[src[current]][phase]
            avg_nbhd_out[current]=sum_nbhd[neighbor][dst[current]]/num_nbhd[neighbor][dst[current]]
            num_nbhd_out[current]=num_nbhd[neighbor][dst[current]]
            avg_global_out[current]=sum_global/num_global
            num_global_out[current]=num_global
    return avg_ratee_out,num_ratee_out,avg_rater_out,num_rater_out,avg_nbhd_out,num_nbhd_out,avg_global_out,num_global_out
    
    #prev is the number of previous phases to consider
def produce_matrix(T,step, prev, step, src, dst, w, time):
    avg_ratee_out,num_ratee_out,avg_rater_out,num_rater_out,avg_nbhd_out,num_nbhd_out,avg_global_out,num_global_out=increment(T, step, src, dst, w, time)
    inputmatrix
    # is this quotient evaluated as int??? I don't rmbr
    for t in range (0, (T/step)*step):
        # I want to concatenate but not sure if this is a valid method;;;also need to put entries as 0 if out_of_bound 
        # this constructs the input matrix (y,X) for regression/learning 
        inputmatrix[t] =w[t]+avg_ratee_out[T-prev:t]+num_ratee_out[T-prev:t]+avg_rater_out[T-prev:t]+num_rater_out[T-prev:t]+avg_nbhd_out[T-prev:t]+num_nbhd_out[T-prev:t]+num_global[t]+sum_global[t]
    return inputmatrix

def init_graph():
  G = nx.DiGraph()
  with open("soc-sign-bitcoinotc.csv") as f:
    reader = csv.reader(f)
    for src, dst, w, time in reader:
      G.add_node(src)
      G.add_node(dst)

      G.add_edge(src, dst, weight=normalize(w), time=time)
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

