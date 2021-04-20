import csv
import networkx as nx
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random as rand
import torch.nn.functional as F

#total # of transactions =35592
device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda:0")
NUM_EPOCH=90
FEATURES = 8
batch_size=256
def normalize(v: [-10, 10]) -> [0,1]: return (v+10)/20

# helper function
#add the integer incre to the [key1][key2]-entry of "dict"
def update(incre: int, dict, key1, key2):
    if key1 in dict:
        if key2 in dict[key1]:
            dict[key1][key2]=dict[key1][key2]+incre
        else:
            dict[key1][key2]=incre
    else: 
        dict[key1]={}
        dict[key1][key2]=incre
    return

#helper function
# return [key1][key2]-entry of "dict" if the entry is nonempty
def check_entry(dict, key1, key2):
    if key1 in dict:
        if key2 in dict[key1]:
            return dict[key1][key2]
    return 0

#splits data into batches
def split_data(featMat, labelVec): 
    #in general we use X for feature matrix, Y for target label vector
    num_batch = len(featMat)//batch_size
    num_test=num_batch//4
    num_train=num_batch-num_test
    
    split_indicator=np.append(np.zeros(num_test, dtype=int),np.ones(num_train, dtype=int))
    # if this is 0, assign the row to test data batch
    # if this is 1, assign the row to training data batch
    
    rand.shuffle(split_indicator)
    # we want to randomly assign rows to test/training data, so we do this by randomizing the indices
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    # this code should work, but this must be translated to the operations of tensors instead of lists
    # here, we actually assign data to batches
    for index in range(num_batch):
        if split_indicator[index] == 0:
            test_X.append(featMat[index*batch_size:(index+1)*batch_size,])
            test_Y.append(labelVec[index*batch_size:(index+1)*batch_size])
        else:
            train_X.append(featMat[index*batch_size:(index+1)*batch_size,])
            train_Y.append(labelVec[index*batch_size:(index+1)*batch_size])
    return num_train, num_test, torch.stack(train_X, dim=0), torch.stack(train_Y, dim=0), torch.stack(test_X, dim=0), torch.stack(test_Y, dim=0)
                
        
        
    
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
  G = nx.DiGraph()
    
  # feature vectors for each phase for each node
  feats = torch.zeros(phase_len * phases, FEATURES, device=device, dtype=torch.float)
  # How was this person rated on average before this phase?
  labels = torch.zeros(phase_len * phases, device=device, dtype=torch.float)
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
      n_rte[d] += 1
      G.add_node(s)
      G.add_node(d)
      for nbr in G.neighbors(s):
            update(r, nbhd_s, nbr, d)
            update(1, n_nbhd, nbr, d)
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
    hidden_size=32,
    features=8,
    out = 1,
  ):
    super().__init__()
    self.layer1 = nn.Linear(features, hidden_size)
    self.layer2 = nn.Linear(hidden_size, out)
    self.act = nn.LeakyReLU()
  def forward(self, feat_vecs: ["BATCH", "FEATURES"]):
    x = self.layer1(feat_vecs)
    y = self.layer2(self.act(x))
    return y

#this trains the model using the data from the csv file
class Trainer():
    def __init__(self,net=None,optim=None,loss_function=None, train_loader=None):
        self.net = net
        self.optim = optim
        self.loss_function = loss_function
        self.train_loader = train_loader

    def train(self,epochs): 
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            for data in self.train_loader:              
                X = data[..., :-1].to(device)
                y = data[...,  -1].to(device) 
                prediction = self.net(X)               
                loss = self.loss_function(prediction, y).mean()
                self.optim.zero_grad() 
                loss.backward()
                self.optim.step()
                epoch_loss += loss.item()
                epoch_steps += 1
            losses.append(epoch_loss / epoch_steps)
            #losses=(epoch_loss / epoch_steps)
            print("epoch [%d]: loss %.3f" % (epoch+1, losses[-1]))
        return losses

num_train, num_test, train_X, train_Y, test_X, test_Y = split_data(feats, labels)

#print(num_train,num_test,np.array(train_X).shape)#train_Y.shape,test_X.shape,test_Y.shape)

# B, Time, Features/Label
print(test_Y.unsqueeze(-1).shape, test_X.shape)
print(train_Y.unsqueeze(-1).shape, train_X.shape)
train_loader = torch.cat((train_Y.unsqueeze(-1),train_X), dim=2)
test_loader = torch.cat((test_Y.unsqueeze(-1), test_X), dim=2)
net = Predictor()
net = net.to(device) 
opt = optim.RMSprop(net.parameters(), lr=0.007)
loss_function = nn.PairwiseDistance()

trainer = Trainer(net=net, optim=opt, loss_function=loss_function, train_loader=train_loader)

losses = trainer.train(NUM_EPOCH)

#assert(losses[-1] < 0.03)
#assert(len(losses)==num_epochs)

for data in test_loader:
    X = data[..., :-1].to(device)
    y = data[..., -1].to(device)
    output = net(X)
    #print(output, y)
    #print(output-y)
    print(F.pairwise_distance(output, y).mean().item())
        
# Given some model, and a node "n", as well as a graph G which has prior ratings and timestamps,
# and outputs a predicted "trust" in [0,1].
def predict_trust(model, n, G):
  ...

# Given some model, and a node "n", as well as a graph G which has prior ratings and timestamps,
# and outputs a predicted "local trust" based on its neighbors ratings in [0,1].
def predict_local_trust(model, n, G):
  ...



