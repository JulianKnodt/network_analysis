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
FEATURES = 9
batch_size=256
num_of_copy=1
NUM_EPOCH=90*num_of_copy
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
    # we want to randomly assign rows to test/training data, so we do this by randomizing the indices
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    for copy in range(num_of_copy):
        rand.shuffle(split_indicator)
        for index in range(num_batch):
            if split_indicator[index] == 0:
                test_X.append(featMat[index*batch_size:(index+1)*batch_size,])
                test_Y.append(labelVec[index*batch_size:(index+1)*batch_size])
            else:
                train_X.append(featMat[index*batch_size:(index+1)*batch_size,])
                train_Y.append(labelVec[index*batch_size:(index+1)*batch_size])
    tr_X = torch.stack(train_X, dim=0)
    tr_Y = torch.stack(train_Y, dim=0)
    ts_X = torch.stack(test_X, dim=0)
    ts_Y = torch.stack(test_Y, dim=0)
    tr_X = tr_X[torch.randperm(tr_X.size()[0])]
    tr_Y = tr_Y[torch.randperm(tr_Y.size()[0])]
    ts_X = ts_X[torch.randperm(ts_X.size()[0])]
    ts_Y = ts_Y[torch.randperm(ts_Y.size()[0])]
    return num_train, num_test,tr_X, tr_Y, ts_X , ts_Y
    #this uses the trick from https://stackoverflow.com/questions/46826218/pytorch-how-to-get-the-shape-of-a-tensor-as-a-list-of-int
        
        
    
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
  labels = torch.zeros(phase_len * phases, device=device, dtype=torch.long)
  #total past ratings by the neighbors in this phase
  nbhd_s = {}
  #number of past ratings by the neighbors in this phase
  n_nbhd = {}
  #total past ratings by the neighbors in the whole history
  hist_nbhd_s = {}
  #number of past ratings by the neighbors in the whole history
  hist_n_nbhd = {}

  for p in range(0, phases):
    start = phase_len * p
    #rater's total ratings in this phase
    rtr_sums = [0] * max_node
    #ratee's total ratings in this phase
    rte_sums = [0] * max_node
    #rater's number of ratings in this phase
    n_rtr = [0] * max_node
    #ratee's number of ratings in this phase
    n_rte = [0] * max_node
    # same as above, but in the whole history
    hist_n_rtr = [0] * max_node
    hist_n_rte = [0] * max_node
    hist_rtr_sums = [0] * max_node
    hist_rte_sums = [0] * max_node
    # same as above, but in the previous phase
    prev_n_rtr = [0] * max_node
    prev_n_rte = [0] * max_node
    prev_rtr_sums = [0] * max_node
    prev_rte_sums = [0] * max_node
    for t in range(start, start+phase_len):
      s,d,r = src[t], dst[t], ratings[t]
      rtr_sums[s] += r
      n_rtr[s] += 1
      rte_sums[d] += r
      n_rte[d] += 1
      hist_rtr_sums[s] += r
      hist_n_rtr[s] += 1
      hist_rte_sums[d] += r
      hist_n_rte[d] += 1
      G.add_node(s)
      G.add_node(d)
      for nbr in G.neighbors(s):
            update(r, nbhd_s, nbr, d)
            update(1, n_nbhd, nbr, d)
            update(r, hist_nbhd_s, nbr, d)
            update(1, hist_n_nbhd, nbr, d)
      G.add_edge(s, d)
      #this should not create duplicates (see MultiDiGraph)
      total_ratings += r
      n_total_ratings += 1
      feats[t, 0] = total_ratings/ n_total_ratings
      feats[t, 1] = rtr_sums[s]/n_rtr[s]
      feats[t, 2] = rte_sums[d]/n_rte[d]
      feats[t, 3] = check_entry(nbhd_s,s,d)/(check_entry(n_nbhd,s,d)+epsilon)
      feats[t, 4] = hist_rtr_sums[s]/hist_n_rtr[s]
      feats[t, 5] = hist_rte_sums[d]/hist_n_rte[d]
      feats[t, 6] = check_entry(hist_nbhd_s,s,d)/(check_entry(hist_n_nbhd,s,d)+epsilon)
      feats[t, 7] = prev_rtr_sums[s]/(prev_n_rtr[s]+epsilon)
      feats[t, 8] = prev_rte_sums[d]/(prev_n_rte[d]+epsilon)
      labels[t] = round(r)
    for person in range(max_node):
      prev_rtr_sums[person] =rtr_sums[person]
      prev_rte_sums[person] =rte_sums[person]
      prev_n_rte[person] =n_rte[person]
      prev_n_rtr[person] =n_rtr[person]
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
    features=FEATURES,
    out =2,
  ):
    super().__init__()
    self.layer1 = nn.Linear(features, hidden_size)
    self.layer2 = nn.Linear(hidden_size, hidden_size)
    self.layer3 = nn.Linear(hidden_size, out)
    self.act = nn.LeakyReLU()
  def forward(self, feat_vecs: ["BATCH", "FEATURES"]):
    x = self.layer1(self.act(feat_vecs))
    y = self.layer2(self.act(x))
    z = self.layer3(self.act(y))
    return z
  def predict(self, x):
    y = forward(x)
    likeliness_indicator = F.softmax(self.forward(x))
    output = []
    for t in likeliness_indicator:
        if t[0]>t[1]:
            output.append(0)
        else:
            output.append(1)
    return torch.tensor(output)
#reference for the implementation above: https://medium.com/@prudhvirajnitjsr/simple-classifier-using-pytorch-37fba175c25c

#this trains the model using the data from the csv file
class Trainer():
    def __init__(self,net=None, optim=None,loss_function=None, train_loader_X=None, train_loader_Y=None):
        self.net = net
        self.optim = optim
        self.loss_function = loss_function
        self.train_loader_X = train_loader_X
        self.train_loader_Y = train_loader_Y

    def train(self,epochs): 
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            for data in range(len(self.train_loader_X)):              
                X = self.train_loader_X[data].to(device)
                y = self.train_loader_Y[data].to(device)
                prediction = self.net(X)               
                loss = self.loss_function(prediction.squeeze(-1), y.squeeze(-1)).mean()
                self.optim.zero_grad() 
                loss.backward()
                self.optim.step()
                epoch_loss += loss.item()
                epoch_steps += 1            
            losses.append(epoch_loss / epoch_steps)
            #losses=(epoch_loss / epoch_steps)
            print("epoch [%d]: loss %.3f" % (epoch+1, losses[-1]))
        return losses

num_train, num_test, train_loader_X, train_loader_Y, test_X, test_Y = split_data(feats, labels)

#print(num_train,num_test,np.array(train_X).shape)#train_Y.shape,test_X.shape,test_Y.shape)

# B, Time, Features/Label
#print(test_Y.unsqueeze(-1).shape, test_X.shape)
#print(train_Y.unsqueeze(-1).shape, train_X.shape)
net = Predictor()
net = net.to(device) 
opt = optim.RMSprop(net.parameters(), lr=0.03)
#loss_function = nn.BCEWithLogitsLoss()
loss_function = nn.CrossEntropyLoss()
trainer = Trainer(net=net, optim=opt, loss_function=loss_function, train_loader_X=train_loader_X, train_loader_Y=train_loader_Y)

losses = trainer.train(NUM_EPOCH)


for data in range(len(test_X)):
    #print(test_X.size())
    #print(test_X[data].size())
    #print(test_X[data,].size())
    X = test_X[data,].to(device)
    y = test_Y[data,].to(device)
    output = net(X)
    print("output")
    print(output[:10])
    print("y")
    print(y[:10])
    print(F.binary_cross_entropy_with_logits(output.squeeze(-1), y).mean().item())
        
# Given some model, and a node "n", as well as a graph G which has prior ratings and timestamps,
# and outputs a predicted "trust" in [0,1].
def predict_trust(model, n, total_ratings, n_total_ratings, rtr_sums, rte_sums, nbhd_s,n_nbhd, hist_rtr_sums, hist_n_rtr, hist_rte_sums, \
                  hist_n_rte, hist_nbhd_s,hist_n_nbhd, prev_rte_sums,prev_n_rte,prev_rtr_sums,prev_n_rtr):
    feats = [0]*FEATURES
    feats[ 0] = total_ratings/ n_total_ratings
    feats[ 1] = rtr_sums[s]/n_rtr[s]
    feats[ 2] = rte_sums[d]/n_rte[d]
    feats[ 3] = check_entry(nbhd_s,s,d)/(check_entry(n_nbhd,s,d)+epsilon)
    feats[ 4] = hist_rtr_sums[s]/hist_n_rtr[s]
    feats[ 5] = hist_rte_sums[d]/hist_n_rte[d]
    feats[ 6] = check_entry(hist_nbhd_s,s,d)/(check_entry(hist_n_nbhd,s,d)+epsilon)
    feats[ 7] = prev_rtr_sums[s]/(prev_n_rtr[s]+epsilon)
    feats[ 8] = prev_rte_sums[d]/(prev_n_rte[d]+epsilon)
    return   model.net(feats)