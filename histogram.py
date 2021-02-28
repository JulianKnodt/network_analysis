import csv
import os
import random
import numpy as np
import matplotlib.pyplot as plt


f = open("soc-sign-bitcoinotc.csv")
reader = csv.reader(f)
vs = np.array([0] * 21)
give = {}
num_give = {}
rcv = {}
num_rcv = {}
for src, dst, w, time in reader:
  w = int(w)
  vs[w + 10] += 1

  give[src] = getattr(give, src, 0) + w
  num_give[src] = getattr(give, src, 0) + 1

  rcv[dst] = getattr(rcv, dst, 0) + w
  num_rcv[dst] = getattr(rcv, dst, 0) + 1

plt.bar(np.arange(-10, 11), vs)
plt.savefig("raw_weights.png")
plt.clf()

avg_give = np.array([
  total/num_give[k] for k, total in give.items()
])

avg_rcv = np.array([
  total/num_rcv[k] for k, total in rcv.items()
])

plt.hist(avg_give, label="Rater")
plt.hist(avg_rcv, label="Ratee", alpha=0.5, width=1)
plt.legend()
plt.savefig("average.png")
