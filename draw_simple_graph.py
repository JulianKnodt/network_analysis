import csv
import os
import random


print("digraph show {")
f = open("soc-sign-bitcoinotc.csv")
reader = csv.reader(f)
for src, dst, w, time in reader:
  if random.random() > 0.90:
    w = int(w)
    assert(w != 0);
    w = w/10
    if w < 0:
      r,b=255*abs(w),0
    else:
      r,b=0,255*w
    r = int(min(r, 255))
    b = int(min(b, 255))
    print(f"{src} -> {dst} [color=\"#{r:02x}00{b:02x}\", weight={int(abs(w)*10)}];")

print("}")
