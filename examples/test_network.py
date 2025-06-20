from collections import deque
from torchinspect.utils.flatten import flatten
import torch

zero = torch.zeros(1)
a = ( (zero, zero), zero)
b = [zero, zero, zero]
c = {"1": zero, "2":zero, "3": zero}

aq = flatten(a)
bq = flatten(b)
cq =  flatten(c)
print(f"aq {aq}, bq {bq}, cq {cq}")