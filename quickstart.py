# here we are learning to do basic tensor operations
# these are the building blocks to make ML models
# multivariate optimizations are of incredble import
# https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
#

import torch
import numpy as np


tensor = torch.ones(4, 4)

tensor[:, 1] = 0

# print(tensor)

# print('-----------------')  # line break

# arithmetic operations
# matrix multiplu between two tensors

y1 = tensor @ tensor.T

y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)

torch.matmul(tensor, tensor.T, out=y3)

# single element tensors

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))


